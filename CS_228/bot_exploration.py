# %%
# %pip install -q minigrid gymnasium transformers accelerate torch sentencepiece


# %%
# # %pip install --upgrade torch torchvision torchaudio


# %%
import torch
print(torch.__version__)


# %%
import json
from pathlib import Path
import time
from datetime import datetime, timezone
from collections import deque

import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym
from minigrid.core.actions import Actions
from minigrid.wrappers import FullyObsWrapper
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import seaborn as sns
from tqdm import tqdm
import subprocess
from openai import OpenAI

# Use non-interactive backend for Modal/Server
import matplotlib
matplotlib.use('Agg')


# %%
MODEL_CANDIDATES = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]
MAX_NEW_TOKENS = 16
TEMPERATURE = 0.1
MOCK_MODE = False
USE_VLLM = True # Set to False for native HF transformers
VLLM_PORT = 8000
HISTORY_WINDOWS = (0, 1, 2)
ENVIRONMENTS = [
    "MiniGrid-LavaGapS7-v0",
    # "MiniGrid-BlockedUnlockPickup-v0",
    # "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-Empty-8x8-v0",
]
N_EPISODES_LIST = (10,)
MAX_STEPS_LIST = (100,)
BUFFER_SIZES = (2, 3, 5)
OUTPUT_ROOT = "CS 228/outputs"
RUN_TAG = "bot_sweep"


# %%
from dataclasses import dataclass

@dataclass
class ThoughtTemplate:
    name: str
    description: str
    reasoning_pattern: str
    usage_count: int = 0
    success_rate: float = 0.0

    def instantiate(self, distilled_obs: dict) -> str:
        """Process the template with local facts into a concrete plan."""
        context_parts = []
        if distilled_obs.get("agent_pos"):
            context_parts.append(f"You are at {distilled_obs['agent_pos']} facing {distilled_obs.get('facing', 'unknown')}.")
        if distilled_obs.get("target_pos"):
            context_parts.append(f"Target is at {distilled_obs['target_pos']}.")
            
        front = distilled_obs.get("front_object")
        if front:
            context_parts.append(f"Right in front of you is a {front}.")
        
        nearby = distilled_obs.get("nearby_objects", [])
        if nearby:
            context_parts.append(f"Nearby objects: {', '.join(nearby)}.")
            
        context_str = " ".join(context_parts)
        return f"[TEMPLATE: {self.name}]\nContext: {context_str}\nPattern: {self.reasoning_pattern}"

ALL_TEMPLATES = [
    ThoughtTemplate(
        name="Direct Navigation",
        description="Moving towards a visible target in an empty space.",
        reasoning_pattern="1. Locate target coordinates. 2. Identify relative direction. 3. Align orientation. 4. Move forward.",
    ),
    ThoughtTemplate(
        name="Obstacle Avoidance",
        description="Navigating around walls, lava, or closed doors.",
        reasoning_pattern="1. Identify blocking object. 2. Scan for open path. 3. Plan detour. 4. Execute lateral move.",
    ),
    ThoughtTemplate(
        name="Object Acquisition",
        description="Locating and picking up a key or other inventory item.",
        reasoning_pattern="1. Scan grid for target object (key/ball). 2. Navigate to the object. 3. Face the object. 4. Use pickup action.",
    ),
    ThoughtTemplate(
        name="Unlock Door",
        description="Using a held key to open a locked door.",
        reasoning_pattern="1. Confirm key is in inventory. 2. Navigate to the locked door. 3. Face the door. 4. Use toggle action to unlock and open.",
    ),
    ThoughtTemplate(
        name="Clear Path",
        description="Moving blocking objects out of the way to reach a target.",
        reasoning_pattern="1. Identify the blocking object. 2. Face the blocker. 3. Pickup or push the blocker aside. 4. Drop it in a clear cell. 5. Return to original objective.",
    ),
]

class MetaBuffer:
    def __init__(self, buffer_size=4):
        self.buffer_size = buffer_size
        self.templates = {}  # name -> ThoughtTemplate
        self._learn_counter = 0
        self._initialize_templates(buffer_size)

    def _initialize_templates(self, buffer_size):
        for t in ALL_TEMPLATES[:buffer_size]:
            self.add_template(t)

    def add_template(self, template: ThoughtTemplate):
        self.templates[template.name] = template

    def retrieve(self, problem_description: dict):
        desc = str(problem_description).lower()
        if not self.templates:
            return None
            
        best_template = None
        max_score = -1.0
        
        for t_name, template in self.templates.items():
            # If a template has never been used, pretend it has a 0.5 success rate
            # so it isn't completely ignored against slightly failing templates
            base_score = template.success_rate if template.usage_count > 0 else 0.5
            
            # Keyword Bonus matches original heuristic checks
            keyword_bonus = 0.0
            if "Obstacle Avoidance" in t_name and "lava" in desc:
                keyword_bonus = 1.0
            elif "Unlock Door" in t_name and ("door" in desc or "key" in desc):
                keyword_bonus = 1.0
            elif "Object Acquisition" in t_name and ("pickup" in desc or "ball" in desc):
                keyword_bonus = 1.0
            elif "Direct Navigation" in t_name and not any(k in desc for k in ["lava", "door", "key", "pickup", "ball"]):
                keyword_bonus = 1.0
                
            score = base_score + keyword_bonus
            
            if score > max_score:
                max_score = score
                best_template = template
                
        return best_template

    def update_stats(self, template_name, success):
        template = self.templates.get(template_name)
        if template:
            template.usage_count += 1
            # Moving average for success rate
            old_success = template.success_rate
            template.success_rate = old_success + (float(success) - old_success) / template.usage_count

    def learn(self, action_log: list, llm_client) -> bool:
        """Replace a consistently failing template with a new learned strategy."""
        failing_template = None
        for t_name, t in self.templates.items():
            if t.usage_count >= 3 and t.success_rate < 0.2:
                failing_template = t_name
                break
                
        if not failing_template:
            return False # Nothing needs replacing
            
        # Distill successful trajectory
        trajectory = "\n".join([f"{entry['observation']} -> {entry['parsed_action']}" for entry in action_log])
        system_prompt = "You are a MiniGrid expert. Summarize the following successful action sequence into a 3-4 step generic reasoning pattern."
        user_prompt = f"Sequence:\n{trajectory}\n\nProvide ONLY the numbered reasoning steps."
        
        response, _, _ = llm_client.query(system_prompt, user_prompt)
        
        self._learn_counter += 1
        new_name = f"Learned Strategy {self._learn_counter}"
        new_template = ThoughtTemplate(
            name=new_name,
            description="Dynamically learned from successful episode.",
            reasoning_pattern=response.strip()
        )
        
        # Replace
        del self.templates[failing_template]
        self.add_template(new_template)
        print(f"\n🧠 [Buffer Learning] Replaced {failing_template} with new learned strategy: {new_name}")
        return True

import re
class ProblemDistiller:
    @staticmethod
    def distill(obs_text: str) -> dict:
        """Parse raw text observation into a structured dictionary."""
        distilled = {
            "agent_pos": None,
            "facing": None,
            "target_pos": None,
            "front_object": None,
            "nearby_objects": []
        }
        
        # Extract agent pos
        agent_match = re.search(r"Agent is at \[(\d+),\s*(\d+)\] facing (\w+)", obs_text)
        if agent_match:
            distilled["agent_pos"] = [int(agent_match.group(1)), int(agent_match.group(2))]
            distilled["facing"] = agent_match.group(3)
            
        # Extract target pos
        target_match = re.search(r"(?:Goal|Target).*?is at \[(\d+),\s*(\d+)\]", obs_text)
        if target_match:
            distilled["target_pos"] = [int(target_match.group(1)), int(target_match.group(2))]
            
        # Extract front object
        front_match = re.search(r"In front of you is a (.*?)\.", obs_text)
        if front_match:
            distilled["front_object"] = front_match.group(1).strip()
            
        # Extract nearby objects
        nearby_match = re.search(r"Nearby objects:\s*(.*?)\.", obs_text)
        if nearby_match:
            objects = re.findall(r"([^,]+? at \[\d+,\s*\d+\])", nearby_match.group(1))
            distilled["nearby_objects"] = [o.strip() for o in objects]
            
        return distilled

class BufferManager:
    def __init__(self, buffer_size=2):
        self.meta_buffer = MetaBuffer(buffer_size=buffer_size)
        self.thought_history = deque(maxlen=50)



# %%
class MinigridTextWrapper:
    def __init__(self, env_id, render_mode=None):
        self.env = gym.make(env_id, render_mode=render_mode)
        self.env = FullyObsWrapper(self.env)
        self.action_map = {
            "turn_left": Actions.left,
            "turn_right": Actions.right,
            "forward": Actions.forward,
            "pickup": Actions.pickup,
            "drop": Actions.drop,
            "toggle": Actions.toggle,
            "done": Actions.done
        }

    def _base_env(self):
        return self.env.unwrapped

    def _find_goal_pos(self):
        base = self._base_env()
        grid = base.grid
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None and getattr(cell, "type", None) == "goal":
                    return (int(x), int(y))
        return None

    def _scan_grid_objects(self):
        """Scan the full grid for notable objects and their positions."""
        base = self._base_env()
        grid = base.grid
        objects = []
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is None:
                    continue
                obj_type = getattr(cell, 'type', None)
                if obj_type in ("key", "door", "box", "ball", "lava"):
                    state = ""
                    if obj_type == "door":
                        if getattr(cell, 'is_locked', False):
                            state = " (locked)"
                        elif getattr(cell, 'is_open', False):
                            state = " (open)"
                        else:
                            state = " (closed)"
                    color = getattr(cell, "color", "")
                    label = f"{color} {obj_type}".strip() if color else obj_type
                    objects.append(f"{label}{state} at [{x}, {y}]")
        return objects

    def _find_fallback_target(self):
        """For envs with no goal tile, find a box or ball as target."""
        base = self._base_env()
        grid = base.grid
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None and getattr(cell, "type", None) in ("box", "ball"):
                    return (int(x), int(y)), getattr(cell, 'type', 'box')
        return None, None

    def get_text_obs(self, obs):
        base = self._base_env()

        if hasattr(base, "agent_pos") and base.agent_pos is not None:
            ax, ay = int(base.agent_pos[0]), int(base.agent_pos[1])
            agent_pos_text = f"[{ax}, {ay}]"
        else:
            agent_pos_text = "None"

        agent_dir = int(base.agent_dir) if hasattr(base, "agent_dir") else None
        goal_pos = self._find_goal_pos()

        if goal_pos is not None:
            target_text = f"Goal is at [{goal_pos[0]}, {goal_pos[1]}]."
        else:
            fallback_pos, fallback_type = self._find_fallback_target()
            if fallback_pos is not None:
                target_text = f"Target ({fallback_type}) is at [{fallback_pos[0]}, {fallback_pos[1]}]."
            else:
                target_text = "No target found."

        dirs = ["right", "down", "left", "up"]
        facing = dirs[agent_dir] if agent_dir is not None and 0 <= agent_dir < 4 else "unknown"
        desc = f"Agent is at {agent_pos_text} facing {facing}. {target_text}"

        front_obj = None
        if hasattr(base, "front_pos") and hasattr(base, "grid"):
            fx, fy = int(base.front_pos[0]), int(base.front_pos[1])
            front_obj = base.grid.get(fx, fy)

        if front_obj:
            desc += f" In front of you is a {front_obj.type}."
        else:
            desc += " The path in front is clear."

        grid_objects = self._scan_grid_objects()
        if grid_objects:
            desc += " Nearby objects: " + ", ".join(grid_objects) + "."

        if hasattr(base, "carrying") and base.carrying is not None:
            desc += f" You are carrying a {base.carrying.type}."

        return desc

    def reset(self, seed=None):
        reset_out = self.env.reset(seed=seed) if seed is not None else self.env.reset()
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, _ = reset_out
        else:
            obs = reset_out
        return self.get_text_obs(obs)

    def step(self, action_str):
        action = self.action_map.get(action_str.lower(), Actions.forward)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return self.get_text_obs(obs), reward, done, info


# %%
class VLLMServer:
    def __init__(self, model_name, port=8000):
        self.model_name = model_name
        self.port = port
        self.process = None

    def start(self):
        print(f"🚀 Starting vLLM server for model: {self.model_name} on port {self.port}...")
        command = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--port", str(self.port),
            "--gpu-memory-utilization", "0.8", # Leave some room for other things
            "--disable-log-requests"
        ]
        self.process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        # Wait for server to be ready
        import requests
        url = f"http://localhost:{self.port}/v1/models"
        max_retries = 40
        for i in range(max_retries):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print("✅ vLLM server is ready!")
                    return True
            except:
                pass
            if i % 5 == 0:
                print(f"Waiting for vLLM server... ({i}/{max_retries})")
            time.sleep(10)
        
        print("❌ vLLM server failed to start.")
        self.stop()
        return False

    def stop(self):
        if self.process:
            print(f"Stopping vLLM server for {self.model_name}...")
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            # Allow GPU memory to clear
            time.sleep(5)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# %%
class LLMClient:
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct", max_new_tokens=16, temperature=0.0, mock=False, use_vllm=False):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.mock = mock
        self.use_vllm = use_vllm
        self.total_tokens = 0
        self.total_latency = 0
        self.model = None
        self.tokenizer = None
        self.client = None

        if not self.mock:
            if self.use_vllm:
                self._init_vllm_client()
            else:
                self._load_hf_model()

    def _init_vllm_client(self):
        self.client = OpenAI(
            base_url=f"http://localhost:{VLLM_PORT}/v1",
            api_key="empty"
        )
        print(f"vLLM client initialized for port {VLLM_PORT}")

    def _load_hf_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"Loaded model: {self.model_name}")

    def query(self, system_prompt, user_prompt):
        start_time = time.time()

        if self.mock:
            response = self._mock_logic(user_prompt)
            tokens = len(system_prompt + user_prompt) // 4
        elif self.use_vllm:
            response, tokens = self._query_vllm(system_prompt, user_prompt)
        else:
            response, tokens = self._query_hf(system_prompt, user_prompt)

        latency = time.time() - start_time
        self.total_tokens += tokens
        self.total_latency += latency

        return response, tokens, latency

    def _query_vllm(self, system_prompt, user_prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        text = response.choices[0].message.content.strip()
        tokens = response.usage.completion_tokens
        return text, tokens

    def _query_hf(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_tokens = inputs["input_ids"].shape[-1]

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = 0.9

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated = outputs[0][input_tokens:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        tokens = int(generated.shape[-1])
        return response, tokens

    @staticmethod
    def _extract_state_from_prompt(prompt):
        import re

        coord_pat = r"[\[\(]\s*(\d+)\s*,\s*(\d+)\s*[\]\)]"
        pos_match = re.search(rf"Agent is at {coord_pat} facing (\w+)", prompt)
        goal_match = re.search(rf"Goal is at {coord_pat}", prompt)
        if goal_match is None:
            goal_match = re.search(rf"Target \(\w+\) is at {coord_pat}", prompt)

        if not pos_match or not goal_match:
            return None

        ax, ay = int(pos_match.group(1)), int(pos_match.group(2))
        face = pos_match.group(3)
        gx, gy = int(goal_match.group(1)), int(goal_match.group(2))
        return ax, ay, face, gx, gy

    @staticmethod
    def _mock_policy_action(ax, ay, face, gx, gy):
        if face == "right":
            if gx > ax:
                return "forward"
            return "turn_left" if gy < ay else "turn_right"
        if face == "left":
            if gx < ax:
                return "forward"
            return "turn_right" if gy < ay else "turn_left"
        if face == "up":
            if gy < ay:
                return "forward"
            return "turn_right" if gx > ax else "turn_left"
        if face == "down":
            if gy > ay:
                return "forward"
            return "turn_left" if gx > ax else "turn_right"
        return "forward"

    def _mock_logic(self, prompt):
        state = self._extract_state_from_prompt(prompt)
        if state is None:
            return "forward"

        ax, ay, face, gx, gy = state
        return self._mock_policy_action(ax, ay, face, gx, gy)


# %%
class BoTAgent:
    def __init__(self, llm_client, buffer_manager, use_history=True, history_window=3):
        self.llm = llm_client
        self.buffer_manager = buffer_manager
        self.use_history = use_history
        self.history_window = history_window
        self.memory = []
        self.action_log = []
        self.used_templates = []

    @staticmethod
    def _parse_action(response_text):
        action = response_text.strip().lower()
        if "forward" in action:
            return "forward"
        if "left" in action:
            return "turn_left"
        if "right" in action:
            return "turn_right"
        if "toggle" in action:
            return "toggle"
        if "pickup" in action:
            return "pickup"
        if "drop" in action:
            return "drop"
        return "forward"

    def act(self, observation):
        distilled = ProblemDistiller.distill(observation)
        template = self.buffer_manager.meta_buffer.retrieve(distilled)
        if template:
            self.used_templates.append(template.name)
            aid = template.instantiate(distilled)
        else:
            aid = ""

        system_prompt = "You are a MiniGrid navigation agent. Choose exactly one action token: forward, turn_left, turn_right, toggle, pickup, drop."
        history_context = self.memory[-self.history_window:] if self.use_history else []
        user_prompt = (
            f"Current Obs: {observation}\n"
            f"{aid}\n"
            f"Recent history: {history_context}\n"
            "Return only one action token."
        )

        response, tokens, latency = self.llm.query(system_prompt, user_prompt)
        action = self._parse_action(response)

        self.memory.append((observation, action))
        self.action_log.append({"raw_llm_output": response, "parsed_action": action, "observation": observation})
        return action


def _resolve_env_id(candidates):
    for env_id in candidates:
        try:
            test_env = gym.make(env_id)
            test_env.close()
            return env_id
        except Exception:
            continue
    return None

def _resolve_environment_pairs(environments):
    env_candidate_map = {
        "MiniGrid-LavaGapS7-v0": ["MiniGrid-LavaGapS7-v0"],
        "MiniGrid-BlockedUnlockPickup-v0": ["MiniGrid-BlockedUnlockPickup-v0", "MiniGrid-UnlockPickup-v0"],
        "MiniGrid-DoorKey-5x5-v0": ["MiniGrid-DoorKey-5x5-v0"],
        "MiniGrid-Empty-8x8-v0": ["MiniGrid-Empty-8x8-v0"],
    }
    resolved_envs = []
    for env_name in environments:
        candidates = env_candidate_map.get(env_name, [env_name])
        resolved = _resolve_env_id(candidates)
        if resolved:
            resolved_envs.append((env_name, resolved))
    return resolved_envs

def _run_single_episode(env, agent, max_steps, seed=None):
    obs = env.reset(seed=seed) if seed is not None else env.reset()
    done = False
    steps = 0
    total_reward = 0.0
    
    goal_pos = env._find_goal_pos()
    
    while not done and steps < max_steps:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
    # Robust success detection: coord OR reward
    is_on_goal = False
    if goal_pos:
        agent_pos = env._base_env().agent_pos
        is_on_goal = (int(agent_pos[0]) == goal_pos[0] and int(agent_pos[1]) == goal_pos[1])
        
    success = (total_reward > 0) or is_on_goal
    
    return success, steps, total_reward

def _build_episode_row(env_id, episode, model_name, success, steps, reward, buffer_size, history_window, tokens=0, latency=0.0, most_used_template="None", learned_templates_count=0, total_templates=0):
    return {
        "env": env_id,
        "episode": episode,
        "model": model_name,
        "success": success,
        "steps": steps,
        "reward": reward,
        "buffer_size": buffer_size,
        "history_window": history_window,
        "tokens": tokens,
        "time": latency,
        "most_used_template": most_used_template,
        "learned_templates_count": learned_templates_count,
        "total_templates": total_templates,
    }

def run_bot_experiments(model_names, environments, n_episodes_list, max_steps_list, buffer_sizes=(2,), history_windows=(3,), mock=True, max_new_tokens=16, temperature=0.0, return_errors=False):
    resolved_envs = _resolve_environment_pairs(environments)
    frames = []; all_action_logs = []; errors = []
    
    total_conditions = len(model_names) * len(resolved_envs) * len(n_episodes_list) * len(max_steps_list) * len(buffer_sizes) * len(history_windows)
    pbar = tqdm(total=total_conditions, desc="Overall Progress")
    
    for model_name in model_names:
        vllm_server = None
        if not mock and USE_VLLM:
            vllm_server = VLLMServer(model_name, port=VLLM_PORT)
            if not vllm_server.start():
                print(f"Skipping model {model_name} due to vLLM failure.")
                continue

        llm_client = LLMClient(model_name=model_name, max_new_tokens=max_new_tokens, temperature=temperature, mock=mock, use_vllm=USE_VLLM)
        print(f"\n--- Starting Model: {model_name} ---")
        for canonical_name, resolved_env_id in resolved_envs:
            for n_episodes in n_episodes_list:
                for max_steps in max_steps_list:
                    for buffer_size in buffer_sizes:
                        for history_window in history_windows:
                            try:
                                # Update progress bar and log condition
                                cond_str = f"Model={model_name.split('/')[-1]}, Env={canonical_name.split('-')[1]}, Buf={buffer_size}, Hist={history_window}"
                                pbar.set_postfix({"cond": cond_str})
                                print(f"Running Condition: {cond_str}")
                                
                                env = MinigridTextWrapper(resolved_env_id)
                                buffer_mgr = BufferManager(buffer_size=buffer_size)
                                results = []
                                # Optional: inner progress bar for episodes if n_episodes is large
                                for i in range(n_episodes):
                                    agent = BoTAgent(llm_client, buffer_mgr, use_history=True, history_window=history_window)
                                    # Reset metrics for this episode
                                    start_tok = llm_client.total_tokens
                                    start_lat = llm_client.total_latency
                                    
                                    # Use episode index as seed for consistent across-model comparison
                                    success, steps, reward = _run_single_episode(env, agent, max_steps, seed=42+i)
                                    
                                    ep_tok = llm_client.total_tokens - start_tok
                                    ep_lat = llm_client.total_latency - start_lat
                                    
                                    most_used = "None"
                                    if agent.used_templates:
                                        most_used = max(set(agent.used_templates), key=agent.used_templates.count)
                                        
                                    learned_count = sum(1 for t in buffer_mgr.meta_buffer.templates if "Learned Strategy" in t)
                                    total_count = len(buffer_mgr.meta_buffer.templates)
                                    
                                    results.append(_build_episode_row(
                                        canonical_name, i, model_name, success, steps, reward, buffer_size, history_window, 
                                        tokens=ep_tok, latency=ep_lat, most_used_template=most_used,
                                        learned_templates_count=learned_count, total_templates=total_count
                                    ))
                                    
                                    # Update dynamic buffer stats
                                    for t_name in set(agent.used_templates):
                                        buffer_mgr.meta_buffer.update_stats(t_name, success)
                                        
                                    # Buffer Learning Step: Only learn from efficient trajectories
                                    if success and not mock and steps < (max_steps * 0.5):
                                        buffer_mgr.meta_buffer.learn(agent.action_log, llm_client)

                                    for entry in agent.action_log:
                                        entry.update({"model": model_name, "env": canonical_name, "episode": i, "buffer_size": buffer_size, "history_window": history_window})
                                    all_action_logs.extend(agent.action_log)
                                frames.append(pd.DataFrame(results))
                                pbar.update(1)
                            except Exception as e:
                                print(f"Error in condition: {e}")
                                errors.append({"model": model_name, "env": canonical_name, "error": str(e)})
                                pbar.update(1)
        
        # GPU Memory Management: Clear cache between model switches
        if not mock:
            if USE_VLLM and vllm_server:
                vllm_server.stop()
            elif torch.cuda.is_available():
                print(f"Clearing GPU cache after model: {model_name}")
                del llm_client
                torch.cuda.empty_cache()
            
    pbar.close()
                                
    all_results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined_logs = pd.concat([pd.DataFrame(l) for l in [all_action_logs] if l], ignore_index=True) if all_action_logs else pd.DataFrame()
    errors_df = pd.DataFrame(errors) if errors else pd.DataFrame(columns=["model", "env", "error"])
    
    if return_errors:
        return all_results, combined_logs, errors_df
    return all_results, combined_logs


def build_run_config(model_candidates, mock_mode, max_new_tokens, temperature, history_windows, environments, n_episodes_list, max_steps_list, buffer_sizes):
    return {
        "model_candidates": model_candidates,
        "mock_mode": mock_mode,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "history_windows": list(history_windows),
        "environments": environments,
        "n_episodes_list": list(n_episodes_list),
        "max_steps_list": list(max_steps_list),
        "buffer_sizes": list(buffer_sizes),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

def save_run_outputs(all_results, action_logs, run_errors, output_root, run_tag, config, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save episodes
    all_results.to_csv(run_dir / "episodes.csv", index=False)
    
    # Save summary
    if not all_results.empty:
        summary = all_results.groupby(["model", "env", "buffer_size", "history_window"])[
            ["success", "steps", "reward", "tokens", "time"]
        ].mean()
        summary.to_csv(run_dir / "summary.csv")
    
    # Save action logs
    action_logs.to_csv(run_dir / "actions.csv", index=False)
    
    # Save errors if any
    if not run_errors.empty:
        run_errors.to_csv(run_dir / "errors.csv", index=False)
        
    # Save config
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(config, f, indent=2)
        
    return {"run_dir": str(run_dir)}


# Redundant plotting block removed for cleanup.


# %%
import sys

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()

# 1. Setup logging directory and file
run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{RUN_TAG}"
run_dir = Path(OUTPUT_ROOT) / run_id
run_dir.mkdir(parents=True, exist_ok=True)
log_path = run_dir / "run_log.log"

print(f"Logging experiment output to: {log_path}")

with open(log_path, "a", encoding="utf-8") as f_log:
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, f_log)
    try:
        # 2. Run experiments
        all_results, action_logs, run_errors = run_bot_experiments(
            model_names=MODEL_CANDIDATES,
            environments=ENVIRONMENTS,
            n_episodes_list=N_EPISODES_LIST,
            max_steps_list=MAX_STEPS_LIST,
            buffer_sizes=BUFFER_SIZES,
            history_windows=HISTORY_WINDOWS,
            mock=MOCK_MODE,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            return_errors=True,
        )
    finally:
        sys.stdout = original_stdout

# 3. Build config and save results
run_config = build_run_config(
    model_candidates=MODEL_CANDIDATES,
    mock_mode=MOCK_MODE,
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    history_windows=HISTORY_WINDOWS,
    environments=ENVIRONMENTS,
    n_episodes_list=N_EPISODES_LIST,
    max_steps_list=MAX_STEPS_LIST,
    buffer_sizes=BUFFER_SIZES,
)

run_artifacts = save_run_outputs(
    all_results=all_results,
    action_logs=action_logs,
    run_errors=run_errors,
    output_root=OUTPUT_ROOT,
    run_tag=RUN_TAG,
    config=run_config,
    run_dir=run_dir,  # Pass the pre-created directory
)

print("\nExperiment Results Summary")
print(f"Rows: {len(all_results)}")
print(f"Action log rows: {len(action_logs)}")
print(f"Errors: {len(run_errors)}")
print(f"Saved to: {run_artifacts['run_dir']}")

print(all_results.head(10))
if not run_errors.empty:
    print(run_errors.head(10))


# %%
if "all_results" not in dir() or all_results.empty:
    print("No experiment results available to visualize.")
else:
    df = all_results.copy()
    env_short = df["env"].str.replace("MiniGrid-", "").str.replace("-v0", "")
    df["env_short"] = env_short

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    model_env = df.groupby(["env_short", "model"])["success"].mean().unstack(fill_value=0)
    model_env.plot(kind="bar", ax=axes[0, 0], colormap="Set2")
    axes[0, 0].set_title("Success Rate: 3B vs 7B", fontweight="bold")
    axes[0, 0].set_ylabel("Success Rate")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis='x', rotation=30)
    axes[0, 0].legend(title='Model', fontsize=8)

    buf_env = df.groupby(["env_short", "buffer_size"])["success"].mean().unstack(fill_value=0)
    buf_env.columns = [f"buf={int(c)}" for c in buf_env.columns]
    buf_env.plot(kind="bar", ax=axes[0, 1], colormap="Paired")
    axes[0, 1].set_title("Success Rate: Buffer Size 2 vs 5", fontweight="bold")
    axes[0, 1].set_ylabel("Success Rate")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis='x', rotation=30)
    axes[0, 1].legend(title='Templates', fontsize=8)

    hw_env = df.groupby(["history_window", "env_short"])["success"].mean().unstack(fill_value=0)
    hw_env.plot(kind="line", marker="o", ax=axes[0, 2], linewidth=2)
    axes[0, 2].set_title("Success Rate vs History Window", fontweight="bold")
    axes[0, 2].set_xlabel("History Window")
    axes[0, 2].set_ylabel("Success Rate")
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].set_xticks(sorted(df['history_window'].unique()))
    axes[0, 2].legend(title='Env', fontsize=7)

    df["config"] = (
        df["model"].str.split("/").str[-1].str.replace("Qwen2.5-", "") + "\n"
        + "buf=" + df["buffer_size"].astype(str)
        + " k=" + df["history_window"].astype(str)
    )
    heat_data = df.groupby(["env_short", "config"])["success"].mean().unstack(fill_value=0)
    sns.heatmap(
        heat_data,
        annot=True,
        fmt=".2f",
        cmap="YlGn",
        vmin=0,
        vmax=1,
        ax=axes[1, 0],
        cbar_kws={'label': 'Success Rate'},
    )
    axes[1, 0].set_title("Success Rate Heatmap (all configs)", fontweight="bold")
    axes[1, 0].set_ylabel("Environment")
    axes[1, 0].set_xlabel("Config")
    axes[1, 0].tick_params(axis='x', rotation=45, labelsize=7)
    axes[1, 0].tick_params(axis='y', rotation=0)

    model_short = df["model"].str.split("/").str[-1]
    sns.boxplot(
        data=df.assign(model_short=model_short),
        x="env_short",
        y="steps",
        hue="model_short",
        ax=axes[1, 1],
        palette="Set2",
    )
    axes[1, 1].set_title("Steps Distribution by Env & Model", fontweight="bold")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Steps")
    axes[1, 1].tick_params(axis='x', rotation=30)
    axes[1, 1].legend(title='Model', fontsize=8)

    reward_data = df.groupby(["env_short", "buffer_size"])["reward"].mean().unstack(fill_value=0)
    reward_data.columns = [f"buf={int(c)}" for c in reward_data.columns]
    reward_data.plot(kind="bar", ax=axes[1, 2], colormap="coolwarm")
    axes[1, 2].set_title("Average Reward by Env & Buffer Size", fontweight="bold")
    axes[1, 2].set_ylabel("Reward")
    axes[1, 2].set_xlabel("")
    axes[1, 2].tick_params(axis='x', rotation=30)
    axes[1, 2].legend(title='Templates', fontsize=8)

    plt.tight_layout()

    artifacts = globals().get("run_artifacts")
    if artifacts:
        out_path = Path(artifacts["run_dir"]) / "overview.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {out_path}")

    plt.show()

    print("\nDetailed Summary:")
    summary = df.groupby(["model", "env", "buffer_size", "history_window"])[
        ["success", "steps", "reward", "tokens", "time"]
    ].mean()
    print(summary)


# %%
import shutil
shutil.make_archive("CS228", "zip", "CS 228")



