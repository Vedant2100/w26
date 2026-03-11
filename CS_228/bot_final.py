import re
import json
import time
import sys
import importlib
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import gymnasium as gym
import requests
from minigrid.core.actions import Actions
from minigrid.wrappers import FullyObsWrapper


def tqdm(iterable, **kwargs):
    try:
        _tqdm = importlib.import_module("tqdm").tqdm
        return _tqdm(iterable, **kwargs)
    except Exception:
        _ = kwargs
        return iterable


def start_vllm_server(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    port=8000,
    log_file="vllm_server.log",
):
    command = (
        f"nohup {sys.executable} -m vllm.entrypoints.openai.api_server "
        f"--model {model_name} --dtype auto --api-key empty --port {port} "
        f"> {log_file} 2>&1 &"
    )
    subprocess.run(command, shell=True, check=False)


def check_vllm_ready(port=8000):
    url = f"http://localhost:{port}/v1/models"
    headers = {"Authorization": "Bearer empty"}
    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"status={r.status_code} body={r.text[:200]}")
    model = r.json().get("data", [{}])[0].get("id", "unknown")
    return model


def tail_file(path, n_lines=80):
    p = Path(path)
    if not p.exists():
        return f"{path} not found"
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n_lines:])
    except Exception as e:
        return f"Failed to read {path}: {e}"


def wait_for_vllm(port=8000, retries=60, sleep_s=10, log_file="vllm_server.log"):
    last_err = ""
    for _ in tqdm(range(retries), desc="vLLM startup", leave=False):
        try:
            _ = check_vllm_ready(port=port)
            return True, ""
        except Exception as e:
            last_err = str(e)
        time.sleep(sleep_s)

    diag = [f"Last readiness error: {last_err}"]
    diag.append("\n--- Last vLLM log lines ---")
    diag.append(tail_file(log_file, n_lines=80))
    return False, "\n".join(diag)


class RunLogger:
    def __init__(self, root_dir="bot_logs"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(root_dir) / f"run_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.log"
        self.current_env = None
        self.env_dirs = {}

    @staticmethod
    def _safe_env_name(env_name):
        return str(env_name).replace("/", "_").replace(" ", "_")

    def set_env(self, env_name):
        self.current_env = env_name
        if env_name is None:
            return
        safe = self._safe_env_name(env_name)
        env_dir = self.run_dir / safe
        env_dir.mkdir(parents=True, exist_ok=True)
        self.env_dirs[env_name] = env_dir

    def _env_events_path(self, env_name):
        if env_name not in self.env_dirs:
            self.set_env(env_name)
        return self.env_dirs[env_name] / "events.jsonl"

    def _env_summary_path(self, env_name):
        if env_name not in self.env_dirs:
            self.set_env(env_name)
        return self.env_dirs[env_name] / "summary.log"

    def log_event(self, event_type, payload, echo=False):
        env_name = payload.get("env") or self.current_env
        row = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "event": event_type,
            "env": env_name,
            "payload": payload,
        }
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
        if env_name is not None:
            env_events = self._env_events_path(env_name)
            with env_events.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        _ = echo

    def log_summary(self, text, env_name=None):
        with self.summary_path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")
        env_name = env_name or self.current_env
        if env_name is not None:
            env_summary = self._env_summary_path(env_name)
            with env_summary.open("a", encoding="utf-8") as f:
                f.write(text + "\n")


class MinigridTextWrapper:
    """Observation wrapper intentionally aligned with ToT text style."""

    DIR_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
    DIR_NAMES = ["right", "down", "left", "up"]

    def __init__(self, env_id, render_mode=None):
        self.env = gym.make(env_id, render_mode=render_mode)
        self.env = FullyObsWrapper(self.env)
        self.action_map = {
            "turn_left": Actions.left,
            "turn_right": Actions.right,
            "move_forward": Actions.forward,
        }

    def _base(self):
        return self.env.unwrapped

    def _cell_content(self, grid, x, y):
        if x < 0 or y < 0 or x >= grid.width or y >= grid.height:
            return "wall"
        cell = grid.get(x, y)
        return cell.type if cell else "empty"

    def get_text_obs(self, _obs):
        base = self._base()
        grid = base.grid
        ax, ay = int(base.agent_pos[0]), int(base.agent_pos[1])
        agent_dir = int(base.agent_dir)
        facing = self.DIR_NAMES[agent_dir]

        desc = f"Agent at [{ax},{ay}] facing {facing}. "

        fdx, fdy = self.DIR_VEC[agent_dir]
        fx, fy = ax + fdx, ay + fdy
        front = self._cell_content(grid, fx, fy)

        rdx, rdy = self.DIR_VEC[(agent_dir + 1) % 4]
        right = self._cell_content(grid, ax + rdx, ay + rdy)

        ldx, ldy = self.DIR_VEC[(agent_dir - 1) % 4]
        left = self._cell_content(grid, ax + ldx, ay + ldy)

        bdx, bdy = self.DIR_VEC[(agent_dir + 2) % 4]
        behind = self._cell_content(grid, ax + bdx, ay + bdy)

        desc += (
            f"Nearby - front: {front}, left: {left}, right: {right}, behind: {behind}. "
        )
        desc += f"The cell directly in front of you contains {front}. "

        goals, lavas, interior_walls = [], [], []
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is None:
                    continue
                t = getattr(cell, "type", None)
                if t == "goal":
                    goals.append((x, y))
                elif t == "lava":
                    lavas.append(f"[{x},{y}]")
                elif t == "wall":
                    if 0 < x < grid.width - 1 and 0 < y < grid.height - 1:
                        interior_walls.append(f"[{x},{y}]")

        if goals:
            gx, gy = goals[0]
            desc += f"Goal at [{gx},{gy}]. "

            dx, dy = gx - ax, gy - ay
            manhattan = abs(dx) + abs(dy)
            dirs_needed = []
            if dx > 0:
                dirs_needed.append("right")
            if dx < 0:
                dirs_needed.append("left")
            if dy > 0:
                dirs_needed.append("down")
            if dy < 0:
                dirs_needed.append("up")

            desc += f"Distance: {manhattan} steps. "
            if dirs_needed:
                desc += f"You need to go: {', '.join(dirs_needed)}. "
                if facing in dirs_needed:
                    desc += "You are facing TOWARD the goal. "
                else:
                    desc += "You are NOT facing the goal - consider turning. "

            new_dist = abs(fx - gx) + abs(fy - gy)
            if front not in ("wall", "lava"):
                if new_dist < manhattan:
                    desc += "Moving forward brings you CLOSER to the goal. "
                elif new_dist == manhattan:
                    desc += "Moving forward does NOT change distance to goal. "
                else:
                    desc += "Moving forward takes you FARTHER from goal. "

        if lavas:
            desc += f"Lava: {', '.join(lavas)}. "
        if interior_walls:
            desc += f"Interior walls: {', '.join(interior_walls)}. "

        return desc

    def reset(self):
        obs, _ = self.env.reset()
        return self.get_text_obs(obs)

    def step(self, action_str):
        action = self.action_map.get(action_str.lower(), Actions.forward)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return self.get_text_obs(obs), reward, done, info


@dataclass
class ThoughtTemplate:
    name: str
    reasoning_pattern: str
    usage_count: int = 0
    success_rate: float = 0.5

    def instantiate(self, distilled_obs):
        parts = []
        if distilled_obs.get("agent_pos") is not None:
            parts.append(f"agent={distilled_obs['agent_pos']}")
        if distilled_obs.get("facing") is not None:
            parts.append(f"facing={distilled_obs['facing']}")
        if distilled_obs.get("goal") is not None:
            parts.append(f"goal={distilled_obs['goal']}")
        if distilled_obs.get("front") is not None:
            parts.append(f"front={distilled_obs['front']}")
        if distilled_obs.get("lava"):
            parts.append(f"lava_count={len(distilled_obs['lava'])}")
        return f"[TEMPLATE={self.name}] {' | '.join(parts)} | {self.reasoning_pattern}"


class ProblemDistiller:
    @staticmethod
    def distill(obs_text, env=None):
        out = {
            "agent_pos": None,
            "facing": None,
            "goal": None,
            "front": None,
            "left": None,
            "right": None,
            "lava": set(),
            "walls": set(),
            "forward_progress": False,
            "turn_pref": "right",
            "gap_target": None,
        }

        if env is not None and hasattr(env, "unwrapped"):
            base = env.unwrapped
            out["agent_pos"] = (int(base.agent_pos[0]), int(base.agent_pos[1]))
            out["facing"] = MinigridTextWrapper.DIR_NAMES[int(base.agent_dir)]

            fx, fy = int(base.front_pos[0]), int(base.front_pos[1])
            front_obj = base.grid.get(fx, fy)
            out["front"] = front_obj.type if front_obj else "empty"

            ax, ay = out["agent_pos"]
            dir_idx = int(base.agent_dir)
            ldx, ldy = MinigridTextWrapper.DIR_VEC[(dir_idx - 1) % 4]
            rdx, rdy = MinigridTextWrapper.DIR_VEC[(dir_idx + 1) % 4]
            left_obj = base.grid.get(ax + ldx, ay + ldy)
            right_obj = base.grid.get(ax + rdx, ay + rdy)
            out["left"] = left_obj.type if left_obj else "empty"
            out["right"] = right_obj.type if right_obj else "empty"

            for x in range(base.grid.width):
                for y in range(base.grid.height):
                    c = base.grid.get(x, y)
                    if c is None:
                        continue
                    t = getattr(c, "type", None)
                    if t == "goal":
                        out["goal"] = (x, y)
                    elif t == "lava":
                        out["lava"].add((x, y))
                    elif t == "wall":
                        if 0 < x < base.grid.width - 1 and 0 < y < base.grid.height - 1:
                            out["walls"].add((x, y))

            # --- remove gap oracle: do not compute gap_candidate ---
            out["gap_target"] = None
            # keep only local percepts (front/left/right) and let agent discover gaps

            target = out["goal"]
            if target is not None:
                tx, ty = target
                fdx, fdy = MinigridTextWrapper.DIR_VEC[dir_idx]
                cur_dist = abs(tx - ax) + abs(ty - ay)
                next_dist = abs(tx - (ax + fdx)) + abs(ty - (ay + fdy))
                out["forward_progress"] = (
                    out["front"] not in ("wall", "lava") and next_dist < cur_dist
                )

                left_dist = abs(tx - (ax + ldx)) + abs(ty - (ay + ldy))
                right_dist = abs(tx - (ax + rdx)) + abs(ty - (ay + rdy))
                out["turn_pref"] = "left" if left_dist <= right_dist else "right"
            return out

        m = re.search(r"Agent at \[(\d+),(\d+)\] facing (right|down|left|up)", obs_text)
        if m:
            out["agent_pos"] = (int(m.group(1)), int(m.group(2)))
            out["facing"] = m.group(3)

        m = re.search(r"Goal at \[(\d+),(\d+)\]", obs_text)
        if m:
            out["goal"] = (int(m.group(1)), int(m.group(2)))

        m = re.search(r"Nearby - front: (\w+)", obs_text)
        if m:
            out["front"] = m.group(1)

        return out


def _classify_grid(lava_set, walls_set, grid_w=0, grid_h=0):
    """Return a coarse category string for the grid layout."""
    if not lava_set and not walls_set:
        return "open"
    if walls_set and not lava_set:
        return "walled"
    # Check for lava barrier rows / columns.
    lava_rows = {}
    lava_cols = {}
    for lx, ly in lava_set:
        lava_rows.setdefault(ly, set()).add(lx)
        lava_cols.setdefault(lx, set()).add(ly)
    row_thresh = max(3, (grid_w - 2) // 2) if grid_w > 0 else 3
    col_thresh = max(3, (grid_h - 2) // 2) if grid_h > 0 else 3
    has_row_barrier = any(len(xs) >= row_thresh for xs in lava_rows.values())
    has_col_barrier = any(len(ys) >= col_thresh for ys in lava_cols.values())
    if has_row_barrier:
        return "lava_row"
    if has_col_barrier:
        return "lava_col"
    return "lava_scattered"


class MetaBuffer:
    MAX_TEMPLATES = 8
    SEED_TEMPLATES = {
        "Direct Navigation",
        "Lava Crossing",
        "Obstacle Avoidance",
        "Recover",
    }

    def __init__(self):
        self.templates = {
            "Direct Navigation": ThoughtTemplate(
                "Direct Navigation",
                "Align with goal direction and move_forward on safe cells.",
            ),
            "Lava Crossing": ThoughtTemplate(
                "Lava Crossing",
                "In lava maps, move_forward only on safe cells that reduce distance; if blocked by lava, turn to the safer side and continue.",
            ),
            "Obstacle Avoidance": ThoughtTemplate(
                "Obstacle Avoidance",
                "When blocked by lava or wall, turn to find a safe direction before moving.",
            ),
            "Recover": ThoughtTemplate(
                "Recover",
                "If stuck in repeats, prefer alternate turns to break loops.",
            ),
        }
        self.seen_failures = {}  # signature -> count

    def retrieve(self, distilled_obs, repeated_state):
        if repeated_state:
            return self.templates["Recover"]

        scores = {
            name: (t.success_rate if t.usage_count > 0 else 0.5)
            for name, t in self.templates.items()
        }

        front = distilled_obs.get("front")
        has_lava = len(distilled_obs.get("lava", set())) > 0
        has_walls = len(distilled_obs.get("walls", set())) > 0
        agent_pos = distilled_obs.get("agent_pos")
        goal = distilled_obs.get("goal")
        facing = distilled_obs.get("facing")

        forward_reduces_dist = False
        if (
            agent_pos is not None
            and goal is not None
            and facing in MinigridTextWrapper.DIR_NAMES
        ):
            dir_idx = MinigridTextWrapper.DIR_NAMES.index(facing)
            dx, dy = MinigridTextWrapper.DIR_VEC[dir_idx]
            ax, ay = agent_pos
            gx, gy = goal
            cur_dist = abs(gx - ax) + abs(gy - ay)
            next_dist = abs(gx - (ax + dx)) + abs(gy - (ay + dy))
            forward_reduces_dist = next_dist < cur_dist

        if front in ("wall", "lava"):
            scores["Obstacle Avoidance"] += 1.5
        if has_lava:
            scores["Lava Crossing"] += 1.4
            scores["Obstacle Avoidance"] += 0.8
        if has_walls:
            scores["Obstacle Avoidance"] += 1.0

        if front == "empty" and forward_reduces_dist:
            scores["Direct Navigation"] += 1.0
            if has_lava:
                scores["Lava Crossing"] += 1.0

        if goal is not None and not has_lava and not has_walls:
            scores["Direct Navigation"] += 1.0

        # Score learned templates by matching their signature to current state.
        grid_cat = _classify_grid(
            distilled_obs.get("lava", set()),
            distilled_obs.get("walls", set()),
        )
        for name in scores:
            if not name.startswith("Learned:"):
                continue
            # name format: "Learned:{cause}_{grid_cat}"
            tag = name[len("Learned:") :]
            # Match grid category.
            if grid_cat in tag:
                scores[name] += 1.2
            # Match cause pattern to current situation.
            if "lava_death" in tag and front == "lava":
                scores[name] += 2.0
            elif "lava_death" in tag and has_lava:
                scores[name] += 1.0
            if "stuck_loop" in tag and repeated_state:
                scores[name] += 1.5
            if "timeout" in tag and not forward_reduces_dist and has_lava:
                scores[name] += 1.0

        best_name = max(sorted(scores.keys()), key=lambda n: scores[n])
        return self.templates[best_name]

    def update_stats(self, template_name, success):
        t = self.templates[template_name]
        t.usage_count += 1
        t.success_rate += (float(success) - t.success_rate) / t.usage_count

    def add_template(self, name, reasoning_pattern):
        """Add a learned template, respecting max cap and dedup."""
        # Dedup: skip if an existing template already covers similar keywords.
        new_kw = set(reasoning_pattern.lower().split())
        for existing in self.templates.values():
            existing_kw = set(existing.reasoning_pattern.lower().split())
            overlap = len(new_kw & existing_kw)
            if overlap >= 0.6 * max(len(new_kw), 1):
                return False  # too similar

        # Cap: if full, evict the worst non-seed learned template.
        if len(self.templates) >= self.MAX_TEMPLATES:
            worst_name, worst_sr = None, float("inf")
            for n, t in self.templates.items():
                if n in self.SEED_TEMPLATES:
                    continue
                if t.success_rate < worst_sr:
                    worst_sr = t.success_rate
                    worst_name = n
            if worst_name is None:
                return False  # all slots are seed templates
            del self.templates[worst_name]

        self.templates[name] = ThoughtTemplate(name, reasoning_pattern)
        return True

    def record_failure(self, signature):
        """Track failure signature. Returns True when count reaches 2 (trigger)."""
        self.seen_failures[signature] = self.seen_failures.get(signature, 0) + 1
        return self.seen_failures[signature] == 2

    def has_template_for_signature(self, signature):
        """Check if a learned template already targets this signature."""
        target_name = f"Learned:{signature[0]}_{signature[1]}"
        return target_name in self.templates

    def snapshot(self):
        return {
            name: {
                "usage_count": t.usage_count,
                "success_rate": round(t.success_rate, 4),
            }
            for name, t in self.templates.items()
        }


class BoTAgent:
    """Lean Buffer-of-Thought policy with template retrieval + LLM-guided action choice."""

    VALID_ACTIONS = ["turn_left", "turn_right", "move_forward"]
    DIR_NAMES = ["right", "down", "left", "up"]
    DIR_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

    def __init__(
        self,
        model="Qwen/Qwen2.5-7B-Instruct",
        history_size=6,
        rollout_depth=3,
        base_url="http://localhost:8000/v1",
        api_key="empty",
        temperature=0.1,
        logger=None,
    ):
        self.buffer = MetaBuffer()
        self.model = model
        self.history_size = history_size
        self.rollout_depth = rollout_depth
        self.temperature = temperature
        OpenAI = importlib.import_module("openai").OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.memory = []
        self.state_trace = []
        self._last_template = "Direct Navigation"
        self._episode_trace = []  # [(obs_snippet, action), ...] for learning
        self._episode_lava = set()
        self._episode_walls = set()
        self._episode_grid_wh = (0, 0)
        self.logger = logger

    def _dir_idx(self, name):
        return self.DIR_NAMES.index(name) if name in self.DIR_NAMES else 0

    def _template_action_order(self, template_name, blocked_front):
        # Prefer safe turns first and bias toward the side that improves target progress.
        # distilled info is added in act() via closure-style attributes below.
        if template_name == "Obstacle Avoidance":
            order = ["turn_right", "turn_left", "move_forward"]
        elif template_name == "Lava Crossing":
            order = ["move_forward", "turn_right", "turn_left"]
        elif template_name == "Recover":
            order = ["turn_left", "turn_right", "move_forward"]
        else:
            order = ["move_forward", "turn_right", "turn_left"]

        if blocked_front:
            order = [a for a in order if a != "move_forward"]
            if not order:
                order = ["turn_right", "turn_left"]
        return order

    def _target_aware_action_order(self, template_name, blocked_front, distilled_obs):
        left_blocked = distilled_obs.get("left") in ("wall", "lava")
        right_blocked = distilled_obs.get("right") in ("wall", "lava")
        turn_pref = distilled_obs.get("turn_pref", "right")
        forward_progress = bool(distilled_obs.get("forward_progress", False))

        safe_turns = []
        if not right_blocked:
            safe_turns.append("turn_right")
        if not left_blocked:
            safe_turns.append("turn_left")
        if not safe_turns:
            safe_turns = ["turn_right", "turn_left"]

        if turn_pref == "left":
            safe_turns = sorted(safe_turns, key=lambda a: 0 if a == "turn_left" else 1)
        else:
            safe_turns = sorted(safe_turns, key=lambda a: 0 if a == "turn_right" else 1)

        # Map learned templates to the closest seed-style ordering.
        effective_name = template_name
        if template_name.startswith("Learned:"):
            tag = template_name.lower()
            if "lava_death" in tag:
                effective_name = "Obstacle Avoidance"  # prioritise turning away
            elif "stuck_loop" in tag:
                effective_name = "Recover"
            else:
                effective_name = (
                    "Lava Crossing" if "lava" in tag else "Direct Navigation"
                )

        if effective_name == "Lava Crossing":
            if blocked_front:
                order = safe_turns
            elif forward_progress:
                order = ["move_forward"] + safe_turns
            else:
                order = safe_turns + ["move_forward"]
        elif effective_name == "Obstacle Avoidance":
            order = safe_turns + ["move_forward"]
        elif effective_name == "Recover":
            order = ["turn_left", "turn_right", "move_forward"]
        else:
            order = (
                ["move_forward"] + safe_turns
                if forward_progress
                else safe_turns + ["move_forward"]
            )

        return order

    def _llm_pick_action(self, obs, template_text, options, blocked_front):
        options = options[:3]
        system_msg = (
            "You are a MiniGrid action selector. "
            "Choose exactly one action from the provided options. "
            "Actions are: turn_left, turn_right, move_forward. "
            "If front is blocked, never choose move_forward. "
            "Output only the action token."
        )
        user_msg = (
            f"Observation:\n{obs}\n\n"
            f"Template:\n{template_text}\n\n"
            f"Candidate actions (priority order): {', '.join(options)}\n"
            f"Front blocked: {blocked_front}\n\n"
            "Return one action:"
        )

        if self.logger:
            self.logger.log_event(
                "llm_input",
                {
                    "template": template_text,
                    "options": options,
                    "blocked_front": blocked_front,
                    "system_msg": system_msg,
                    "user_msg": user_msg,
                },
                echo=True,
            )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self.temperature,
                max_tokens=12,
            )
            raw = (resp.choices[0].message.content or "").strip().lower()
        except Exception:
            if self.logger:
                self.logger.log_event(
                    "llm_output", {"raw": "llm_error_fallback"}, echo=True
                )
            return options[0], "llm_error_fallback"

        parsed = None
        for act in self.VALID_ACTIONS:
            if act in raw:
                parsed = act
                break
        if parsed is None:
            parsed = options[0]

        if self.logger:
            self.logger.log_event(
                "llm_output",
                {"raw": raw, "parsed_action": parsed},
                echo=True,
            )

        return parsed, raw

    def act(self, obs, env=None):
        t0 = time.time()
        d = ProblemDistiller.distill(obs, env)

        if d["agent_pos"] is None or d["facing"] is None:
            return "turn_right", "fallback:no_state", time.time() - t0

        dir_idx = self._dir_idx(d["facing"])
        state = (d["agent_pos"][0], d["agent_pos"][1], dir_idx)
        repeated_state = self.state_trace.count(state) >= 2
        repeated_pos = sum(1 for s in self.state_trace if s[:2] == state[:2]) >= 4
        repeated_state = repeated_state or repeated_pos

        template = self.buffer.retrieve(d, repeated_state)
        self._last_template = template.name

        # Final safety override.
        blocked_front = d.get("front") in ("wall", "lava")
        template_text = template.instantiate(d)
        action_order = self._target_aware_action_order(template.name, blocked_front, d)
        llm_action, llm_raw = self._llm_pick_action(
            obs=obs,
            template_text=template_text,
            options=action_order,
            blocked_front=blocked_front,
        )

        chosen_action = llm_action

        reasoning = f"{template_text} | options={action_order} | llm={llm_raw}"

        if self.logger:
            self.logger.log_event(
                "agent_step",
                {
                    "template_chosen": template.name,
                    "ranked_actions": action_order,
                    "chosen_action": chosen_action,
                    "buffer_state": self.buffer.snapshot(),
                    "reasoning": reasoning,
                },
                echo=True,
            )

        # Keep compact trace for post-episode learning.
        snippet = obs[:200] if len(obs) > 200 else obs
        self._episode_trace.append((snippet, chosen_action))
        self._episode_lava = d.get("lava", set())
        self._episode_walls = d.get("walls", set())
        if env is not None and hasattr(env, "unwrapped"):
            b = env.unwrapped
            self._episode_grid_wh = (int(b.grid.width), int(b.grid.height))

        self.memory.append((obs, chosen_action))
        if len(self.memory) > self.history_size:
            self.memory = self.memory[-self.history_size :]

        self.state_trace.append(state)
        if len(self.state_trace) > 50:
            self.state_trace = self.state_trace[-50:]

        return chosen_action, reasoning, time.time() - t0

    def learn(self, success, last_action=None, episode_reward=0.0):
        """Post-episode learning: on failure, extract signature and possibly generate a new template."""
        if success:
            return

        # Determine failure cause.
        if episode_reward < 0 or (
            last_action == "move_forward" and episode_reward <= 0
        ):
            cause = "lava_death"
        else:
            pos_counts = {}
            for s in self.state_trace:
                pos_counts[s[:2]] = pos_counts.get(s[:2], 0) + 1
            max_visits = max(pos_counts.values()) if pos_counts else 0
            cause = "stuck_loop" if max_visits >= 4 else "timeout"

        grid_cat = _classify_grid(
            self._episode_lava,
            self._episode_walls,
            self._episode_grid_wh[0],
            self._episode_grid_wh[1],
        )
        sig = (cause, grid_cat)

        if self.buffer.has_template_for_signature(sig):
            return

        should_generate = self.buffer.record_failure(sig)
        if not should_generate:
            if self.logger:
                self.logger.log_event(
                    "failure_recorded",
                    {
                        "signature": list(sig),
                        "count": self.buffer.seen_failures.get(sig, 0),
                    },
                )
            return

        # Build trace summary for LLM (last 10 steps).
        trace_lines = []
        for obs_snip, act in self._episode_trace[-10:]:
            trace_lines.append(f"  obs: {obs_snip}")
            trace_lines.append(f"  action: {act}")
        trace_summary = "\n".join(trace_lines)

        template_name = f"Learned:{cause}_{grid_cat}"
        reasoning = self._llm_generate_template(cause, grid_cat, trace_summary)

        added = self.buffer.add_template(template_name, reasoning)
        if self.logger:
            self.logger.log_event(
                "template_learned",
                {
                    "signature": list(sig),
                    "template_name": template_name,
                    "reasoning": reasoning,
                    "added": added,
                    "buffer_state": self.buffer.snapshot(),
                },
                echo=True,
            )

    def _llm_generate_template(self, cause, grid_cat, trace_summary):
        """Ask LLM to produce a short reasoning_pattern for a new template."""
        system_msg = (
            "You create navigation strategy templates for a MiniGrid agent. "
            "Given a failure cause and grid type, produce a ONE-SENTENCE action strategy. "
            "Be specific: mention which actions to prefer and when. "
            "Output only the strategy sentence, nothing else."
        )
        user_msg = (
            f"Failure cause: {cause}\n"
            f"Grid type: {grid_cat}\n"
            f"Recent trace (last 10 steps):\n{trace_summary}\n\n"
            "Write a one-sentence strategy to avoid this failure:"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=60,
            )
            raw = (resp.choices[0].message.content or "").strip()
            # Sanitise: keep only the first sentence.
            raw = raw.split(".")[0].strip() + "."
            return raw
        except Exception:
            return f"Avoid {cause} in {grid_cat} by varying turns and checking surroundings."

    def reset(self):
        self.memory = []
        self.state_trace = []
        self._last_template = "Direct Navigation"
        self._episode_trace = []
        self._episode_lava = set()
        self._episode_walls = set()
        self._episode_grid_wh = (0, 0)


def evaluate_agent(
    agent,
    env_name="MiniGrid-Empty-8x8-v0",
    num_episodes=10,
    max_steps_per_episode=100,
    logger=None,
    update_buffer_during_eval=False,
    save_episode_gifs=True,
    gif_fps=6,
):
    """ToT-style evaluation: reset() without fixed seeds, reward-based success."""
    env = MinigridTextWrapper(env_name, render_mode="rgb_array")
    imageio_mod = None
    if save_episode_gifs:
        try:
            imageio_mod = importlib.import_module("imageio.v2")
        except Exception:
            try:
                imageio_mod = importlib.import_module("imageio")
            except Exception:
                save_episode_gifs = False
    if logger:
        logger.set_env(env_name)

    gif_dir = None
    if save_episode_gifs:
        if logger and env_name in logger.env_dirs:
            gif_dir = logger.env_dirs[env_name] / "gifs"
        elif logger:
            gif_dir = logger.run_dir / RunLogger._safe_env_name(env_name) / "gifs"
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            gif_dir = (
                Path("CS_228") / "bot_gifs" / ts / RunLogger._safe_env_name(env_name)
            )
        gif_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "success_count": 0,
        "total_steps_success": 0,
        "total_inference_time": 0.0,
        "total_actions": 0,
    }

    pbar = tqdm(range(num_episodes), desc=env_name, leave=False)
    for ep in pbar:
        obs = env.reset()
        agent.reset()
        done = False
        steps = 0
        episode_reward = 0.0
        frames = []
        if save_episode_gifs:
            frame0 = env.env.render()
            if frame0 is not None:
                frames.append(frame0)

        while not done and steps < max_steps_per_episode:
            action, _reason, latency = agent.act(obs, env.env)
            metrics["total_inference_time"] += latency
            metrics["total_actions"] += 1

            obs, reward, done, _ = env.step(action)
            if save_episode_gifs:
                frame = env.env.render()
                if frame is not None:
                    frames.append(frame)
            episode_reward += reward
            steps += 1

            if logger:
                logger.log_event(
                    "env_step",
                    {
                        "env": env_name,
                        "episode": ep + 1,
                        "step": steps,
                        "action": action,
                        "reward": reward,
                        "done": done,
                    },
                    echo=False,
                )

        success = episode_reward > 0
        if success:
            metrics["success_count"] += 1
            metrics["total_steps_success"] += steps

        # Determine last action for failure classification.
        last_action = agent.memory[-1][1] if agent.memory else None
        if update_buffer_during_eval:
            agent.learn(success, last_action=last_action, episode_reward=episode_reward)

        if save_episode_gifs and gif_dir is not None and len(frames) > 0:
            gif_path = gif_dir / f"episode_{ep + 1:03d}.gif"
            imageio_mod.mimsave(gif_path, frames, fps=gif_fps)
            print(f"[GIF SAVED] {env_name} episode {ep + 1}: {gif_path}", flush=True)
            if logger:
                logger.log_event(
                    "episode_gif_saved",
                    {
                        "env": env_name,
                        "episode": ep + 1,
                        "gif_path": str(gif_path),
                        "frames": len(frames),
                    },
                    echo=False,
                )

        running_sr = 100.0 * metrics["success_count"] / (ep + 1)
        pbar.set_postfix(sr=f"{running_sr:.1f}%", steps=steps)
        if logger:
            logger.log_event(
                "episode_end",
                {
                    "env": env_name,
                    "episode": ep + 1,
                    "success": bool(success),
                    "steps": steps,
                    "episode_reward": episode_reward,
                },
                echo=True,
            )

    pbar.close()

    success_rate = 100.0 * metrics["success_count"] / num_episodes
    avg_steps = (
        metrics["total_steps_success"] / metrics["success_count"]
        if metrics["success_count"] > 0
        else float("inf")
    )
    avg_inf = (
        metrics["total_inference_time"] / metrics["total_actions"]
        if metrics["total_actions"] > 0
        else 0.0
    )

    if logger:
        logger.log_summary(
            f"[SUMMARY] {env_name} | episodes={num_episodes} | success_rate={success_rate:.2f} | avg_steps={avg_steps:.2f} | avg_inf={avg_inf:.4f}",
            env_name=env_name,
        )
        if save_episode_gifs and gif_dir is not None:
            logger.log_summary(
                f"[GIFS] {env_name} | dir={gif_dir}",
                env_name=env_name,
            )
    metrics["success_rate"] = success_rate
    metrics["avg_steps_success"] = avg_steps
    metrics["avg_inference_time"] = avg_inf
    if save_episode_gifs and gif_dir is not None:
        metrics["gif_dir"] = str(gif_dir)
    return metrics


def run_full_experiment(ensure_vllm=False, performance_mode=True):
    logger = RunLogger(root_dir="CS_228/bot_logs")

    if ensure_vllm:
        try:
            _ = check_vllm_ready(port=8000)
        except Exception:
            start_vllm_server(
                model_name="Qwen/Qwen2.5-7B-Instruct",
                port=8000,
                log_file="vllm_server.log",
            )

        ok, diag = wait_for_vllm(
            port=8000,
            retries=60,
            sleep_s=10,
            log_file="vllm_server.log",
        )
        if not ok:
            raise RuntimeError(
                "vLLM failed to become ready on http://localhost:8000\n" + diag
            )

    envs = [
        "MiniGrid-Empty-8x8-v0",
        "MiniGrid-LavaGapS6-v0",
        "MiniGrid-LavaCrossingS9N2-v0",
    ]
    learn_episodes = 20
    eval_episodes = 20
    max_steps = 100

    all_results = {}

    for env_name in envs:
        agent = BoTAgent(
            model="Qwen/Qwen2.5-7B-Instruct",
            history_size=6,
            rollout_depth=3,
            base_url="http://localhost:8000/v1",
            api_key="empty",
            temperature=0.1,
            logger=logger,
        )

        # Phase 1: Learning (template creation allowed).
        print(f"\n{'='*72}")
        print(f"[LEARN] {env_name} — {learn_episodes} episodes (learning ON)")
        print(f"{'='*72}")
        learn_metrics = evaluate_agent(
            agent,
            env_name=env_name,
            num_episodes=learn_episodes,
            max_steps_per_episode=max_steps,
            logger=logger,
            update_buffer_during_eval=True,
            save_episode_gifs=False,
        )
        learned_templates = list(agent.buffer.templates.keys())
        print(
            f"[LEARN] {env_name} done — learning SR: {learn_metrics['success_rate']:.1f}%"
        )
        print(f"[LEARN] Templates after learning: {learned_templates}")

        if logger:
            logger.log_summary(
                f"[LEARN] {env_name} | episodes={learn_episodes} | "
                f"sr={learn_metrics['success_rate']:.2f}% | "
                f"templates={learned_templates}",
                env_name=env_name,
            )

        # Phase 2: Evaluation (no learning, fresh metrics).
        print(f"\n[EVAL] {env_name} — {eval_episodes} episodes (learning OFF)")
        eval_metrics = evaluate_agent(
            agent,
            env_name=env_name,
            num_episodes=eval_episodes,
            max_steps_per_episode=max_steps,
            logger=logger,
            update_buffer_during_eval=False,
            save_episode_gifs=True,
        )
        all_results[env_name] = eval_metrics

        if logger:
            logger.log_summary(
                f"[EVAL] {env_name} | episodes={eval_episodes} | "
                f"sr={eval_metrics['success_rate']:.2f}% | "
                f"avg_steps={eval_metrics['avg_steps_success']:.2f}",
                env_name=env_name,
            )

    print(f"\n{'='*72}")
    print("Final Evaluation Results (after learning phase)")
    print("=" * 72)
    print(
        f"{'Env':<30} | {'Success Rate':>12} | {'Avg Steps':>9} | {'Avg Inf/step':>12}"
    )
    print("-" * 72)
    for env_name, m in all_results.items():
        print(
            f"{env_name:<30} | {m['success_rate']:10.2f}% | {m['avg_steps_success']:9.2f} | {m['avg_inference_time']:.4f}s"
        )
    print("=" * 72)


if __name__ == "__main__":
    # Set ensure_vllm=True if you want this script to spawn and wait for vLLM.
    run_full_experiment(ensure_vllm=True)
