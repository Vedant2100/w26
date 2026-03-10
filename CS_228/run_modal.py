import modal
import os
import shutil
from pathlib import Path
from datetime import datetime, timezone

# 1. Define the App and Image
app = modal.App("cs228-bot-sweep")

# Image with all necessary data science and RL libraries
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "gymnasium",
        "minigrid",
        "pandas",
        "matplotlib",
        "seaborn",
        "tqdm",
        "vllm",
        "openai",
        "requests",
        "sentencepiece",
    )
    .add_local_dir(".", remote_path="/root", ignore=["CS 228", "outputs", "run_log.log", "__pycache__", ".ipynb_checkpoints"])
)

# 2. Setup Persistent Volume for results and HF models
volume = modal.Volume.from_name("cs228-vol", create_if_missing=True)

# 3. Define the Execution Function
@app.function(
    gpu="A100",
    cpu=4,
    memory="40Gi",
    image=image,
    volumes={"/data": volume},
    timeout=10800, # 3 hours for deep sweep
)
def run_experiment_on_modal():
    import subprocess
    
    print("🚀 Starting Modal Experiment Run at:", datetime.now(timezone.utc).isoformat())
    
    # Ensure HuggingFace cache is on persistent volume
    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/data/hf_cache"
    
    # The script saves to "CS 228/outputs"
    # We want "CS 228" to be a symlink to "/data/CS 228"
    vol_output_path = Path("/data/CS 228")
    vol_output_path.mkdir(parents=True, exist_ok=True)
    
    local_output_path = Path("/root/CS 228")
    if local_output_path.exists():
        if local_output_path.is_symlink():
            local_output_path.unlink()
        else:
            shutil.rmtree(local_output_path)
            
    os.symlink(vol_output_path, local_output_path)
    print(f"✅ Linked {local_output_path} -> {vol_output_path}")

    # Execute the Python script
    script_name = "bot_exploration.py"
    print(f"🐍 Executing {script_name}...")
    
    try:
        subprocess.run(["python", script_name], check=True, cwd="/root")
        print("✨ Experiment execution successful!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment execution failed: {e}")
        raise

# 4. Local Entrypoint
@app.local_entrypoint()
def main():
    print("🛰️  Deploying to Modal GPU cluster...")
    run_experiment_on_modal.remote()
    print("\n" + "═"*50)
    print("🏁 EXPERIMENT COMPLETE")
    print("═"*50)
    print("To download all results to your local directory, run:")
    print("modal volume get cs228-vol \"CS 228\" .")
