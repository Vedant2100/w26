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
        "openai",
        "requests",
        "sentencepiece",
        "jupyter",
        "nbconvert",
        "imageio"
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
    import os
    from pathlib import Path
    from datetime import datetime, timezone
    import shutil
    
    print("🚀 Starting Modal Experiment Run at:", datetime.now(timezone.utc).isoformat())
    
    # Ensure HuggingFace cache is on persistent volume
    os.environ["HF_HOME"] = "/data/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/data/hf_cache"
    
    # Symlink persistent volume for results
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

    # Notebook name
    notebook_name = "Vedant_Borkute_DL_Project_BoT_Implementation.ipynb"
    script_name = "nb_run.py"
    
    print(f"📓 Converting {notebook_name} to script...")
    try:
        subprocess.run(["jupyter", "nbconvert", "--to", "script", notebook_name, "--output", "nb_run"], check=True, cwd="/root")
        
        # EXTENSION FIX: nbconvert sometimes outputs .txt instead of .py if metadata is messy
        if not os.path.exists(f"/root/{script_name}"):
            if os.path.exists("/root/nb_run.txt"):
                print("⚠️  nbconvert produced .txt; renaming to .py")
                os.rename("/root/nb_run.txt", f"/root/{script_name}")
            else:
                # Search for any file named nb_run.*
                files = os.listdir("/root")
                found = False
                for f in files:
                    if f.startswith("nb_run."):
                        print(f"⚠️  Found alternative extension: {f}; renaming to .py")
                        os.rename(f"/root/{f}", f"/root/{script_name}")
                        found = True
                        break
                if not found:
                    raise FileNotFoundError("Could not find nbconvert output script!")

        # Patch the script to save gifs into the persistent volume folder
        with open(f"/root/{script_name}", "r") as f:
            content = f.read()
        
        # Redirect outputs to the symlinked CS 228 folder so they persist
        content = content.replace('gif_folder="episode_gifs"', 'gif_folder="CS 228/episode_gifs"')
        
        with open(f"/root/{script_name}", "w") as f:
            f.write(content)
            
        print(f"🐍 Executing converted script {script_name}...")
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
