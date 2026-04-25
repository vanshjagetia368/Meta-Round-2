"""
==============================================================================
Universal-Node-Resolver — Final Hackathon Packager
==============================================================================

Verifies the existence of critical files and compresses the entire project
into a clean `.zip` archive, ready for upload to the hackathon portal.
"""

import os
import shutil
import sys
from pathlib import Path

# Required files to guarantee a complete submission
REQUIRED_FILES = [
    "assets/training_reward_curve.png",
    "README.md",
    "openenv.yaml",
    "run.py",
    "scripts/simulate_github_pr.py",
    "api/main.py",
    "docker-compose.yml"
]

OUTPUT_ZIP = "Universal_Node_Resolver_Final"

def check_requirements(root_dir: Path) -> bool:
    print(f"📦 Verifying project directory: {root_dir}")
    missing = []
    
    for req in REQUIRED_FILES:
        target = root_dir / req
        if not target.exists():
            # Special case for assets to avoid failing if they haven't run train.py
            if req.startswith("assets/"):
                print(f"⚠️ Warning: Missing {req}. Ensure you've run the training/evaluation scripts.")
            else:
                missing.append(req)
                print(f"❌ Missing critical file: {req}")
        else:
            print(f"✅ Found: {req}")

    if missing:
        print("\n🚨 Packaging aborted due to missing critical files.")
        return False
        
    return True


def package_project(root_dir: Path):
    print(f"\n🤐 Compressing project to {OUTPUT_ZIP}.zip...")
    
    # We use shutil.make_archive but need to ignore pycache and git
    def ignore_patterns(dirname, filenames):
        return [f for f in filenames if f == "__pycache__" or f.endswith(".pyc") or f == ".git"]

    # Create a temporary directory to copy clean files into
    temp_dir = root_dir / "temp_package_build"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        
    shutil.copytree(
        root_dir, 
        temp_dir, 
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git', '.env', 'temp_package_build', f'{OUTPUT_ZIP}.zip')
    )
    
    try:
        shutil.make_archive(
            base_name=OUTPUT_ZIP,
            format="zip",
            root_dir=temp_dir
        )
        print(f"\n✨ SUCCESS! Created {OUTPUT_ZIP}.zip in the root directory.")
        print("🚀 Ready for hackathon submission!")
    finally:
        # Cleanup temp
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    
    if check_requirements(project_root):
        package_project(project_root)
    else:
        sys.exit(1)
