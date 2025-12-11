#!/usr/bin/env python3
"""
Helper script to update the embedding model configuration in embedding.ipynb
This switches from the non-existent google/embedding-gemma-300m to BAAI/bge-large-en-v1.5
"""

import json
import sys
from pathlib import Path


def update_notebook_config(notebook_path: str):
    """Update the USE_EMBEDDING_GEMMA flag in the notebook"""

    # Read the notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Find and update the configuration cell
    updated = False
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            # Check if this is the configuration cell
            source = "".join(cell["source"])
            if "USE_EMBEDDING_GEMMA" in source:
                # Update the source
                new_source = []
                for line in cell["source"]:
                    if "USE_EMBEDDING_GEMMA = True" in line:
                        new_source.append(line.replace("True", "False"))
                        updated = True
                        print(f"✓ Updated: {line.strip()} -> {new_source[-1].strip()}")
                    else:
                        new_source.append(line)
                cell["source"] = new_source

    if updated:
        # Write back the notebook
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"\n✓ Successfully updated {notebook_path}")
        print("\nNow using: BAAI/bge-large-en-v1.5")
        print("  - Dimensions: 1024")
        print("  - Context: 512 tokens")
        print("  - No authentication required!")
        print("\nNext steps:")
        print("  1. Restart your Jupyter kernel")
        print("  2. Re-run the configuration cell")
        print("  3. Re-run the model loading cell")
    else:
        print("⚠ Could not find USE_EMBEDDING_GEMMA in the notebook")
        sys.exit(1)


if __name__ == "__main__":
    notebook_path = Path(__file__).parent / "embedding.ipynb"

    if not notebook_path.exists():
        print(f"Error: {notebook_path} not found")
        sys.exit(1)

    print(f"Updating {notebook_path}...")
    update_notebook_config(str(notebook_path))
