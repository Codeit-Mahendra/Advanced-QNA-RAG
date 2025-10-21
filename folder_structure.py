import os

def print_folder(root):
    """Print folder structure."""
    for root, dirs, files in os.walk(root):
        level = root.replace('.', '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        for file in sorted(files):
            print(f"{indent}  {file}")

def check_files():
    """Check for required files."""
    required = [
        "data/Three Thousand Stitches by Sudha Murthy.pdf",
        "src/helper.py",
        "store_index.py",
        "app.py",
        "requirements.txt",
        "render.yaml",
        "runtime.txt",
        ".env"
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("\nMissing files:")
        for f in missing:
            print(f"- {f}")
    else:
        print("\nAll required files present.")

if __name__ == "__main__":
    print("Folder structure:\n")
    print_folder(".")
    check_files()