

import os

def print_tree(startpath, prefix=""):
    for item in sorted(os.listdir(startpath)):
        path = os.path.join(startpath, item)
        if os.path.isdir(path):
            print(f"{prefix}├── {item}/")
            print_tree(path, prefix + "│   ")
        else:
            print(f"{prefix}├── {item}")

# Replace with your project root
project_root = "D:/Advanced-QNA-RAG"
print(f"{os.path.basename(project_root)}/")
print_tree(project_root)