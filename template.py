
import os

# Define directories and files
directories = ['src', 'research']
files = [
    'src/__init__.py',
    'src/helper.py',
    'src/prompt.py',
    '.env',
    'setup.py',
    'app.py',
    'research/trials.ipynb',
    'requirements.txt'
]

# Create directories
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Create files
for file in files:
    with open(file, 'w') as f:
        pass  # Creates an empty file

print("Directory and files created successfully!")

