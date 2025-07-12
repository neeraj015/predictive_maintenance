import os

folders = ['data', 'notebooks', 'src', 'models', 'app', 'logs']
files = {
    'README.md': '# Predictive Maintenance Project\n',
    '.gitignore': 'venv/\n__pycache__/\n*.pyc\n*.log\n',
    'requirements.txt': '',
    'app/app.py': '',
    'src/__init__.py': '',
    'notebooks/eda.ipynb': '',  # You can create this in Jupyter later
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for path, content in files.items():
    with open(path, 'w') as f:
        f.write(content)

print("✅ Project structure created.")
