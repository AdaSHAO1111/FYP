import json

# Open the notebook
with open('Ada_new_code_with_QSMF.ipynb', 'r') as f:
    notebook = json.load(f)

changes_made = 0

# Find the cell with the file paths and update them
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            # Fix data file paths
            if 'file_path =' in line and '.TXT' in line:
                # Replace the file path with the corrected version
                old_path = cell['source'][i]
                # Extract the filename
                parts = old_path.split('\"')
                if len(parts) >= 3:
                    filename = parts[1]
                    # Create the new path with data/ prefix and lowercase extension
                    new_filename = 'data/' + filename.replace('.TXT', '.txt')
                    # Replace the filename in the original string
                    new_path = old_path.replace(filename, new_filename)
                    cell['source'][i] = new_path
                    print(f'Updated: {old_path.strip()} to {new_path.strip()}')
                    changes_made += 1
            
            # Fix marker image paths
            if 'imread(\"start.png\")' in line:
                old_path = cell['source'][i]
                new_path = old_path.replace('\"start.png\"', '\"data/start.png\"')
                cell['source'][i] = new_path
                print(f'Updated: {old_path.strip()} to {new_path.strip()}')
                changes_made += 1
            
            if 'imread(\"enda.png\")' in line:
                old_path = cell['source'][i]
                new_path = old_path.replace('\"enda.png\"', '\"data/enda.png\"')
                cell['source'][i] = new_path
                print(f'Updated: {old_path.strip()} to {new_path.strip()}')
                changes_made += 1

# Save the modified notebook
with open('Ada_new_code_with_QSMF.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print(f'Notebook updated successfully with {changes_made} changes.') 