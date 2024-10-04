

'''def find_missing_files(base_dir):
    # Directory for the specific run
    run_dir = os.path.join(base_dir, 'run_005')
    
    # Directories for RGB, depth, and JSON files
    rgb_dir = os.path.join(run_dir, 'rgb')
    depth_dir = os.path.join(run_dir, 'disparity')
    json_dir = os.path.join(run_dir, 'json')

    # Get all file names (without paths) in each directory
    rgb_files = {remove_suffix(f, ['_rgb.jpg']) for f in os.listdir(rgb_dir) if f.endswith('.jpg')}
    depth_files = {remove_suffix(f, ['_disparity.png']) for f in os.listdir(depth_dir) if f.endswith('.png')}
    json_files = {remove_suffix(f, ['.json']) for f in os.listdir(json_dir) if f.endswith('.json')}

    # Find files in depth and json that are not in rgb
    missing_depth_in_rgb = depth_files - rgb_files
    missing_json_in_rgb = json_files - rgb_files

    # Print results
    if missing_depth_in_rgb:
        print("Removing depth files missing in RGB:")
        for file in missing_depth_in_rgb:
            file_path = os.path.join(depth_dir, f"{file}_disparity.png")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed: {file_path}")

    if missing_json_in_rgb:
        print("Removing JSON files missing in RGB:")
        for file in missing_json_in_rgb:
            file_path = os.path.join(json_dir, f"{file}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed: {file_path}")

if __name__ == "__main__":
    dataset_dir = "/home/h2x/Desktop/REAL_TIME_WORKING/run_local/09-20-2024/rc_data"
    find_missing_files(dataset_dir)'''
# Here we remove the files which dont have the same subsequent json or rgb or depth file 
'''def find_missing_files(run_dir):
    # Directories for RGB, depth, and JSON files
    rgb_dir = os.path.join(run_dir, 'rgb')
    depth_dir = os.path.join(run_dir, 'disparity')
    json_dir = os.path.join(run_dir, 'json')

    # Get all file names (without paths) in each directory
    rgb_files = {remove_suffix(f, ['_rgb.jpg']) for f in os.listdir(rgb_dir) if f.endswith('.jpg')}
    depth_files = {remove_suffix(f, ['_disparity.png']) for f in os.listdir(depth_dir) if f.endswith('.png')}
    json_files = {remove_suffix(f, ['.json']) for f in os.listdir(json_dir) if f.endswith('.json')}

    # Find files in depth and json that are not in rgb
    missing_depth_in_rgb = depth_files - rgb_files
    missing_json_in_rgb = json_files - rgb_files

    # Print results and remove missing files
    if missing_depth_in_rgb:
        # print("Removing depth files missing in RGB:")
        for file in missing_depth_in_rgb:
            file_path = os.path.join(depth_dir, f"{file}_disparity.png")
            if os.path.exists(file_path):
                os.remove(file_path)
                # print(f"Removed: {file_path}")

    if missing_json_in_rgb:
        # print("Removing JSON files missing in RGB:")
        for file in missing_json_in_rgb:
            file_path = os.path.join(json_dir, f"{file}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                # print(f"Removed: {file_path}")'''

import os

def remove_suffix(filename, suffixes):
    """Remove specified suffixes from the filename."""
    for suffix in suffixes:
        if filename.endswith(suffix):
            return filename[:-len(suffix)]
    return filename

def find_missing_files(run_dir):
    # Directories for RGB, depth, and JSON files
    rgb_dir = os.path.join(run_dir, 'rgb')
    depth_dir = os.path.join(run_dir, 'disparity')
    json_dir = os.path.join(run_dir, 'json')

    # Get all file names (without paths) in each directory
    rgb_files = {remove_suffix(f, ['_rgb.jpg']) for f in os.listdir(rgb_dir) if f.endswith('.jpg')}
    depth_files = {remove_suffix(f, ['_disparity.png']) for f in os.listdir(depth_dir) if f.endswith('.png')}
    json_files = {remove_suffix(f, ['.json']) for f in os.listdir(json_dir) if f.endswith('.json')}

    # Get the set of all unique file names across all three types
    all_files = rgb_files.union(depth_files).union(json_files)

    # Check each file in the union for existence in all directories
    for file in all_files:
        missing = []
        if file not in rgb_files:
            missing.append('RGB')
        if file not in depth_files:
            missing.append('Depth')
        if file not in json_files:
            missing.append('JSON')

        # If any file is missing, remove corresponding files from all directories
        if missing:
            for file_type in missing:
                if file_type == 'RGB':
                    file_path = os.path.join(rgb_dir, f"{file}_rgb.jpg")
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"Removed RGB file: {file_path}")
                elif file_type == 'Depth':
                    file_path = os.path.join(depth_dir, f"{file}_disparity.png")
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"Removed Depth file: {file_path}")
                elif file_type == 'JSON':
                    file_path = os.path.join(json_dir, f"{file}.json")
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        print(f"Removed JSON file: {file_path}")

if __name__ == "__main__":
    # Provide the full path to the run directory
    run_directory = "/home/h2x/Desktop/REAL_TIME_WORKING/run_local/09-20-2024/rc_data/run_003"
    find_missing_files(run_directory)


#############################
#this is for checking the integrity of the files
import os
import glob
import json

def check_dataset(run_dir):
    rgb_dir = os.path.join(run_dir, 'rgb')
    depth_dir = os.path.join(run_dir, 'disparity')
    json_dir = os.path.join(run_dir, 'json')

    rgb_files = glob.glob(os.path.join(rgb_dir, '*.jpg'))
    depth_files = glob.glob(os.path.join(depth_dir, '*.png'))
    json_files = glob.glob(os.path.join(json_dir, '*.json'))

    # Create a set of base names for files
    rgb_basenames = {os.path.basename(f).replace('_rgb.jpg', '') for f in rgb_files}
    depth_basenames = {os.path.basename(f).replace('_disparity.png', '') for f in depth_files}
    json_basenames = {os.path.basename(f).replace('.json', '') for f in json_files}

    # Check JSON files
    for json_file in json_files:
        base_name = os.path.basename(json_file).replace('.json', '')
        try:
            with open(json_file, 'r') as f:
                json.load(f)  # Raises an error if the file is empty or malformed
        except Exception as e:
            # print(f"Invalid JSON file: {json_file} - {str(e)}")
            # Remove related files if they exist
            if base_name in rgb_basenames:
                os.remove(os.path.join(rgb_dir, f"{base_name}_rgb.jpg"))
                # print(f"Removed RGB file: {base_name}_rgb.jpg")
            if base_name in depth_basenames:
                os.remove(os.path.join(depth_dir, f"{base_name}_disparity.png"))
                # print(f"Removed Depth file: {base_name}_disparity.png")
            os.remove(json_file)
            # print(f"Removed invalid JSON file: {json_file}")

    # Check RGB files for emptiness
    for rgb_file in rgb_files:
        if os.path.getsize(rgb_file) == 0:
            # print(f"Empty RGB file: {rgb_file}")
            base_name = os.path.basename(rgb_file).replace('_rgb.jpg', '')
            # Remove related files
            if base_name in depth_basenames:
                os.remove(os.path.join(depth_dir, f"{base_name}_disparity.png"))
                # print(f"Removed Depth file: {base_name}_disparity.png")
            if base_name in json_basenames:
                os.remove(os.path.join(json_dir, f"{base_name}.json"))
                # print(f"Removed JSON file: {base_name}.json")
            os.remove(rgb_file)

    # Check Depth files for emptiness
    for depth_file in depth_files:
        if os.path.getsize(depth_file) == 0:
            # print(f"Empty Depth file: {depth_file}")
            base_name = os.path.basename(depth_file).replace('_disparity.png', '')
            # Remove related files
            if base_name in rgb_basenames:
                os.remove(os.path.join(rgb_dir, f"{base_name}_rgb.jpg"))
                # print(f"Removed RGB file: {base_name}_rgb.jpg")
            if base_name in json_basenames:
                os.remove(os.path.join(json_dir, f"{base_name}.json"))
                # print(f"Removed JSON file: {base_name}.json")
            os.remove(depth_file)

# # Example usage
# check_dataset('/home/h2x/Desktop/REAL_TIME_WORKING/run_local/09-20-2024/rc_data/run_003')

