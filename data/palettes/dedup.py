# NOTE: I asked an LLM to write this because I was lazy.
# The code works, but don't touch it. I don't entirely trust it.

import os
import hashlib
from PIL import Image

IMPORTANT_DIRS = ["vanilla", "baba", "new_adv", "museum", "vanilla-extensions", "legacy"]

def get_pixel_hash(filepath):
    try:
        with Image.open(filepath) as im:
            im = im.convert("RGBA")
            pixel_data = im.tobytes()
            return hashlib.md5(pixel_data).hexdigest()
    except IOError:
        print(f"Warning: Could not process file as an image: {filepath}")
        return None

def deduplicate_images_with_priority(root_path='.'):
    print("Starting global image deduplication process with priority directories...")
    
    image_hashes_map = {}
    files_deleted = 0
    
    priority_paths = {os.path.normpath(os.path.join(root_path, d)) for d in IMPORTANT_DIRS}

    print(f"\n[Phase 1] Scanning priority directories: {', '.join(IMPORTANT_DIRS)}")
    for priority_dir in IMPORTANT_DIRS:
        priority_path = os.path.join(root_path, priority_dir)
        if not os.path.isdir(priority_path):
            print(f"  Warning: Priority directory '{priority_dir}' not found. Skipping.")
            continue
            
        for dirpath, _, filenames in os.walk(priority_path):
            print(f"  Scanning: '{dirpath}'")
            for filename in filenames:
                if filename.lower().endswith('.png'):
                    file_path = os.path.join(dirpath, filename)
                    image_hash = get_pixel_hash(file_path)
                    
                    if not image_hash:
                        continue

                    if image_hash not in image_hashes_map:
                        image_hashes_map[image_hash] = file_path
                    else:
                        original_file_path = image_hashes_map[image_hash]
                        print(f"  - Deleting intra-priority duplicate: {file_path}")
                        print(f"    (Original found at: {original_file_path})")
                        try:
                            os.remove(file_path)
                            files_deleted += 1
                        except OSError as e:
                            print(f"    Error deleting file {file_path}: {e}")
                            
    print(f"  Found {len(image_hashes_map)} unique images in priority directories.")
    print("\n[Phase 2] Scanning remaining directories for duplicates...")
    for dirpath, _, filenames in os.walk(root_path):
        is_sub_of_priority = any(dirpath.startswith(p) for p in priority_paths)
        if os.path.normpath(dirpath) in priority_paths or is_sub_of_priority:
            continue

        print(f"  Scanning: '{dirpath}'")
        for filename in filenames:
            if filename.lower().endswith('.png'):
                file_path = os.path.join(dirpath, filename)
                image_hash = get_pixel_hash(file_path)

                if not image_hash:
                    continue

                if image_hash in image_hashes_map:
                    original_file_path = image_hashes_map[image_hash]
                    print(f"  - Deleting duplicate: {file_path}")
                    print(f"    (Original found at: {original_file_path})")
                    try:
                        os.remove(file_path)
                        files_deleted += 1
                    except OSError as e:
                        print(f"    Error deleting file {file_path}: {e}")
                else:
                    image_hashes_map[image_hash] = file_path

    print("\nProcess finished.")
    print(f"Found {len(image_hashes_map)} total unique images.")
    print(f"Total duplicate files deleted: {files_deleted}")

if __name__ == '__main__':
    deduplicate_images_with_priority()

