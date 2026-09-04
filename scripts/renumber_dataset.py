"""Copy a folder of images or labels into a consistent 001, 002, 003 numbering.

Used to bring the RGB, depth and label folders into the same ascending order so
a frame's colour, depth and annotation share a base name.

    python scripts/renumber_dataset.py <source_folder> <destination_folder>
"""

import argparse
import os
import shutil


def renumber(source_folder, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)

    # Get all files in the source folder
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    # Sort files to maintain consistent order
    files.sort()

    # Loop and copy + rename
    for idx, filename in enumerate(files, start=1):
        ext = os.path.splitext(filename)[1]
        new_name = f"{idx:03}{ext}"  # Names like 001.png, 002.jpg, etc.
        src = os.path.join(source_folder, filename)
        dst = os.path.join(destination_folder, new_name)
        shutil.copy2(src, dst)  # Use copy2 to preserve metadata

    print(f"Copied and renumbered {len(files)} files into {destination_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_folder", help="folder to read from")
    parser.add_argument("destination_folder", help="folder to write renumbered copies into")
    args = parser.parse_args()
    renumber(args.source_folder, args.destination_folder)
