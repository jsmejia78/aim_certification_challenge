import os

def process_file(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 3:
        return None  # Not enough lines to process

    title = lines[0].strip()
    third_line = lines[2].strip()

    if len(third_line) <= 30:
        return None  # Skip if line 3 is too short

    content = "".join(lines[2:]).strip()  # Skip the second line
    return f"{title} - {content}"

def transform_folder_structure(input_dir, output_dir):
    # Step 1: Pre-create subfolder structure
    for root, dirs, _ in os.walk(input_dir):
        for dir_name in dirs:
            source_subdir = os.path.join(root, dir_name)
            rel_path = os.path.relpath(source_subdir, input_dir)
            target_subdir = os.path.join(output_dir, rel_path)
            os.makedirs(target_subdir, exist_ok=True)

    # Step 2: Process and write files
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".txt"):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)

                new_content = process_file(input_path)
                if new_content is None:
                    continue  # Skip short files

                with open(output_path, "w", encoding="utf-8") as f_out:
                    f_out.write(new_content)

if __name__ == "__main__":
    input_folder = "/home/jsmejia/the_ai_eng_bootcamp/code/aim_certification_challenge/data/pd_blogs_originals"
    output_folder = "/home/jsmejia/the_ai_eng_bootcamp/code/aim_certification_challenge/data/pd_blogs_filtered"

    transform_folder_structure(input_folder, output_folder)
    print("Folders created and files processed.")
