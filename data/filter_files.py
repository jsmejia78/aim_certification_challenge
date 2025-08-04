import os

def process_file(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 2:
        return None  # Skip files with only 1 line or empty

    title = lines[0].strip()
    content = "".join(lines[2:]).strip()  # Skip the second line
    return f"{title} - {content}"

def transform_folder_structure(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".txt"):
                input_path = os.path.join(root, file)

                # Compute the relative path and target path
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)

                # Process file
                new_content = process_file(input_path)
                if new_content is None:
                    continue  # Skip files with only 1 line

                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Write new content
                with open(output_path, "w", encoding="utf-8") as f_out:
                    f_out.write(new_content)

if __name__ == "__main__":
    input_folder = "/home/jsmejia/the_ai_eng_bootcamp/code/aim_certification_challenge/data/pd_blogs_original"
    output_folder = "/home/jsmejia/the_ai_eng_bootcamp/code/aim_certification_challenge/data/pd_blogs_filtered"

    transform_folder_structure(input_folder, output_folder)
    print("All valid files processed and saved.")
