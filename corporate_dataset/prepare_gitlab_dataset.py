import os
import shutil
from git import Repo

# Configure Repository to Extract GitLab Handbook
#
REPO_URL = "https://github.com/AnswerDotAI/gitlab-handbook"
TEMP_CLONE_DIR = "./gitlab_handbook_temp"
TARGET_DATASET_DIR = "./gitlab_engineering_workflow_dataset"

WORKFLOW_RELATIVE_PATH = "content/handbook/engineering/workflow"

def setup_dataset():
    if os.path.exists(TEMP_CLONE_DIR):
        print(f"Removing old temporary clone directory: {TEMP_CLONE_DIR}...")
        shutil.rmtree(TEMP_CLONE_DIR)
        
    if os.path.exists(TARGET_DATASET_DIR):
        print(f"Removing old dataset directory: {TARGET_DATASET_DIR}...")
        shutil.rmtree(TARGET_DATASET_DIR)

    os.makedirs(TARGET_DATASET_DIR, exist_ok=True)

    print(f"Cloning GitLab handbook (this might take a minute, please wait)...")
    try:
        Repo.clone_from(REPO_URL, TEMP_CLONE_DIR, depth=1)
        print("Repository successfully cloned!")
    except Exception as e:
        print(f"Failed to clone repository: {e}")
        return

    source_workflow_dir = os.path.join(TEMP_CLONE_DIR, WORKFLOW_RELATIVE_PATH)
    
    if not os.path.exists(source_workflow_dir):
        print(f"Could not find target directory at: {source_workflow_dir}")
        print("Please check if the folder structure in the GitLab repository has changed.")
        return

    print(f"Extracting Markdown files from {WORKFLOW_RELATIVE_PATH}...")
    markdown_count = 0
    
    for root, dirs, files in os.walk(source_workflow_dir):
        for file in files:
            if file.endswith('.md'):
                source_file_path = os.path.join(root, file)
                relative_sub_path = os.path.relpath(source_file_path, source_workflow_dir)
                flattened_name = relative_sub_path.replace(os.sep, '_')
                
                target_file_path = os.path.join(TARGET_DATASET_DIR, flattened_name)
                
                shutil.copy2(source_file_path, target_file_path)
                markdown_count += 1

    print(f"\nSuccess! Extracted {markdown_count} operational Markdown files.")
    print(f"Your clean dataset is located here: {os.path.abspath(TARGET_DATASET_DIR)}")

    # 5. Clean up temporary files to save hard drive space
    print(f"Cleaning up temporary repository folder...")
    shutil.rmtree(TEMP_CLONE_DIR)
    print("Dataset generation completed successfully.")

if __name__ == "__main__":
    setup_dataset()
