import os
import subprocess
import glob

# --- Configuration ---
CIRRASCALE_API_KEY = "sk-64e26e25a04c44ebb6d31c911f745a27" 
input_pdf_folder = "/Volumes/ExtremeSSD/NARA Visit 2 Scans/RG 60 Warrants/separated"
workspace_folder = "/Users/toddnobles/Documents/pows/data/processed/rg60_warrants_extracted"
model_name = "olmOCR-2-7B-1025"
server_url = "https://ai2endpoints.cirrascale.ai/api"

DRY_RUN = False  
SAMPLE_SIZE = 15

def run_olmocr_pipeline():
    original_cwd = os.getcwd()
    
    # 1. Validation
    if not os.path.exists(input_pdf_folder):
        print(f"Error: SSD not found at {input_pdf_folder}")
        return

    # 2. CHANGE WORKING DIRECTORY TO THE SSD
    # This is the "magic" step that prevents deep nesting
    os.chdir(input_pdf_folder)
    
    # Get just the filenames, not the full paths
    pdf_files = sorted(glob.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDFs found in {input_pdf_folder}")
        os.chdir(original_cwd)
        return

    if DRY_RUN:
        pdf_files = pdf_files[:SAMPLE_SIZE]

    # 3. Create manifest with FILENAMES ONLY
    # We save the manifest inside the input folder temporarily
    manifest_name = "pdf_manifest.txt"
    with open(manifest_name, "w") as f:
        for pdf_name in pdf_files:
            f.write(pdf_name + "\n")
            
    print(f"Created manifest with {len(pdf_files)} files.")

    # 4. Build the command
    # Note: we use the absolute path for workspace_folder so it knows where to send results
    command = [
        "python", "-m", "olmocr.pipeline",
        workspace_folder,
        "--server", server_url,
        "--api_key", CIRRASCALE_API_KEY,
        "--model", model_name,
        "--pdfs", manifest_name, 
        "--markdown",
        "--pages_per_group", "5" 
    ]

    print(f"Starting olmOCR pipeline from the SSD root...")
    
    try:
        # Running the command while CWD is the input folder
        subprocess.run(command, check=True, cwd = input_pdf_folder)
        print(f"\nSuccess! Check {workspace_folder}/markdown/")
    except subprocess.CalledProcessError as e:
        print(f"\nError: {e}")
    finally:
        # Cleanup and return home
        if os.path.exists(manifest_name):
            os.remove(manifest_name)
        os.chdir(original_cwd)

if __name__ == "__main__":
    run_olmocr_pipeline()