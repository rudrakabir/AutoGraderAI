import os
import fitz  # PyMuPDF
from docx import Document
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
import time
import traceback

# --- Configuration ---
# Attempt to load API key from .env file
load_dotenv()
DEFAULT_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Model Name
MODEL_NAME = "gemini-2.5-flash-preview-04-17" # Or "gemini-pro"

# Delay between API calls (in seconds) to avoid hitting rate limits
API_DELAY = 2 # Increase if you hit rate limits

# --- Helper Functions (Mostly same as before, minor adjustments) ---

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        text = ' '.join(text.split()) # Basic cleaning
        return text
    except Exception as e:
        print(f"  ERROR reading PDF {os.path.basename(pdf_path)}: {e}")
        # traceback.print_exc() # Uncomment for full error details during debugging
        return f"Error extracting text: {e}" # Return error message

def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file."""
    try:
        doc = Document(docx_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        text = ' '.join(text.split()) # Basic cleaning
        return text
    except Exception as e:
        print(f"  ERROR reading DOCX {os.path.basename(docx_path)}: {e}")
        # traceback.print_exc() # Uncomment for full error details during debugging
        return f"Error extracting text: {e}" # Return error message

def construct_prompt(rubric, student_text):
    """Constructs the prompt for the Gemini API."""
    # --- This prompt is crucial - Refine it for your specific needs! ---
    prompt = f"""
You are an expert AI grading assistant for a university-level writing class. Your task is to evaluate the provided student assignment based *strictly* and *exclusively* on the detailed grading rubric below.

**Grading Rubric:**
---
{rubric}
---

**Student Assignment Text:**
---
{student_text}
---

**Instructions:**
1.  Carefully read the entire student assignment text.
2.  Apply *only* the criteria outlined in the provided rubric. Do not introduce external criteria or biases.
3.  Provide specific, constructive feedback citing examples from the text where possible. Mention both strengths and areas for improvement as they relate to the rubric items.
4.  Determine a final score or grade based *only* on the rubric's scoring guide (e.g., points per section, overall grade description).
5.  **Format your response clearly.** Start with the overall grade/score. Then, provide detailed feedback, perhaps section by section according to the rubric. Ensure the response is well-structured for easy reading.

**Example Output Structure (Adapt based on Rubric):**

Overall Grade: [Insert Grade/Score based on Rubric]

Strengths (according to Rubric):
*   [Point related to rubric criterion 1]
*   [Point related to rubric criterion 2]

Areas for Improvement (according to Rubric):
*   [Point related to rubric criterion 3]
*   [Point related to rubric criterion 4]

Detailed Comments:
[Optional: More elaborate discussion]

---
**Begin Evaluation:**
"""
    return prompt

def grade_text_with_gemini(api_key, rubric, student_text, model_name=MODEL_NAME):
    """Sends the prompt to the Gemini API and returns the response text or an error message."""
    if student_text.startswith("Error extracting text:"):
        return "Skipped API call due to text extraction error."

    try:
        genai.configure(api_key=api_key)

        generation_config = {
            "temperature": 0.5,
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 8192,
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        model = genai.GenerativeModel(model_name=model_name,
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)

        prompt = construct_prompt(rubric, student_text)
        response = model.generate_content(prompt)

        if not response.parts:
             if response.prompt_feedback.block_reason:
                 block_reason = f"API call blocked due to safety settings. Reason: {response.prompt_feedback.block_reason}"
                 print(f"  WARNING: {block_reason}")
                 return block_reason
             else:
                 unknown_reason = "API returned an empty response. The prompt might be unsuitable or there was an unknown issue."
                 print(f"  WARNING: {unknown_reason}")
                 return unknown_reason

        return response.text

    except Exception as e:
        error_msg = f"Error during Gemini API call: {e}"
        print(f"  ERROR: {error_msg}")
        # traceback.print_exc() # Uncomment for full error details during debugging
        return error_msg # Return the error message to be logged

# --- Main Script Logic ---

def main():
    print("--- AI Bulk Assignment Grader ---")

    # 1. Get API Key
    api_key = input(f"Enter your Google AI Studio API Key (or press Enter to use .env): ").strip()
    if not api_key:
        api_key = DEFAULT_API_KEY
    if not api_key:
        print("ERROR: API Key not found in .env or provided input. Exiting.")
        return
    print("API Key loaded.")

    # 2. Get Assignments Folder
    while True:
        assignments_folder = input("Enter the path to the folder containing assignment files: ").strip()
        if os.path.isdir(assignments_folder):
            break
        else:
            print(f"ERROR: Folder not found at '{assignments_folder}'. Please enter a valid path.")

    # 3. Get Rubric
    rubric_source = input("Load rubric from file (f) or paste directly (p)? [f/p]: ").strip().lower()
    rubric = ""
    if rubric_source == 'f':
        while True:
            rubric_file_path = input("Enter the path to the rubric text file: ").strip()
            try:
                with open(rubric_file_path, 'r', encoding='utf-8') as f:
                    rubric = f.read()
                if rubric.strip():
                    print("Rubric loaded from file.")
                    break
                else:
                    print("ERROR: Rubric file appears empty.")
            except FileNotFoundError:
                print(f"ERROR: Rubric file not found at '{rubric_file_path}'.")
            except Exception as e:
                print(f"ERROR: Could not read rubric file: {e}")
    elif rubric_source == 'p':
        print("Paste your rubric below. Press Enter twice (or Ctrl+D/Ctrl+Z then Enter on some systems) to finish:")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError: # Indicates end of input
            pass
        rubric = "\n".join(lines).strip()
        if not rubric:
            print("ERROR: No rubric pasted. Exiting.")
            return
        print("Rubric received.")
    else:
        print("Invalid choice for rubric source. Exiting.")
        return

    # 4. Prepare for Processing
    all_files = [f for f in os.listdir(assignments_folder) if os.path.isfile(os.path.join(assignments_folder, f))]
    valid_files = [f for f in all_files if f.lower().endswith((".pdf", ".docx"))]
    skipped_files = [f for f in all_files if not f.lower().endswith((".pdf", ".docx"))]

    results = [] # List to store results for each file

    print(f"\nFound {len(valid_files)} PDF/DOCX files to process in '{assignments_folder}'.")
    if skipped_files:
        print(f"Skipping {len(skipped_files)} non-PDF/DOCX files: {', '.join(skipped_files)}")
    print("-" * 20)

    # 5. Process Files
    for i, filename in enumerate(valid_files):
        print(f"Processing file {i+1}/{len(valid_files)}: {filename} ...")
        file_path = os.path.join(assignments_folder, filename)
        extracted_text = ""
        llm_response = ""
        status = "Pending"

        # Extract Text
        try:
            if filename.lower().endswith(".pdf"):
                extracted_text = extract_text_from_pdf(file_path)
            elif filename.lower().endswith(".docx"):
                extracted_text = extract_text_from_docx(file_path)

            if extracted_text.startswith("Error extracting text:"):
                 status = "Text Extraction Failed"
                 llm_response = extracted_text # Log extraction error
            else:
                 status = "Text Extracted"

        except Exception as e:
            print(f"  CRITICAL ERROR during text extraction for {filename}: {e}")
            traceback.print_exc()
            extracted_text = f"Critical extraction error: {e}"
            llm_response = extracted_text
            status = "Text Extraction Failed (Critical)"


        # Grade with Gemini (only if text extraction succeeded)
        if status == "Text Extracted":
            print("  Sending to Gemini for grading...")
            try:
                llm_response = grade_text_with_gemini(api_key, rubric, extracted_text)
                if llm_response.startswith("Error during Gemini API call:") or \
                   llm_response.startswith("API call blocked") or \
                   llm_response.startswith("API returned an empty response") or \
                   llm_response.startswith("Skipped API call"):
                    status = "Grading Failed (API Error)"
                else:
                    status = "Grading Complete"
                print(f"  Status: {status}")

            except Exception as e:
                 print(f"  CRITICAL ERROR during API call for {filename}: {e}")
                 traceback.print_exc()
                 llm_response = f"Critical API error: {e}"
                 status = "Grading Failed (Critical)"

            # Add delay between API calls
            if i < len(valid_files) - 1: # Don't sleep after the last file
                print(f"  Waiting {API_DELAY}s before next API call...")
                time.sleep(API_DELAY)


        # Store result
        results.append({
            "Filename": filename,
            "Processing_Status": status,
            # "Extracted_Text": extracted_text, # Optional: Uncomment to include full text in CSV (can make file large)
            "LLM_Response": llm_response
        })

    # 6. Save Results
    output_filename = "grading_results.csv"
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_filename, index=False, encoding='utf-8')
        print("\n" + "=" * 30)
        print(f"Processing complete. Results saved to '{output_filename}'")
        print("=" * 30)
    except Exception as e:
        print(f"\nERROR: Failed to save results to CSV: {e}")
        print("Printing results here instead:")
        for res in results:
            print(res)

    print("\n--- Grading Process Finished ---")
    print("**Reminder:** Always review the LLM's output carefully. Use this as a tool to assist grading.")

if __name__ == "__main__":
    main()