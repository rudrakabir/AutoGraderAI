import os
import fitz  # PyMuPDF
from docx import Document
import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
import time
import traceback
import gradio as gr
import tempfile
import shutil
import zipfile

# --- Configuration ---
# Attempt to load API key from .env file
load_dotenv()
DEFAULT_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Model Name
MODEL_NAME = "gemini-1.5-pro-latest" # Or "gemini-pro"

# Delay between API calls (in seconds) to avoid hitting rate limits
API_DELAY = 2 # Increase if you hit rate limits

# --- Helper Functions (Minor adjustments for Gradio file objects) ---

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file path."""
    # pdf_path is the temporary path provided by Gradio
    original_filename = os.path.basename(pdf_path) # Get filename for logging
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        text = ' '.join(text.split()) # Basic cleaning
        if not text.strip():
             return f"Error extracting text: No text found in {original_filename}"
        return text
    except Exception as e:
        error_msg = f"Error extracting text from PDF {original_filename}: {e}"
        print(f"  ERROR: {error_msg}")
        # traceback.print_exc() # Uncomment for full error details during debugging
        return error_msg # Return error message

def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file path."""
    # docx_path is the temporary path provided by Gradio
    original_filename = os.path.basename(docx_path) # Get filename for logging
    try:
        doc = Document(docx_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        text = ' '.join(text.split()) # Basic cleaning
        if not text.strip():
             return f"Error extracting text: No text found or readable in {original_filename}"
        return text
    except Exception as e:
        error_msg = f"Error extracting text from DOCX {original_filename}: {e}"
        print(f"  ERROR: {error_msg}")
        # traceback.print_exc() # Uncomment for full error details during debugging
        return error_msg # Return error message

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
5.  **Format your response clearly.** Start with the overall grade/score. Then, provide detailed feedback, perhaps section by section according to the rubric. Ensure the response is well-structured for easy reading. Use Markdown for formatting if appropriate (like bullet points).

**Example Output Structure (Adapt based on Rubric):**

**Overall Grade:** [Insert Grade/Score based on Rubric]

**Strengths (according to Rubric):**
*   [Point related to rubric criterion 1]
*   [Point related to rubric criterion 2]

**Areas for Improvement (according to Rubric):**
*   [Point related to rubric criterion 3]
*   [Point related to rubric criterion 4]

**Detailed Comments:**
[Optional: More elaborate discussion]

---
**Begin Evaluation:**
"""
    return prompt

def grade_text_with_gemini(api_key, rubric, student_text, model_name=MODEL_NAME):
    """Sends the prompt to the Gemini API and returns the response text or an error message."""
    log_message = "" # To capture specific warnings/errors from this function

    if not student_text or student_text.startswith("Error extracting text:"):
        log_message = "Skipped API call due to text extraction error or empty text."
        print(f"  INFO: {log_message}")
        return student_text, log_message # Return original error

    try:
        genai.configure(api_key=api_key)

        generation_config = {
            "temperature": 0.6, # Slightly increased for potentially more nuanced feedback
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

        # Refined check for empty or blocked response
        if response.prompt_feedback.block_reason:
             block_reason = f"API call blocked due to safety settings. Reason: {response.prompt_feedback.block_reason}"
             print(f"  WARNING: {block_reason}")
             log_message = block_reason
             return block_reason, log_message
        elif not response.parts:
             unknown_reason = "API returned an empty response. The prompt might be unsuitable, the input text too short/problematic, or there was an unknown issue."
             print(f"  WARNING: {unknown_reason}")
             log_message = unknown_reason
             return unknown_reason, log_message
        else:
            # Check if text is empty even if parts exist (sometimes happens)
            if not response.text.strip():
                empty_text_reason = "API returned a response part but the text content was empty. Check model compatibility or input."
                print(f"  WARNING: {empty_text_reason}")
                log_message = empty_text_reason
                return empty_text_reason, log_message

            return response.text, log_message # Return successful response and empty log message

    except Exception as e:
        error_msg = f"Error during Gemini API call: {e}"
        print(f"  ERROR: {error_msg}")
        # traceback.print_exc() # Uncomment for full error details during debugging
        log_message = error_msg
        return error_msg, log_message # Return the error message

# --- Gradio Processing Function ---

def process_assignments(api_key_input, rubric_source, rubric_file, rubric_text, assignment_files):
    """
    Main function called by Gradio to process assignments.
    Takes inputs from the Gradio interface and returns outputs.
    """
    logs = ["--- Starting Grading Process ---"]
    results = []
    individual_feedback = {} # Store feedback string for each file
    output_zip_path = None # Path to the final zip file

    # 1. Validate API Key
    api_key = api_key_input if api_key_input else DEFAULT_API_KEY
    if not api_key:
        logs.append("ERROR: API Key is missing. Please enter your Google AI Studio API Key or set the GOOGLE_API_KEY environment variable.")
        return "\n".join(logs), pd.DataFrame(), None # Log, Empty DataFrame, No file path
    logs.append("API Key loaded.")
    try:
        # Test configuration early
         genai.configure(api_key=api_key)
         logs.append("Gemini API configured successfully.")
    except Exception as e:
         logs.append(f"ERROR: Failed to configure Gemini API with the provided key: {e}")
         return "\n".join(logs), pd.DataFrame(), None

    # 2. Load Rubric
    rubric = ""
    if rubric_source == "Upload File":
        if rubric_file is not None:
            try:
                with open(rubric_file.name, 'r', encoding='utf-8') as f:
                    rubric = f.read()
                if not rubric.strip():
                    logs.append("ERROR: Uploaded rubric file is empty.")
                    return "\n".join(logs), pd.DataFrame(), None
                logs.append(f"Rubric loaded successfully from file: {os.path.basename(rubric_file.name)}")
            except Exception as e:
                logs.append(f"ERROR: Could not read rubric file '{os.path.basename(rubric_file.name)}': {e}")
                return "\n".join(logs), pd.DataFrame(), None
        else:
            logs.append("ERROR: 'Upload File' selected for rubric, but no file was uploaded.")
            return "\n".join(logs), pd.DataFrame(), None
    elif rubric_source == "Paste Text":
        if rubric_text and rubric_text.strip():
            rubric = rubric_text.strip()
            logs.append("Rubric loaded successfully from pasted text.")
        else:
            logs.append("ERROR: 'Paste Text' selected for rubric, but no text was provided.")
            return "\n".join(logs), pd.DataFrame(), None
    else:
        logs.append("ERROR: Invalid rubric source selected.") # Should not happen with Radio choices
        return "\n".join(logs), pd.DataFrame(), None

    # 3. Check for Assignment Files
    if not assignment_files:
        logs.append("ERROR: No assignment files uploaded.")
        return "\n".join(logs), pd.DataFrame(), None

    valid_files = []
    skipped_files = []
    for file in assignment_files:
        # Use file.name which is the temp path. Get original name for logging/output.
        # Gradio uses tempfile._TemporaryFileWrapper - .name gives the path
        # We need the original name. Newer Gradio versions might have file.orig_name
        # Let's try to get it, otherwise use the temp name's basename
        original_filename = getattr(file, 'orig_name', os.path.basename(file.name))

        if original_filename.lower().endswith((".pdf", ".docx")):
            valid_files.append({'path': file.name, 'orig_name': original_filename})
        else:
            skipped_files.append(original_filename)

    logs.append(f"Found {len(valid_files)} PDF/DOCX files to process.")
    if skipped_files:
        logs.append(f"Skipping {len(skipped_files)} non-PDF/DOCX files: {', '.join(skipped_files)}")
    logs.append("-" * 20)

    if not valid_files:
        logs.append("ERROR: No valid (.pdf or .docx) assignment files found among uploads.")
        return "\n".join(logs), pd.DataFrame(), None

    # 4. Process Files
    temp_feedback_dir = None # Initialize temporary directory path
    try:
        # Create a temporary directory to store individual feedback files before zipping
        temp_feedback_dir = tempfile.mkdtemp()
        logs.append(f"Created temporary directory for feedback files: {temp_feedback_dir}")

        for i, file_info in enumerate(valid_files):
            filename = file_info['orig_name']
            file_path = file_info['path']
            logs.append(f"Processing file {i+1}/{len(valid_files)}: {filename} ...")

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
                     logs.append(f"  ERROR: {llm_response}")
                else:
                     status = "Text Extracted"
                     logs.append("  Text extracted successfully.")

            except Exception as e:
                critical_error_msg = f"CRITICAL ERROR during text extraction for {filename}: {e}"
                logs.append(f"  {critical_error_msg}")
                # traceback.print_exc() # Keep commented unless debugging heavily
                extracted_text = f"Critical extraction error: {e}"
                llm_response = extracted_text
                status = "Text Extraction Failed (Critical)"

            # Grade with Gemini (only if text extraction succeeded)
            api_call_log = ""
            if status == "Text Extracted":
                logs.append("  Sending to Gemini for grading...")
                try:
                    # Pass the clean extracted_text to Gemini
                    llm_response, api_call_log = grade_text_with_gemini(api_key, rubric, extracted_text, model_name=MODEL_NAME)

                    if api_call_log: # Log warnings/errors from the API call function
                        logs.append(f"  Gemini API Info/Warning: {api_call_log}")

                    # Check response for known error patterns
                    if llm_response.startswith("Error during Gemini API call:") or \
                       llm_response.startswith("API call blocked") or \
                       llm_response.startswith("API returned an empty response") or \
                       llm_response.startswith("Skipped API call"):
                        status = "Grading Failed (API Error)"
                    elif llm_response.startswith("Error extracting text:"):
                        status = "Grading Failed (Extraction Error)" # Should be caught earlier, but double check
                    else:
                        status = "Grading Complete"
                    logs.append(f"  Status: {status}")

                except Exception as e:
                     critical_error_msg = f"CRITICAL ERROR during API call for {filename}: {e}"
                     logs.append(f"  {critical_error_msg}")
                     # traceback.print_exc()
                     llm_response = f"Critical API error: {e}"
                     status = "Grading Failed (Critical)"

                # Add delay between API calls
                if i < len(valid_files) - 1: # Don't sleep after the last file
                    logs.append(f"  Waiting {API_DELAY}s before next API call...")
                    time.sleep(API_DELAY)


            # Store result for CSV
            results.append({
                "Filename": filename,
                "Processing_Status": status,
                # "Extracted_Text": extracted_text, # Optional: Uncomment to include full text in CSV (can make file large)
                "LLM_Response": llm_response # This is the feedback/grade or an error message
            })

            # Prepare individual feedback file content
            feedback_content = f"--- Feedback for: {filename} ---\n\n"
            feedback_content += f"Processing Status: {status}\n\n"
            feedback_content += f"Rubric Used:\n------------\n{rubric[:200]}... (truncated)\n------------\n\n" # Include snippet of rubric
            feedback_content += f"AI Generated Feedback/Grade:\n============================\n{llm_response}\n============================"

            # Save individual feedback to a file in the temporary directory
            try:
                # Sanitize filename slightly for the output text file
                safe_filename_base = os.path.splitext(filename)[0].replace(" ", "_").replace("/", "_")
                feedback_filename = f"{safe_filename_base}_feedback.txt"
                feedback_filepath = os.path.join(temp_feedback_dir, feedback_filename)
                with open(feedback_filepath, 'w', encoding='utf-8') as f_out:
                    f_out.write(feedback_content)
                logs.append(f"  Individual feedback saved to temp file: {feedback_filename}")
            except Exception as e:
                 logs.append(f"  ERROR: Could not write individual feedback file for {filename}: {e}")


        # 5. Create Zip file of individual feedback
        if any(r["Processing_Status"] == "Grading Complete" for r in results): # Only zip if something was likely generated
            try:
                # Create the zip file in a location Gradio can access (like the system's temp dir)
                # shutil.make_archive returns the path to the created archive
                zip_base_name = os.path.join(tempfile.gettempdir(), "individual_feedback")
                output_zip_path = shutil.make_archive(zip_base_name, 'zip', temp_feedback_dir)
                logs.append("-" * 20)
                logs.append(f"Successfully created zip file of individual feedback: {os.path.basename(output_zip_path)}")
            except Exception as e:
                logs.append(f"ERROR: Failed to create zip file from feedback directory: {e}")
                output_zip_path = None # Ensure no zip path is returned on error
        else:
            logs.append("-" * 20)
            logs.append("Skipping zip file creation as no files were successfully graded.")


    except Exception as e:
        critical_error_msg = f"CRITICAL ERROR during main processing loop or zipping: {e}"
        logs.append(critical_error_msg)
        traceback.print_exc() # Log full traceback to console where Gradio runs
        status = "Processing Failed (Critical)"
        # Add a generic error to results if loop failed badly
        if not results:
             results.append({
                "Filename": "N/A",
                "Processing_Status": status,
                "LLM_Response": critical_error_msg
            })
    finally:
        # Clean up the temporary directory containing individual files
        if temp_feedback_dir and os.path.exists(temp_feedback_dir):
            try:
                shutil.rmtree(temp_feedback_dir)
                logs.append(f"Cleaned up temporary feedback directory: {temp_feedback_dir}")
            except Exception as e:
                logs.append(f"Warning: Failed to clean up temporary directory {temp_feedback_dir}: {e}")


    # 6. Prepare Final Outputs
    logs.append("\n" + "=" * 30)
    logs.append("Processing complete.")
    if results:
        logs.append(f"See summary table below. Download feedback zip file if generated.")
    else:
        logs.append("No results generated.")
    logs.append("=" * 30)
    logs.append("\n**Reminder:** Always review the LLM's output carefully. Use this as a tool to assist grading.")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Return log string, dataframe, and path to the zip file
    return "\n".join(logs), df, output_zip_path


# --- Gradio Interface Definition ---

def update_rubric_input_visibility(choice):
    """Updates visibility of rubric file upload vs text area based on radio choice."""
    if choice == "Upload File":
        return gr.update(visible=True), gr.update(visible=False)
    elif choice == "Paste Text":
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False)

# Use Blocks for more layout control
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI Bulk Assignment Grader using Gemini")
    gr.Markdown("Upload student assignments (PDF/DOCX), provide a grading rubric, and get AI-generated feedback based *only* on that rubric. Results include a summary table and downloadable individual feedback files.")

    with gr.Row():
        with gr.Column(scale=1):
            api_key_input = gr.Textbox(
                label="Google AI Studio API Key",
                placeholder="Enter your API key (or leave blank to use .env)",
                value=DEFAULT_API_KEY, # Pre-populate if found in .env
                type="password",
                lines=1
            )
            gr.Markdown("*(API Key is required for grading)*")

            rubric_source_radio = gr.Radio(
                choices=["Upload File", "Paste Text"],
                label="Rubric Source",
                value="Upload File" # Default choice
            )

            # File upload - visible by default
            rubric_file_input = gr.File(
                label="Upload Rubric File (.txt)",
                file_types=[".txt"],
                visible=True # Matches default radio button
            )

            # Text Area - hidden by default
            rubric_text_input = gr.Textbox(
                label="Paste Rubric Text Here",
                lines=10,
                placeholder="Paste your detailed grading rubric here...",
                visible=False # Hidden initially
            )

            # Link radio button changes to visibility updates
            rubric_source_radio.change(
                fn=update_rubric_input_visibility,
                inputs=rubric_source_radio,
                outputs=[rubric_file_input, rubric_text_input]
            )

            assignment_files_input = gr.File(
                label="Upload Assignment Files (.pdf, .docx)",
                file_count="multiple",
                file_types=[".pdf", ".docx"]
            )

            start_button = gr.Button("Start Grading Process", variant="primary")

        with gr.Column(scale=2):
            log_output = gr.Textbox(
                label="Processing Log",
                lines=20,
                interactive=False,
                placeholder="Processing steps and errors will appear here..."
            )
            summary_output = gr.DataFrame(
                label="Grading Summary Results",
                headers=["Filename", "Processing_Status", "LLM_Response"],
                wrap=True
                )
            individual_files_output = gr.File(
                label="Download Individual Feedback (ZIP)",
                interactive=False # Output only
                )

    # Connect the button click to the processing function
    start_button.click(
        fn=process_assignments,
        inputs=[
            api_key_input,
            rubric_source_radio,
            rubric_file_input,
            rubric_text_input,
            assignment_files_input
        ],
        outputs=[
            log_output,
            summary_output,
            individual_files_output
        ]
    )

    gr.Markdown("---")
    gr.Markdown("**Important:** Review all AI-generated feedback carefully. This tool is an assistant, not a replacement for instructor judgment. Ensure the feedback aligns with your course standards and the specific assignment context.")

# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio App...")
    # share=True creates a public link (useful for sharing temporarily)
    # In production, consider security implications of sharing.
    demo.launch(share=False)