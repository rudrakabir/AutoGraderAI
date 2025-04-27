import gradio as gr
import google.generativeai as genai
import os
import fitz  # PyMuPDF
from docx import Document
import traceback # For detailed error messages
import time
from dotenv import load_dotenv # To load .env file

# --- Configuration ---
# Attempt to load API key from .env file for security
load_dotenv()
DEFAULT_API_KEY = os.getenv("GOOGLE_API_KEY", "") # Use empty string if not found

# Model Name - Ensure you have access to this model via your API key
MODEL_NAME = "gemini-1.5-pro-latest" # Or "gemini-pro" if 1.5 isn't available/needed

# --- Helper Functions ---

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        # Basic cleaning - replace multiple newlines/spaces
        text = ' '.join(text.split())
        return text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        traceback.print_exc() # Print full traceback for debugging
        raise ValueError(f"Could not read PDF file. It might be corrupted or password protected. Error: {e}")

def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file."""
    try:
        doc = Document(docx_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        # Basic cleaning - replace multiple newlines/spaces
        text = ' '.join(text.split())
        return text
    except Exception as e:
        print(f"Error reading DOCX {docx_path}: {e}")
        traceback.print_exc() # Print full traceback for debugging
        raise ValueError(f"Could not read DOCX file. It might be corrupted. Error: {e}")

def construct_prompt(rubric, student_text):
    """Constructs the prompt for the Gemini API."""
    # --- THIS IS THE MOST IMPORTANT PART TO CUSTOMIZE ---
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
5.  **Format your response clearly.** Start with the overall grade/score. Then, provide detailed feedback, perhaps section by section according to the rubric.

**Example Output Structure (Adapt based on Rubric):**

Overall Grade: [Insert Grade/Score based on Rubric]

Strengths (according to Rubric):
*   [Point related to rubric criterion 1, e.g., "Thesis statement clearly articulated (Rubric Section A)."]
*   [Point related to rubric criterion 2, e.g., "Strong use of evidence in paragraphs 2 and 4 (Rubric Section C)."]

Areas for Improvement (according to Rubric):
*   [Point related to rubric criterion 3, e.g., "Paragraph transitions could be smoother (Rubric Section B)."]
*   [Point related to rubric criterion 4, e.g., "Conclusion needs to synthesize arguments more effectively (Rubric Section D)."]
*   [Point related to rubric criterion 5, e.g., "Minor grammatical errors noted (Rubric Section E)."]

Detailed Comments:
[Optional: More elaborate discussion, tying specific examples from the text to rubric points.]

---
**Begin Evaluation:**
"""
    return prompt

def grade_assignment_with_gemini(api_key, rubric, student_text):
    """Sends the prompt to the Gemini API and returns the response."""
    if not api_key:
        raise ValueError("Google AI Studio API Key is required.")
    if not rubric:
        raise ValueError("Grading Rubric cannot be empty.")
    if not student_text:
        raise ValueError("Extracted student text is empty.")

    try:
        # Configure the generative AI client
        genai.configure(api_key=api_key)

        # Set up the model
        generation_config = {
            "temperature": 0.5, # Adjust for more deterministic vs creative output
            "top_p": 0.95,
            "top_k": 0,
            "max_output_tokens": 8192, # Gemini 1.5 Pro has a large context window
        }
        safety_settings = [ # Adjust safety settings if needed, be cautious
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        model = genai.GenerativeModel(model_name=MODEL_NAME,
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)

        prompt = construct_prompt(rubric, student_text)

        # Simple retry mechanism for potential transient API issues
        attempts = 3
        for i in range(attempts):
            try:
                print(f"--- Sending Prompt to Gemini (Attempt {i+1}) ---")
                # print(prompt) # Uncomment to debug the exact prompt being sent
                print("--- End Prompt ---")
                response = model.generate_content(prompt)
                print("--- Received Response from Gemini ---")
                # print(response.text) # Uncomment to debug raw response
                print("--- End Response ---")
                
                # Check for safety blocks or empty responses
                if not response.parts:
                     if response.prompt_feedback.block_reason:
                         raise ValueError(f"API call blocked due to safety settings. Reason: {response.prompt_feedback.block_reason}")
                     else:
                          raise ValueError("API returned an empty response. The prompt might be unsuitable or there was an unknown issue.")

                return response.text

            except Exception as api_error:
                print(f"API Error on attempt {i+1}: {api_error}")
                if i < attempts - 1:
                    time.sleep(2) # Wait before retrying
                else:
                    # Re-raise the exception after final attempt fails
                    raise api_error

    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        traceback.print_exc()
        # Provide a more user-friendly error message back to Gradio
        error_message = f"An error occurred while communicating with the Gemini API: {e}\n\n"
        error_message += "Check:\n"
        error_message += "- Your API key is correct and has access to the model.\n"
        error_message += "- Your internet connection.\n"
        error_message += f"- The Gemini model ('{MODEL_NAME}') is available in your region.\n"
        error_message += "- Your prompt/rubric doesn't violate safety settings.\n\n"
        error_message += f"Details:\n{traceback.format_exc()}" # Include traceback for detailed debugging
        raise RuntimeError(error_message)


# --- Gradio Interface Function ---

def process_and_grade(file_obj, rubric, api_key):
    """Main function called by Gradio interface."""
    if file_obj is None:
        return "Please upload a file.", "Please upload a file first."
    if not rubric:
        return "Rubric is empty.", "Please provide the grading rubric."
    if not api_key:
        # Try getting from environment if not provided in field
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
             return "API Key Missing.", "Please enter your Google AI Studio API Key or set the GOOGLE_API_KEY environment variable."

    extracted_text = "Error: Could not extract text."
    llm_response = "Error: Grading could not be performed."
    file_path = file_obj.name # Gradio provides a temporary file path

    try:
        # 1. Extract Text
        if file_path.lower().endswith(".pdf"):
            print(f"Extracting text from PDF: {file_path}")
            extracted_text = extract_text_from_pdf(file_path)
        elif file_path.lower().endswith(".docx"):
            print(f"Extracting text from DOCX: {file_path}")
            extracted_text = extract_text_from_docx(file_path)
        else:
            extracted_text = "Unsupported file type. Please upload PDF or DOCX."
            return extracted_text, llm_response # Return early

        if not extracted_text or extracted_text.isspace():
             extracted_text = "Extracted text was empty. Check the file content."
             return extracted_text, llm_response # Return early

        # 2. Grade with Gemini
        print("Sending extracted text to Gemini for grading...")
        llm_response = grade_assignment_with_gemini(api_key, rubric, extracted_text)
        print("Grading complete.")

        # Limit displayed extracted text length for readability in Gradio UI
        display_text = extracted_text[:5000] + ("..." if len(extracted_text) > 5000 else "")
        return display_text, llm_response

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        # Catch specific errors from helper functions or API calls
        print(f"Caught Error: {e}")
        error_message = f"An error occurred:\n{e}"
        # Display error in both output fields for clarity
        return error_message, error_message
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        error_message = f"An unexpected critical error occurred:\n{e}\n\n{traceback.format_exc()}"
        # Display error in both output fields for clarity
        return error_message, error_message
    finally:
         # Clean up the temporary file if it exists (Gradio might handle this, but good practice)
         if file_path and os.path.exists(file_path):
              try:
                   # os.remove(file_path) # Sometimes Gradio needs the file longer, disable explicit removal
                   pass
              except Exception as cleanup_error:
                   print(f"Warning: Could not remove temporary file {file_path}: {cleanup_error}")


# --- Build Gradio Interface ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI Writing Assignment Grader (using Gemini)")
    gr.Markdown("Upload a student's assignment (PDF or DOCX), provide the grading rubric and your Google AI Studio API Key, then click 'Grade Assignment'.")

    with gr.Row():
        with gr.Column(scale=1):
            api_key_input = gr.Textbox(
                label="Google AI Studio API Key",
                placeholder="Enter your API key (or set GOOGLE_API_KEY env var)",
                type="password",
                value=DEFAULT_API_KEY # Pre-fill if found in .env
            )
            file_upload = gr.File(
                label="Upload Assignment (PDF or DOCX)",
                file_types=[".pdf", ".docx"]
            )
            rubric_input = gr.Textbox(
                label="Grading Rubric",
                placeholder="Paste your detailed grading rubric here...\nExample:\nSection A: Thesis Statement (10 pts)\n- Clear and arguable?\nSection B: Evidence (30 pts)\n- Relevant examples?\n...",
                lines=15
            )
            submit_button = gr.Button("Grade Assignment", variant="primary")

        with gr.Column(scale=2):
            extracted_text_output = gr.Textbox(
                label="Extracted Text (Preview - first 5000 chars)",
                interactive=False,
                lines=10
            )
            grading_result_output = gr.Textbox(
                label="Gemini Grading Result",
                interactive=False,
                lines=20
            )

    # Define interaction
    submit_button.click(
        fn=process_and_grade,
        inputs=[file_upload, rubric_input, api_key_input],
        outputs=[extracted_text_output, grading_result_output]
    )

    gr.Markdown("---")
    gr.Markdown("**Disclaimer:** AI grading is a tool to assist, not replace, human judgment. **Always review the AI's output carefully.** Ensure the rubric is detailed and clear for best results. Protect your API key.")

# --- Launch the App ---
if __name__ == "__main__":
    # Set share=True to get a public link (use with caution, especially with API keys)
    # Set debug=True for more detailed logs in the console
    demo.launch(debug=True)