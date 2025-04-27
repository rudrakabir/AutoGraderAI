import os
import openai
import docx
import fitz  # PyMuPDF
import pandas as pd

# Set your OpenAI API key
openai.api_key = "your-api-key"

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def get_grade_from_llm(text, rubric_prompt):
    messages = [
        {"role": "system", "content": "You are a college writing instructor. Grade essays based on the provided rubric."},
        {"role": "user", "content": rubric_prompt.format(essay=text)},
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.3,
    )
    
    return response['choices'][0]['message']['content']

# Customizable rubric prompt template
rubric_prompt = """
Grade the following essay on a scale of 0 to 10 in each of the following areas: clarity, structure, grammar, and strength of argument. 
Then provide brief bullet-point feedback for each. End with an overall score out of 40.

Essay:
{essay}
"""

def grade_folder(folder_path):
    results = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if file.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        elif file.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        else:
            continue

        print(f"Grading: {file}")
        feedback = get_grade_from_llm(text, rubric_prompt)
        results.append({"File": file, "Feedback and Grade": feedback})

    df = pd.DataFrame(results)
    df.to_csv("grading_results.csv", index=False)
    print("Grading complete. Results saved to grading_results.csv")

# Run it!
if __name__ == "__main__":
    folder = "assignments"  # folder containing your PDF/DOCX files
    grade_folder(folder)
