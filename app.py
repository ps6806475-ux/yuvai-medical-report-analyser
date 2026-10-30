import streamlit as st
import easyocr
from google import genai
from PIL import Image
import numpy as np

# 1. Web UI Header
st.set_page_config(page_title="Rural Health AI", page_icon="🩺")
st.title("🩺 AI Medical Report Analyzer")
st.write("Upload a lab report to get a simple explanation in your local language.")

# 2. Secure API Key Input
api_key = st.secrets["GEMINI_API_KEY"]

# 3. Language Dropdown Menu
languages = [
    "English", "Hindi", "Bengali", "Marathi", "Telugu", "Tamil", 
    "Gujarati", "Urdu", "Kannada", "Odia", "Malayalam", "Punjabi", 
    "Assamese", "Maithili", "Santali", "Kashmiri", "Nepali", 
    "Sindhi", "Dogri", "Konkani", "Manipuri", "Bodo"
]
target_language = st.selectbox("🌍 Select Language / भाषा चुनें:", languages)

# 4. Drag-and-Drop Image Uploader
uploaded_file = st.file_uploader("Upload Medical Report (Image)", type=["jpg", "png", "jpeg", "webp"])

# 5. The "Analyze" Button
if st.button("Analyze Report"):
    if not api_key:
        st.warning("⚠️ Please enter your API Key first.")
    elif not uploaded_file:
        st.warning("⚠️ Please upload an image first.")
    else:
        # Show a loading spinner while the AI works
        with st.spinner(f"Analyzing report and translating to {target_language}..."):
            try:
                # Convert the uploaded web file into an image EasyOCR can read
                image = Image.open(uploaded_file)
                image_np = np.array(image) 

                # Extract Text using CPU (works perfectly on free cloud servers)
                reader = easyocr.Reader(['en'], gpu=False)
                extracted_data = reader.readtext(image_np, detail=0)
                messy_ocr_text = "\n".join(extracted_data)

                # Connect to Gemini
                client = genai.Client(api_key=api_key)
                
                # The precise prompt we perfected earlier
                prompt = f"""
                You are an educational health assistant for rural communities in India. 
                I am going to give you messy text from an OCR scan of a medical lab report. 

                For each test you find, provide:
                1. Test Name: (Correct the spelling in English)
                2. Result: (Extract just the number and unit)
                3. Standard Range: (Extract the normal reference range)
                4. Status: (Write '🟢 Normal', '🔴 High', or '🔴 Low')
                5. Simple Explanation: (Explain this parameter using everyday layman's terms. Do NOT diagnose diseases. WRITE THIS ENTIRE EXPLANATION IN {target_language.upper()}.)

                Messy text:
                {messy_ocr_text}
                """

                # Call the latest model
                response = client.models.generate_content(
                    model='gemini-3.8-flash',
                    contents=prompt
                )

                # Display the AI's output on the website
                st.success("✅ Analysis Complete!")
                st.markdown(response.text)

            except Exception as error:
                st.error(f"⚠️ Error: {error}")
