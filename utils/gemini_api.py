# Functions for communicating with Gemini API
import google.generativeai as genai
from config import Config

# Set up the API key
genai.configure(api_key=Config.GEMINI_API_KEY)

def classify_waste(image_file):
    # Example: Prepare and send the image data to the Gemini model
    with open(image_file, "rb") as img:
        image_data = img.read()
    
    response = genai.generate_text(
        prompt="Classify this image and suggest waste disposal method.",
        image_data=image_data
    )
    
    # Handle the response (for demonstration, you may need to adjust per API specifications)
    classification = response.get("text", "Unable to classify")
    return classification
