import os
import google.generativeai as genai

def load_gemini():
  # Load Gemini model
  api_key = os.getenv('GOOGLE_API_KEY')
  if not api_key:
      raise ValueError("GOOGLE_API_KEY not found in .env file")
  genai.configure(api_key=api_key)
  model = genai.GenerativeModel('gemini-1.5-pro-latest')  # Use the appropriate model name
  return model, None
