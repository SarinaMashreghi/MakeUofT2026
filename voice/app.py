import os
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('models/gemini-2.5-flash')
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_text = request.json.get("text")
    if not user_text:
        return jsonify({"error": "No text provided"}), 400
    
    # Generate response
    response = model.generate_content(user_text)
    return jsonify({"reply": response.text})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)