from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # For consistent results
import pytesseract
from PIL import Image
import os
import json
import fitz  # PyMuPDF
import pdfplumber
import csv

app = Flask(__name__, template_folder=r"C:\Users\DELL\Downloads\frjsonfl")

# Configuration
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'jfif', 'webp', 'bmp', 'pdf'}  # Added 'pdf'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return str(e)

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        # Extract text using PyMuPDF
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text()
            text += page_text + "\n"
        
        # Extract text using pdfplumber for additional text extraction
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                text += page_text + "\n"
    except Exception as e:
        return f"Error extracting text: {e}"
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        if filename.lower().endswith('.pdf'):
            # Extract text from PDF
            text = extract_text_from_pdf(file_path)
        else:
            # Extract text from image
            text = extract_text_from_image(file_path)

        # Generate JSON and CSV output
        json_output = json.dumps({'text': text}, indent=4)
        csv_output = 'text\n' + text.replace('\n', '\n')

        return jsonify({
            'text': text,
            'json': json_output,
            'csv': csv_output
        })
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)


#with nltk
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from langdetect import detect, DetectorFactory
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import pytesseract
from PIL import Image
import os
import json
import fitz  # PyMuPDF
import pdfplumber
import csv
import openai  # For Generative AI

# Ensure NLTK data is downloaded
import nltk
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')

# Set up OpenAI API key
openai.api_key = 'your-openai-api-key'

# Initialize Flask app
app = Flask(__name__, template_folder=r"C:\Users\DELL\Downloads\frjsonfl")

# Configuration
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'jfif', 'webp', 'bmp', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return str(e)

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        # Extract text using PyMuPDF
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text()
            text += page_text + "\n"
        
        # Extract text using pdfplumber for additional text extraction
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                text += page_text + "\n"
    except Exception as e:
        return f"Error extracting text: {e}"
    return text

def analyze_text(text):
    # Tokenization
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    
    # POS Tagging
    pos_tags = pos_tag(words)
    
    # Named Entity Recognition
    named_entities = ne_chunk(pos_tags)
    
    # Frequency Distribution
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    freq_dist = FreqDist(filtered_words)
    
    return {
        'word_count': len(words),
        'sentence_count': len(sentences),
        'named_entities': named_entities,
        'frequency_distribution': freq_dist.most_common(10)
    }

def generate_text_with_ai(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # Use the appropriate engine
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        if filename.lower().endswith('.pdf'):
            # Extract text from PDF
            text = extract_text_from_pdf(file_path)
        else:
            # Extract text from image
            text = extract_text_from_image(file_path)
        
        # Analyze the extracted text
        text_analysis = analyze_text(text)
        
        # Optionally generate additional text
        ai_generated_text = generate_text_with_ai(f"Enhance the following text: {text[:500]}")  # Limit prompt size

        # Generate JSON and CSV output
        json_output = json.dumps({'text': text, 'analysis': text_analysis, 'ai_generated_text': ai_generated_text}, indent=4)
        csv_output = 'text\n' + text.replace('\n', '\n')

        return jsonify({
            'text': text,
            'analysis': text_analysis,
            'ai_generated_text': ai_generated_text,
            'json': json_output,
            'csv': csv_output
        })
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
