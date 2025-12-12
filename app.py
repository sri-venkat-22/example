import os
import io
import pickle
import string
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import shutil  # Import shutil for moving the file

# --- NLTK Data Check ---
try:
    stopwords.words('english')
    WordNetLemmatizer().lemmatize('test')
except LookupError:
    print("NLTK data not found. Downloading...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# --- Initialization ---
app = Flask(__name__)

# --- Base Directory Setup ---
# 1. Define the ABSOLUTE BASE DIRECTORY for RawData
BASE_DATA_DIR = '/Users/srivenkatreddy/Documents/Projects/Document_classifier/RawData'

# 2. Define a temporary folder for initial upload and classification
TEMP_UPLOAD_FOLDER = 'uploads'
app.config['TEMP_UPLOAD_FOLDER'] = TEMP_UPLOAD_FOLDER
os.makedirs(TEMP_UPLOAD_FOLDER, exist_ok=True)

# 3. Define the directory map for routing the files
CATEGORY_DIR_MAP = {
    # Keys must match the exact labels output by label_encoder.inverse_transform()
    'Taxes': os.path.join(BASE_DATA_DIR, 'Taxes'),
    'Agreement': os.path.join(BASE_DATA_DIR, 'Agreements'),
    'Deeds': os.path.join(BASE_DATA_DIR, 'Deeds'),
    'Human Resources': os.path.join(BASE_DATA_DIR, 'Human Resources'),
    'Valuations': os.path.join(BASE_DATA_DIR, 'Valuations')
}

# Ensure all target directories exist
for path in CATEGORY_DIR_MAP.values():
    os.makedirs(path, exist_ok=True)

# Global variables for ML components
vectorizer = None
label_encoder = None
classifier = None

# Load the trained model components
try:
    vectorizer = pickle.load(open('tfidf.pkl', 'rb'))
    label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
    classifier = pickle.load(open('nbmodel.pkl', 'rb'))

    CLASSIFICATION_LABELS = label_encoder.classes_.tolist()

    print("ML components loaded successfully.")

except FileNotFoundError as e:
    print("--- CRITICAL ERROR ---")
    print(f"Required ML file not found: {e.filename}")
    print("Please ensure 'tfidf.pkl', 'label_encoder.pkl', and 'nbmodel.pkl' are in the same directory as app.py.")
    classifier = None


# --- PDF Text Extraction ---
def convert_pdf_to_txt(filepath):
    """Extract full text from a single PDF file using pdfminer."""
    full_text = ""
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)

    try:
        with open(filepath, 'rb') as fp:
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            for page in PDFPage.get_pages(fp, caching=True, check_extractable=True):
                interpreter.process_page(page)
        full_text = retstr.getvalue()
    except Exception as e:
        print(f"Error during PDF extraction: {e}")
        return ""
    finally:
        device.close()
        retstr.close()

    return full_text


# --- Text Preprocessing ---
def preprocess_text(text_data):
    """Applies the same preprocessing steps as in the notebook."""
    if not text_data:
        return ""

    text = " ".join(text_data.lower().split())
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))

    stop = stopwords.words('english')
    text = " ".join(word for word in text.split() if word not in stop)

    stemmer = WordNetLemmatizer()
    text = " ".join(stemmer.lemmatize(word) for word in text.split())

    text = text.replace('shall', '')
    text = text.replace('may', '')

    return text


# --- Flask Routes ---

@app.route('/', methods=['GET'])
def home():
    """Renders the upload form."""
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify_document():
    """Handles multiple file uploads, classifies each document, and moves it to the corresponding directory. """
    if classifier is None:
        return "Error: ML components not loaded. Check server console for details.", 500

    # Get the list of uploaded files using getlist
    uploaded_files = request.files.getlist('files')

    # Check if any valid files were uploaded
    if not uploaded_files or uploaded_files[0].filename == '':
        return redirect(url_for('home'))

    results = []

    for file in uploaded_files:
        if file.filename:
            filename = file.filename
            temp_filepath = os.path.join(app.config['TEMP_UPLOAD_FOLDER'], filename)

            # Default result for cleanup/error handling
            result = {
                'filename': filename,
                'category': 'N/A',
                'status': 'Error',
                'message': 'Processing failed.'
            }

            try:
                # 1. Save file to temporary location
                file.save(temp_filepath)

                # 2. Extract, Preprocess, and Vectorize Text
                raw_text = convert_pdf_to_txt(temp_filepath)

                if not raw_text.strip():
                    result['message'] = "Text extraction failed (file might be image-based or empty)."
                    raise Exception("Extraction failure")

                processed_text = preprocess_text(raw_text)
                input_vector = vectorizer.transform(pd.Series([processed_text]))

                # 3. Predict Category
                prediction_index = classifier.predict(input_vector.toarray())
                predicted_category = label_encoder.inverse_transform(prediction_index)[0]

                result['category'] = predicted_category

                # 4. Route and Save the File to the Final Destination
                destination_dir = CATEGORY_DIR_MAP.get(predicted_category)

                if destination_dir:
                    final_filepath = os.path.join(destination_dir, filename)

                    # Move the file
                    shutil.move(temp_filepath, final_filepath)

                    result['status'] = 'Success'
                    result['message'] = f"Moved to: {os.path.basename(destination_dir)}"
                else:
                    result['message'] = f"Classification: {predicted_category}, but routing path is undefined."

            except Exception as e:
                # Catch any error, ensure message is logged, and clean up
                if result['message'] == 'Processing failed.':
                    result['message'] = f"Error: {e}"

            results.append(result)

            # Cleanup: Ensure the file is deleted from temp, even if an error occurred during move
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)

                # Return the aggregated results to the classification template
    return render_template('classify.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)