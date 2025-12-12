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
    # Assuming pickle files are in the same directory as app.py
    vectorizer = pickle.load(open('tfidf.pkl', 'rb'))
    label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
    classifier = pickle.load(open('nbmodel.pkl', 'rb'))

    # We must know the exact string labels the model returns for routing (e.g., 'Taxes', not 'Taxes ')
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
    """Handles file upload, classifies the document, and moves it to the corresponding directory. """
    if classifier is None:
        return "Error: ML components not loaded. Check server console for details.", 500

    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = file.filename

        # 1. Save file to temporary location for processing
        temp_filepath = os.path.join(app.config['TEMP_UPLOAD_FOLDER'], filename)

        try:
            file.save(temp_filepath)

            # 2. Extract Text
            raw_text = convert_pdf_to_txt(temp_filepath)

            if not raw_text.strip():
                return render_template('classify.html',
                                       filename=filename,
                                       error="Could not extract text from the PDF. File might be image-based or corrupted.")

            # 3. Preprocess and Vectorize Text
            processed_text = preprocess_text(raw_text)
            input_vector = vectorizer.transform(pd.Series([processed_text]))

            # 4. Predict Category
            prediction_index = classifier.predict(input_vector.toarray())
            predicted_category = label_encoder.inverse_transform(prediction_index)[0]

            # 5. Route and Save the File to the Final Destination

            # Use the prediction to find the correct destination path
            # 5. Route and Save the File to the Final Destination
            destination_dir = CATEGORY_DIR_MAP.get(predicted_category)

            if destination_dir:
                final_filepath = os.path.join(destination_dir, filename)

                # Move the file from the temporary location to the final classified directory
                shutil.move(temp_filepath, final_filepath)  # <--- THIS IS THE MOVE ACTION

                # Success message including the file's final location
                result_message = f"Successfully moved to: {destination_dir}"
            else:
                # Should not happen if CATEGORY_DIR_MAP is complete, but good for safety
                result_message = f"Classification successful, but failed to find routing path for category: {predicted_category}"

            return render_template('classify.html',
                                   filename=filename,
                                   category=predicted_category,
                                   message=result_message)

        except Exception as e:
            # Handle any exceptions during the process (e.g., file move error, model error)
            return render_template('classify.html',
                                   filename=filename,
                                   error=f"An unexpected error occurred: {e}")
        finally:
            # Clean up the file from the temporary folder if it still exists (e.g., if the move failed)
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)

    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)