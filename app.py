from flask import Flask, request, render_template, send_file, jsonify, url_for
from sklearn.decomposition import PCA
from skimage import io, color
from skimage.util import img_as_ubyte
import os

app = Flask(__name__)

# Directories for storing images
UPLOAD_FOLDER = 'uploads'
COMPRESSED_FOLDER = 'compressed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COMPRESSED_FOLDER, exist_ok=True)

def compress_image(img, compression_level):
    # Convert to grayscale
    gray_image = color.rgb2gray(img)
    
    # Flatten image
    flattened_img = gray_image.reshape(gray_image.shape[0], -1)
    
    # Apply PCA for compression
    pca = PCA(n_components=compression_level)
    compressed_data = pca.fit_transform(flattened_img)
    
    # Reconstruct the image
    reconstructed_img = pca.inverse_transform(compressed_data)
    
    # Normalize and convert to 8-bit
    compressed_data_normalized = (reconstructed_img - reconstructed_img.min()) / (reconstructed_img.max() - reconstructed_img.min())
    compressed_img = img_as_ubyte(compressed_data_normalized)
    
    return compressed_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    compression_level = float(request.form.get('compression'))
    uploaded_file = request.files['image']

    if uploaded_file:
        # Save the original image
        original_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
        uploaded_file.save(original_path)

        # Load image for compression
        img = io.imread(original_path)

        # Compress the image
        compressed_img = compress_image(img, compression_level)

        # Save compressed image
        compressed_filename = 'compressed_' + uploaded_file.filename
        compressed_path = os.path.join(COMPRESSED_FOLDER, compressed_filename)
        io.imsave(compressed_path, compressed_img)

        # Return download URLs as JSON
        return jsonify({
            'original_url': url_for('download_file', filename=uploaded_file.filename, folder='uploads'),
            'compressed_url': url_for('download_file', filename=compressed_filename, folder='compressed')
        })
    else:
        return "No file uploaded", 400

@app.route('/download/<folder>/<filename>')
def download_file(folder, filename):
    directory = UPLOAD_FOLDER if folder == 'uploads' else COMPRESSED_FOLDER
    return send_file(os.path.join(directory, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
