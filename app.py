import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model 
import socket
import numpy as np
import pickle
from PIL import Image
UPLOAD_FOLDER = 'static/uploads'

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
app = Flask(__name__)
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'jpg'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

with open("models/class_names", "rb") as fp:   # Unpickling
    class_names = pickle.load(fp)
hostname = socket.gethostname()
# Route that will process the file upload
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))#save the file in the uploads folder
            model = load_model('models/model1.h5')#load the model
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            image = Image.open(file_path)
            image = image.resize((256, 256))  # Resize the image to match the input size of the model
            image = np.array(image) / 255.0  # Normalize the image pixel values

            # Reshape the image to match the input shape of the model
            image = np.expand_dims(image, axis=0)

            # Make the prediction
            prediction = model.predict(image)

            predicted_class = np.argmax(prediction)
            predicted_class_name = class_names[predicted_class]
            output = predicted_class_name
            
            

            return render_template('upload.html', filename=filename, output = output, hostname = hostname)#passes the filename in the upload.html file and then renders it, cont in the upload.html file
        else:
            flash("File not allowed. Please upload a .jpg file.")
            return redirect(request.url)
    return render_template('upload.html')

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename = 'uploads/' + filename), code=301) #after being called in the upload.html file, it will redirect to the image
    
@app.route('/home')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
