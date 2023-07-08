from flask import Flask, render_template, request
import numpy as np
import os
import pickle
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def main():
  return render_template('index.html')


@app.route('/home', methods=['POST'])
def home():
  file = request.files.get('image')
  file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
  file_path = app.config['UPLOAD_FOLDER'] + '/' + file.filename
  image = Image.open(file_path)
  grayscale_image = image.convert('L')
  resized_image = grayscale_image.resize((28, 28))

  test_number = np.expand_dims(resized_image, axis=0)
  test_number = np.expand_dims(test_number, axis=-1)

  with open('model.pkl', 'rb') as file:
    pickled_model = pickle.load(file)

  preds = pickled_model.predict(test_number)
  labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  predicted_indices = np.argmax(preds, axis=1)
  predicted_numbers = predicted_indices.tolist()
  predicted_labels = [labels[index] for index in predicted_numbers]

  return render_template('prediction.html', data=predicted_labels)


if __name__ == '__main__':
  app.run(debug="True")
