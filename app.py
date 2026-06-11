# from flask import Flask, request
# import numpy as np

# app = Flask(__name__)

# model = tf.keras.models.load_model("model.h5")

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = np.array(request.json["data"])
#     prediction = model.predict(data)
#     return {"prediction": prediction.tolist()}

from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from PIL import Image 
app = Flask(__name__)
model = tf.keras.models.load_model("grassmodel.h5")

@app.route("/predict/", methods=["POST"])
def predict():
    from PIL import Image

    image = request.files["imagefile"] 
    if image:
        img_array = tf.keras.utils.img_to_array(image)

    processed_img = img_array / 255.0
    img = Image.open(image.stream)  
    img = img.resize((64, 64))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(processed_img)
    return {"data": predictions.tolist()}



@app.route("/tester/")
def tester():
    return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True)