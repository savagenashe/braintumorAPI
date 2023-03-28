import streamlit as st
import tensorflow as tf
#from tensorflow.keras.applications.imagenet_utils import decode_predictions
import cv2
from PIL import Image, ImageOps
import numpy as np
import io
 
# @st.cache_data(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('modelbraintumor1.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()
 
st.write("""
         # Brain Tumor Classification
         """
         )
 
file = st.file_uploader("Upload the image to be classified", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)

# Define the preprocess function
def upload_predict(image):
    # Convert the image to grayscale
    image = Image.fromarray(image).convert('L')
    # Resize the image to match the input shape of the model
    image = image.resize((224, 224))
    # Convert the image to a numpy array
    image = np.array(image)
    # Convert the image to RGB
    image = np.stack([image] * 3, axis=-1)
    # Normalize the pixel values
    image = image / 255.0
    # Expand the dimensions to match the input shape of the model
    image = np.expand_dims(image, axis=0)
    # return name of the processed image
    return image


# Define the predict function
def predict(image):
    # Convert image to numpy array
    img_array = np.array(image)

    # Resize image to required size
    img_resized = cv2.resize(img_array, (224, 224))

    # Expand dimensions of image to match expected input shape of model
    img_expanded = np.expand_dims(img_resized, axis=0)

    # Normalize pixel values of image to be between 0 and 1
    image = img_expanded / 255.0

    prediction = model.predict(image)
    # Get the class label with the highest score
    predicted_class = np.argmax(prediction, axis=-1)
    # Return the predicted class name ["Glioma  Tumor ","Meningioma Tumor" , "No Tumor" , "Pituitary Tumor"]
    if predicted_class == 0:
        st.write("Glioma  Tumor")
        print("Glioma  Tumor")
    elif predicted_class == 1:
        st.write("Meningioma Tumor")
        print("Meningioma Tumor")
    elif predicted_class == 2:
        st.write("No Tumor")
        print("No Tumor")
    else:
        st.write("Pituitary Tumor")
        print("Pituitary Tumor")


#if no file is uploaded
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predict(image)

