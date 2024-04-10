import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Load the pre-trained ResNet50 model
#model = tf.keras.applications.ResNet50(weights='imagenet')
model = tf.keras.models.load_model('E:\DATA SCIENCE PRACTICE\simplilearn\Grade projects\Deep learning with tensorflow\Ship classification\model.h5')
# Define the labels
labels = {'buoy': 0,
 'cruise_ship': 1,
 'ferry_boat': 2,
 'freight_boat': 3,
 'gondola': 4,
 'inflatable_boat': 5,
 'kayak': 6,
 'paper_boat': 7,
 'sailboat': 8}

rev_dict = {}
for k,v in labels.items():
    rev_dict[v] = k

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((192, 192))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to make predictions
def predict(image):
    img_array = preprocess_image(image)
    preds = model.predict(img_array)
    #pred_label = decode_predictions(preds, top=1)[0][0]
    predicted_class = np.argmax(preds)
    #return pred_label[1]
    return rev_dict[predicted_class]

# Streamlit UI
st.title("Ship Predictor")
st.write("Upload an image to predict whether it contains a type of ship.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make prediction when the 'Predict' button is clicked
    if st.button("Predict"):
        with st.spinner("Predicting..."):
            prediction = predict(image)
            st.success(f"Predicted Ship Name : {prediction}")
