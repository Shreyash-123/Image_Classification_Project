import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf


# Function for MobileNetV2 ImageNet model
def mobilenetv2_imagenet():
    st.title("\U0001F4F8 Image Classification with MobileNetV2")

    uploaded_file = st.file_uploader("Choose an image to classify:", type=["jpg", "png"], help="Upload an image for classification using MobileNetV2.")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Convert image to RGB if it has an alpha channel
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        st.image(image, caption='Uploaded Image', use_column_width=False, width=300)
        st.write("Classifying...")

        # Load MobileNetV2 model
        model = tf.keras.applications.MobileNetV2(weights='imagenet')

        # Preprocess the image
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        # Make predictions
        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]

        st.subheader("Prediction:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.subheader(f"**{label}**: {score * 100:.2f}%")


# Function for CIFAR-10 model
def cifar10_classification():
    st.title("\U0001F680 CIFAR-10 Image Classification")
    
    uploaded_file = st.file_uploader("Choose an image to classify:", type=["jpg", "png"], help="Upload an image for classification using the CIFAR-10 model.")

    model_path = 'cnn_model.keras'

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Convert image to RGB if it has an alpha channel
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        st.image(image, caption='Uploaded Image', use_column_width=False, width=300)
        st.write("Classifying...")

        # load model
        model = tf.keras.models.load_model(model_path)

        # CIFAR-10 class names
        class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

        # Preprocess the image
        img = image.resize((32, 32))
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        st.subheader(f"**Predicted Class:** {class_names[predicted_class]}")
        st.subheader(f"**Confidence:** {confidence * 100:.2f}%")


# Main function to control the navigation
def main():
    st.sidebar.title("\U0001F4D6 Navigation")
    st.sidebar.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .radio-text {
        font-size: 24px;
        font-weight: bold;
        color: #FF5733;
        margin-bottom: 10px; /* Remove margin below the title text */
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("<div class='radio-text'>Select a Model:</div>", unsafe_allow_html=True)
    choice = st.sidebar.radio(" ", ("CIFAR-10", "MobileNetV2 (ImageNet)"), label_visibility='collapsed')

    if choice == "MobileNetV2 (ImageNet)":
        mobilenetv2_imagenet()
    elif choice == "CIFAR-10":
        cifar10_classification()

        
if __name__ == "__main__":
    main()
