import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Load model
model = tf.keras.models.load_model("asl_alphabet_model.keras")


# Class labels

class_names = [

'A','B','C','D','E',

'F','G','H','I','J',

'K','L','M','N','O',

'P','Q','R','S','T',

'U','V','W','X','Y'

]


st.title("🤟 ASL Alphabet Recognition")

uploaded_file = st.file_uploader(

"Upload Hand Sign Image",

type=['jpg','jpeg','png']

)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(

        image,

        caption="Uploaded Image",

        width=250

    )


    image = image.convert("L")

    image = image.resize((28,28))


    img_array = np.array(image)


    img_array = img_array/255.0


    img_array = img_array.reshape(

        1,28,28,1

    )


    prediction = model.predict(img_array)


    predicted_class = np.argmax(prediction)


    confidence = np.max(prediction)*100


    st.success(

        f"Prediction: {class_names[predicted_class]}"

    )


    st.write(

        f"Confidence: {confidence:.2f}%"

    )