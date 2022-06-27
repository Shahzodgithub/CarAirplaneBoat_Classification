import streamlit as st
from fastai.vision.all import *

# title 
st.title("Transportni klassifikatsiya qiluvchi model")
file = st.file_uploader('Ramni yuklash', type=['png','jpeg','gif','svg'])
if file: 
    st.image(file)
    img = PILImage.create(file)

    #model yuklash 
    model = load_learner('transport_model.pkl')

    # model baholash
    pred,pred_id, probs = model.predict(img)
    st.success(f'Bashorat: {pred}')
    st.info(f'Ehtimollik: {probs[pred_id]*100: .1f}%')