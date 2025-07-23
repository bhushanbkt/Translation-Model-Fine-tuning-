# app.py

import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

st.set_page_config(page_title="Cultural Idiom Translator")

st.title("🈯 Cultural Idiom Translator (Hindi ➡️ English)")

tokenizer = T5Tokenizer.from_pretrained("./t5_cultural_idioms")
model = T5ForConditionalGeneration.from_pretrained("./t5_cultural_idioms")

def translate(text):
    input_text = "translate Hindi to English: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=64, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

user_input = st.text_area("Enter a Hindi Idiom:", "रस्सी जल गई पर बल नहीं गया")

if st.button("Translate"):
    translation = translate(user_input)
    st.success(f"✅ Translation: {translation}")
