import tensorflow as tf
import re
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
    
def process_abstract(sample_abstract):
    original_abstract_sentences = sample_abstract[:-1].split(". ")
    original_abstract_sentences = original_abstract_sentences[:len(original_abstract_sentences)]
    for i in range(10):
        sample_abstract=[re.sub(str(i),'@',x) for x in sample_abstract]
        sample_abstract=''.join(sample_abstract)
    abstract_sentences = sample_abstract.split(". ")
    abstract_length = [len(abstract_sentences)] * len(abstract_sentences)
    abstract_chars = [" ".join(list(sentence)) for sentence in abstract_sentences]
    abstract_numbers = range(len(abstract_sentences))
    abstract_df = pd.DataFrame.from_dict({'total_lines': abstract_length, 'line_numbers':abstract_numbers})
    
    one_hot_encoder = OneHotEncoder(sparse=False)
    line_numbers_one_hot = tf.one_hot(abstract_df['line_numbers'].to_numpy(), depth=15)
    total_lines_one_hot = tf.one_hot(abstract_df['total_lines'].to_numpy(), depth=20)
    pos_char_token_data = tf.data.Dataset.from_tensor_slices((line_numbers_one_hot,
                                                              total_lines_one_hot,
                                                              abstract_sentences,
                                                              abstract_chars))
    pos_char_token_labels = tf.data.Dataset.from_tensor_slices(np.zeros(len(abstract_sentences)))
    pos_char_token_dataset = tf.data.Dataset.zip((pos_char_token_data, pos_char_token_labels)).batch(32)
    model = tf.keras.models.load_model("token_char_positional_model")
    probs = model.predict(pos_char_token_dataset)
    preds = preds = np.argsort(-probs)
    sections = ['OBJECTIVES', 'BACKGROUND', 'METHODS', 'RESULTS','CONCLUSION']
    predicted_abstract = ''
    predicted_sections = []
    for pred in preds:
        predicted_sections.append(sections[pred[0]])
    prior_section = ''
    for line, sentence in enumerate(original_abstract_sentences):
        if predicted_sections[line] == prior_section:
            predicted_section = ''
        else:
            predicted_section = f'**:blue[{predicted_sections[line]}:]** '
        predicted_abstract = predicted_abstract + predicted_section + sentence + '. '
        prior_section = predicted_sections[line]
    return predicted_abstract

st.title("Abstract Decoder 	:bookmark_tabs:")
st.write("Abstract Decoder is a tool for structuring abstracts. When submitting an abstract that is not divided into sections, Abstract Decoder will predict the section of each sentence.")
st.write("The predicted sections are: Objectives, Background, Methods, Results and Conclusion.")
st.markdown("The Model was trained on the PubMed 20k RCT dataset, a subset of the [200k RCT dataset](https://arxiv.org/abs/1710.06071).")

user_input = st.text_input("Enter your abstract here:")
if st.button("Submit"):
    st.markdown(process_abstract(user_input))