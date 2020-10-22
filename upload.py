import streamlit as st
from PIL import Image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add
import pickle
import numpy as np

def create_img_encoding_model():
    model = ResNet50(weights="imagenet", input_shape=(224, 224, 3))
    model_new = Model(model.input, model.layers[-2].output)
    return model_new

def create_caption_model():

    with open("./saved/embedding_matrix.pkl", "rb") as f:
        embedding_matrix = pickle.load(f)

    max_len = 35
    vocab_size = 1848

    # Processing Image
    input_img_features = Input(shape=(2048,))
    inp_img1 = Dropout(0.3)(input_img_features)
    inp_img2 = Dense(256, activation='relu')(inp_img1)

    # Processing Caption
    input_captions = Input(shape=(max_len,))
    inp_cap1 = Embedding(input_dim=vocab_size, output_dim=50, mask_zero=True)(input_captions)
    inp_cap2 = Dropout(0.3)(inp_cap1)
    inp_cap3 = LSTM(256)(inp_cap2)


    decoder1 = add([inp_img2, inp_cap3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # Combined Model
    model = Model(inputs=[input_img_features, input_captions], outputs=outputs)

    model.layers[2].set_weights([embedding_matrix])
    model.layers[2].trainable = False

    model.compile(loss='categorical_crossentropy', optimizer="adam")
    return model


def preprocess_img(img):

    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # Normalisation
    img = preprocess_input(img)
    return img


def encode_image(img, model_new):
    feature_vector = model_new.predict(img)
    feature_vector = feature_vector.reshape((-1,))
    return feature_vector

def predict_caption(photo, word_to_idx, idx_to_word, model):
    
    max_len = 35
    in_text = "startseq"
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')

        ypred = model.predict([photo.reshape(1,2048), sequence])
        ypred = ypred.argmax()  # WOrd with max prob always - Greedy Sampling
        word = idx_to_word[ypred]
        in_text += (' ' + word)

        if word == "endseq":
            break

    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption



def main():
    st.title("Deep Learning Caption Bot")

    st.write("Please wait, loading essential data")

    img_encoding_model = create_img_encoding_model()
    caption_model = create_caption_model()
    caption_model = load_model('./saved/model_9.h5')

    with open("./saved/word_to_idx.pkl", "rb") as f:
        word_to_idx = pickle.load(f)

    with open("./saved/idx_to_word.pkl", "rb") as f:
        idx_to_word = pickle.load(f)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:

        st.write("Please wait, pre-processing image")

        imagee = Image.open(uploaded_file)
        print(type(imagee))
        img_vec = preprocess_img(imagee)
        img_vec = encode_image(img_vec, img_encoding_model)
        print(img_vec.shape)

        st.image(imagee, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Generating Caption.........")

        label = predict_caption(img_vec, word_to_idx, idx_to_word, caption_model)

        label = "Caption: " + label
        st.write('%s' % (label))

if __name__ == '__main__':
    main()
