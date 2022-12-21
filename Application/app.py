from flask import Flask, render_template, request, redirect
import pickle
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
sw = stopwords.words("english")
with open('static/model/model_pkl', 'rb') as f:
    model = pickle.load(f)
vocab = pd.read_csv('static/model/vocabulary.txt', header=None)
tokens = vocab[0].tolist()


def preprocessing(text):
    data = pd.DataFrame([text])
    data[0] = data[0].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data[0] = data[0].apply(lambda x: " ".join(
        re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE) for x in x.split()))
    data[0] = data[0].str.replace('[^\w\s]', '')
    data[0] = data[0].str.replace('\d', '')
    data[0] = data[0].apply(lambda x: " ".join(
        x for x in x.split() if x not in sw))
    data[0] = data[0].apply(lambda x: " ".join(
        [lemmatizer.lemmatize(word) for word in x.split()]))
    data = data[0]
    pred = get_pred(data)
    return pred


def get_pred(data):
    vectorized_test_lst = []
    for sentence in data:
        sentence_lst = np.zeros(len(tokens))
        for i in range(len(tokens)):
            if tokens[i] in sentence.split():
                sentence_lst[i] = 1
        vectorized_test_lst.append(sentence_lst)
    tested = np.asarray(vectorized_test_lst, dtype=np.float32)
    return model.predict(tested)


app = Flask(__name__)

data = dict()
reviews = []
negative = 0
positive = 0


@app.route('/')
def index():
    data['reviews'] = reviews
    data['negative'] = negative
    data['positive'] = positive
    return render_template('index.html', data=data)


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    pred = preprocessing(text)
    if pred == 1:
        global negative
        negative += 1
    else:
        global positive
        positive += 1
    reviews.insert(0, text)
    return redirect(request.url)


if __name__ == "__main__":
    app.run(debug=True)
