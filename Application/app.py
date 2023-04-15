from flask import Flask, render_template, request, redirect
from pipeline import preprocessing, get_pred


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
    preprocessed_text = preprocessing(text)
    pred = get_pred(preprocessed_text)
    if pred == 1:
        global negative
        negative += 1
    else:
        global positive
        positive += 1
    reviews.insert(0, text)
    return redirect(request.url)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
