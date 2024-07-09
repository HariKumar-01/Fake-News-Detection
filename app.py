from flask import Flask, request, render_template
import joblib
import pandas as pd
import re
import string

app = Flask(__name__)

# Load the models and vectorizer
LR_model = joblib.load('model/LR_model.pkl')
DT_model = joblib.load('model/DT_model.pkl')
GB_model = joblib.load('model/GB_model.pkl')
RF_model = joblib.load('model/RF_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorizer.transform(new_x_test)
    
    pred_LR = LR_model.predict(new_xv_test)
    pred_DT = DT_model.predict(new_xv_test)
    pred_GB = GB_model.predict(new_xv_test)
    pred_RF = RF_model.predict(new_xv_test)
    
    return render_template('prediction.html', 
                           prediction_text_LR=f'LR Prediction: {output_label(pred_LR[0])}',
                           prediction_text_DT=f'DT Prediction: {output_label(pred_DT[0])}',
                           prediction_text_GB=f'GB Prediction: {output_label(pred_GB[0])}',
                           prediction_text_RF=f'RF Prediction: {output_label(pred_RF[0])}')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug=True)
