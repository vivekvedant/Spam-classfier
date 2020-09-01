
from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# cv_vector = pickle.load('feature.pkl')

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/',methods=['POST'])
def predict():
    msg = [request.form['message']]
    
    cv_vector = CountVectorizer(vocabulary=pickle.load(open("vocabulary.pkl", "rb")))
    final_msg = cv_vector.transform(msg)
    # user_data = pd.DataFrame({'age':age,'bmi':bmi,'children':children,'sex':sex,'smoker':smoker,'region':region}, index=[0])
    prediction = model.predict(final_msg)
    if prediction  == 1:
        classified = 'spam'
    else:
        classified = 'not spam'
    return render_template('index.html',prediction_text="This is a  {}".format(classified))

if __name__ == "__main__":
    app.run(debug=True)