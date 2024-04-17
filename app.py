from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import joblib


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
     df= pd.read_csv("preprocessed.csv")
     
     
     X= df['review']
     y = df['sentiment']
     cv = CountVectorizer()
     X = cv.fit_transform(X) # Fit the Data
     from sklearn.model_selection import train_test_split
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
     from sklearn.linear_model import LogisticRegression
     clf = LogisticRegression()
     clf.fit(X_train,y_train)
     clf.score(X_test,y_test)
     if request.method == 'POST':
        message = request.form['message']
        preprocessed_message = message  # Preprocess the input text
        data = [preprocessed_message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
