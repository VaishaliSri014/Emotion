from flask import Flask, render_template, request
import numpy as np
import cv2
from keras.models import load_model
import webbrowser

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

info = {}

haarcascade = "haarcascade_frontalface_default.xml"
label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']
print("+" * 50, "loadin gmmodel")
model = load_model('model.h5')
cascade = cv2.CascadeClassifier(haarcascade)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/choose_singer', methods=["POST"])
def choose_singer():
    info['activity'] = request.form['activity']
    print(info)
    return render_template('choose_singer.html', data=info['activity'])


@app.route('/emotion_detect', methods=["POST"])
def emotion_detect():

    info['language'] = request.form['language']
    #info['Books'] = request.form['Books']
    #info['Music'] = request.form['Music']
    #info['Quotes'] = request.form['Quotes']
    print(info)
    found = False

    cap = cv2.VideoCapture(0)
    while not (found):
        _, frm = cap.read()
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

        faces = cascade.detectMultiScale(gray, 1.4, 1)

        for x, y, w, h in faces:
            found = True
            roi = gray[y:y + h, x:x + w]
            cv2.imwrite("static/face.jpg", roi)

    roi = cv2.resize(roi, (48, 48))

    roi = roi / 255.0

    roi = np.reshape(roi, (1, 48, 48, 1))

    prediction = model.predict(roi)

    print(prediction)

    prediction = np.argmax(prediction)
    prediction = label_map[prediction]

    cap.release()
    if (info['activity']=='Books'):
        #if (prediction == 'Fear'):
            link1 = f"https://www.goodreads.com/search?q={prediction}+books"

            webbrowser.open(link1)

            return render_template("emotion_detect.html", data=prediction, link=link1)

    elif (info['activity']=='Video'):
      link2 = f"https://www.youtube.com/results?search_query={info['activity']}+{prediction}+{info['language']}+song"
      # link2 = f"https://www.youtube.com/results?search_query={info['Video']}+{prediction}+song"

      webbrowser.open(link2)

      return render_template("emotion_detect.html", data=prediction, link=link2)

    elif (info['activity']=='Quotes'):
      link3 = f"https://www.brainyquote.com/search_results?x=0&y=0&q={prediction}+quotes"

      webbrowser.open(link3)

      return render_template("emotion_detect.html", data=prediction, link=link3)

    elif (info['activity']=='Music'):
      link4 = f"https://open.spotify.com/search/{prediction}%20{info['language']}%20songs"
      # link4 = f"https://open.spotify.com/search/{prediction}%20songs"
      webbrowser.open(link4)

      return render_template("emotion_detect.html", data=prediction, link=link4)

    elif (info['activity']=='Games'):
      link5 = f"https://poki.com/en/online"

      webbrowser.open(link5)

      return render_template("emotion_detect.html", data=prediction, link=link5)

if __name__ == "__main__":
    app.run(debug=True)
