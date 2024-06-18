from flask import Flask, render_template, request, jsonify, Response
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import threading
import time
import base64
import io
from PIL import Image

app = Flask(__name__)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

data = []
trained_model = None
is_predicting = False
is_capturing = False
current_class = ""
num_samples = 0
samples_captured = 0
capturing_complete = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed', methods=['POST'])
def video_feed():
    global data, is_capturing, num_samples, samples_captured, current_class, is_predicting, trained_model, capturing_complete

    # Récupérer les données de l'image depuis la requête POST
    image_data = request.json['image']
    image_data = image_data.split(',')[1]  # Supprimer le préfixe de l'URL data:image/jpeg;base64,
    image_data = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_data))
    frame = np.array(image)

    # Traiter l'image avec Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        results = holistic.process(frame_rgb)

        if results.right_hand_landmarks or results.left_hand_landmarks:
            landmarks = []

            if results.right_hand_landmarks:
                for landmark in results.right_hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                )

            if results.left_hand_landmarks:
                for landmark in results.left_hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                )

            if is_capturing and samples_captured < num_samples:
                data.append([current_class] + landmarks)
                samples_captured += 1
                cv2.putText(frame, f'Capturing: {current_class} ({samples_captured}/{num_samples})', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                if samples_captured >= num_samples:
                    capturing_complete = True

            if is_predicting and trained_model:
                prediction = trained_model.predict([landmarks])[0]
                cv2.putText(frame, f'Prediction: {prediction}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return '', 204

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global is_capturing, current_class, num_samples, samples_captured, data, capturing_complete

    capture_info = request.get_json()
    num_samples = int(capture_info['num_samples'])
    class_names = capture_info['class_names']

    for class_name in class_names:
        current_class = class_name
        samples_captured = 0
        is_capturing = True
        while samples_captured < num_samples:
            time.sleep(0.1)
        is_capturing = False

    # Convertir les données en DataFrame et les sauvegarder dans un fichier CSV
    columns = ['label'] + [f'{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('hand_gestures.csv', index=False)

    return jsonify({'message': 'Capture completed.', 'success': True})

@app.route('/train_model', methods=['POST'])
def train_model():
    global trained_model

    df = pd.read_csv('hand_gestures.csv')
    X = df.drop('label', axis=1)
    y = df['label']

    model = DecisionTreeClassifier()
    model.fit(X, y)

    trained_model = model

    return jsonify({'message': 'Model trained successfully.', 'success': True})

@app.route('/start_prediction', methods=['POST'])
def start_prediction():
    global is_predicting
    is_predicting = True
    return jsonify({'message': 'Prediction started.', 'success': True})

@app.route('/stop_prediction', methods=['POST'])
def stop_prediction():
    global is_predicting
    is_predicting = False
    return jsonify({'message': 'Prediction stopped.', 'success': True})

if __name__ == '__main__':
    app.run(debug=True)
