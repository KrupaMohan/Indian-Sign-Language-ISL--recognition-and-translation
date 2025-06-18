import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from flask import Flask, Response, render_template, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
import string
import threading
import time
import uuid
import logging
import json
from collections import deque
from transformers import pipeline

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Sentiment analysis pipeline
sentiment_analyzer = None
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt"
    )
except Exception as e:
    logger.error(f"Failed to load sentiment analysis model: {e}")
    sentiment_analyzer = None

# ISL detection configuration
try:
    model = load_model('model.h5')
    print('Model loaded successfully')
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise
ISL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
cap = None
current_prediction = "Prediction: None"
prediction_history = deque(maxlen=5)
word = []
last_prediction = None
last_time = time.time()
lock = threading.Lock()
COMMON_WORDS = ['HELLO', 'THANK', 'YOU', 'GOOD', 'BYE']
predicted_sentence = ''
NAVIGATION_GESTURES = {'N': 'next', 'B': 'back', 'S': 'submit'}

# Tutorial configuration
TUTORIAL_STEPS = [
    {'letter': 'A', 'prompt': 'Show letter A'},
    {'letter': 'B', 'prompt': 'Show letter B'},
    {'letter': 'C', 'prompt': 'Show letter C'},
    {'letter': 'D', 'prompt': 'Show letter D'},
    {'letter': 'E', 'prompt': 'Show letter E'}
]

# Feedback rules for ISL letters
FEEDBACK_RULES = {
    'A': {
        'correct': 'Correct! You signed "A" perfectly.',
        'incorrect': 'For "A", close your fingers into a fist with your thumb on the side.'
    },
    'B': {
        'correct': 'Great job! Your "B" is spot-on.',
        'incorrect': 'For "B", keep fingers together, palm out, and thumb extended.'
    },
    'C': {
        'correct': 'Nice! Your "C" is clear.',
        'incorrect': 'For "C", curve your fingers to form a "C" shape, palm facing out.'
    },
    'D': {
        'correct': 'Well done! Your "D" is correct.',
        'incorrect': 'For "D", extend index finger up, thumb holding other fingers.'
    },
    'E': {
        'correct': 'Excellent! You nailed "E".',
        'incorrect': 'For "E", tuck fingers down with thumb over them, palm out.'
    }
}

# Track tutorial progress
tutorial_progress = {'completed_steps': 0, 'correct_attempts': 0, 'total_attempts': 0}

# ISL configuration
data_path = "data"
TARGET_SIZE = 100
MAX_CHARS_PER_LINE = 10

def clean_input(text):
    allowed_chars = set(string.ascii_letters + string.digits + " ")
    cleaned = ''.join(ch for ch in text if ch in allowed_chars)
    return cleaned.upper()

def text_to_isl_images(text):
    text = clean_input(text)
    images = []
    for char in text:
        if char == " ":
            images.append(np.ones((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8) * 255)
            continue
        if char not in ISL_LABELS:
            continue
        folder = os.path.join(data_path, char)
        if not os.path.exists(folder):
            logger.warning(f"Folder not found: {folder}")
            continue
        first_image_file = next(
            (f for f in os.listdir(folder) if not f.startswith('.') and os.path.isfile(os.path.join(folder, f))),
            None
        )
        if not first_image_file:
            logger.warning(f"No valid image in folder: {folder}")
            continue
        img_path = os.path.join(folder, first_image_file)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.warning(f"Failed to load image: {img_path}")
            continue
        img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        images.append(img)
    return images

def chunk_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def pad_image_width(img, target_width):
    h, w = img.shape[:2]
    if w == target_width:
        return img
    pad_width = target_width - w
    padding = np.ones((h, pad_width, 3), dtype=np.uint8) * 255
    padded_img = np.concatenate((img, padding), axis=1)
    return padded_img

def generate_isl_image(text):
    isl_images = text_to_isl_images(text)
    if not isl_images:
        logger.warning(f"No ISL images generated for text: {text}")
        return None
    lines = chunk_list(isl_images, MAX_CHARS_PER_LINE)
    h_concat_lines = []
    max_width = 0
    for line_imgs in lines:
        h_concat = cv2.hconcat(line_imgs)
        h_concat_lines.append(h_concat)
        if h_concat.shape[1] > max_width:
            max_width = h_concat.shape[1]
    for i in range(len(h_concat_lines)):
        h_concat_lines[i] = pad_image_width(h_concat_lines[i], max_width)
    combined_image = cv2.vconcat(h_concat_lines)
    output_dir = os.path.join('static', 'isl_output')
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"isl_{uuid.uuid4().hex}.png"
    output_path = os.path.join(output_dir, output_file)
    cv2.imwrite(output_path, combined_image)
    return output_file

def detect_hand(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    skin_pixels = cv2.countNonZero(mask)
    total_pixels = roi.shape[0] * roi.shape[1]
    skin_ratio = skin_pixels / total_pixels
    if skin_ratio > 0.15:
        for contour in contours:
            if cv2.contourArea(contour) > 5000:
                return True
    return False

def generate_frames():
    global cap, current_prediction, prediction_history, word, last_prediction, last_time, predicted_sentence
    if cap is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Failed to open webcam")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        with lock:
            if cap is None or not cap.isOpened():
                break
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame")
                break
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        roi_width, roi_height = 300, 300
        # Center the ROI box
        roi_x = (width - roi_width) // 2
        roi_y = (height - roi_height) // 2
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), 
                     (0, 255, 0), 2)
        roi = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
        action = ''
        if detect_hand(roi):
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (50, 50))
            gray = gray.reshape(1, 50, 50, 1)
            gray = gray.astype('float32') / 255.0
            prediction = model.predict(gray, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
            with lock:
                if confidence > 0.7:
                    current_pred = ISL_LABELS[predicted_class]
                    current_prediction = f"Prediction: {current_pred} ({confidence*100:.1f}%)"
                    prediction_history.append(f"{current_pred} ({confidence*100:.1f}%)")
                    hold_duration = 2.0  # seconds
                    if (len(word) == 0 or current_pred != word[-1]) or (time.time() - last_time > hold_duration):
                        word.append(current_pred)
                        last_time = time.time()
                    last_prediction = current_pred
                    if ''.join(word) in COMMON_WORDS:
                        predicted_sentence = ''.join(word)
                        word.clear()
                        logger.info(f"Sentence predicted: {predicted_sentence}")
                    if current_pred in NAVIGATION_GESTURES:
                        action = NAVIGATION_GESTURES[current_pred]
                else:
                    current_prediction = "Prediction: Low Confidence"
                    last_prediction = None
                    logger.info(f"Hand detected - Low confidence: {confidence:.3f}")
        else:
            with lock:
                current_prediction = "Prediction: No Hand Detected"
                last_prediction = None
                logger.info("No hand detected")
        cv2.putText(frame, "Show ISL sign in the green box", (50, height - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame_data = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.033)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/isl_detection')
def isl_detection():
    return render_template('asl_detection.html')

@app.route('/speech_to_isl')
def speech_to_isl():
    return render_template('speech_to_isl.html')

@app.route('/text_to_isl')
def text_to_isl():
    return render_template('text_to_isl.html')

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    with lock:
        current_pred = current_prediction.split(': ')[1][0] if 'Prediction: ' in current_prediction and current_prediction.split(': ')[1][0] in ISL_LABELS else ''
        return jsonify({
            'current': current_prediction,
            'history': list(prediction_history),
            'confidence': float(current_prediction.split('(')[1].split('%')[0]) if 'Prediction: No Hand' not in current_prediction and 'Low Confidence' not in current_prediction else 0,
            'word': ''.join(word),
            'action': NAVIGATION_GESTURES.get(current_pred, '')
        })

@app.route('/clear_word', methods=['POST'])
def clear_word():
    global word, predicted_sentence
    with lock:
        word = []
        predicted_sentence = ''
        logger.info("Word and sentence cleared")
    return jsonify({'status': 'success'})

@app.route('/stop_camera')
def stop_camera():
    global cap
    with lock:
        if cap is not None and cap.isOpened():
            cap.release()
            cap = None
            logger.info("Camera stopped")
    return "Camera stopped"

@app.route('/process_speech', methods=['POST'])
def process_speech():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    cleaned_text = clean_input(text)
    if not any(ch.isalnum() for ch in cleaned_text):
        return jsonify({'error': 'No valid letters found'}), 400
    output_file = generate_isl_image(cleaned_text)
    if output_file:
        return jsonify({'text': cleaned_text, 'image': f'/static/isl_output/{output_file}'})
    return jsonify({'error': 'No valid ISL images generated'}), 400

@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    cleaned_text = clean_input(text)
    if not any(ch.isalnum() for ch in cleaned_text):
        return jsonify({'error': 'No valid letters found'}), 400
    output_file = generate_isl_image(cleaned_text)
    if output_file:
        return jsonify({'text': cleaned_text, 'image': f'/static/isl_output/{output_file}'})
    return jsonify({'error': 'No valid ISL images generated'}), 400

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = request.json.get('feedback')
    if not feedback:
        return jsonify({'error': 'No feedback provided'}), 400
    sentiment_label = 'Unknown'
    sentiment_score = 0.0
    if sentiment_analyzer:
        try:
            sentiment_result = sentiment_analyzer(feedback)[0]
            sentiment_label = sentiment_result['label'].capitalize()
            sentiment_score = sentiment_result['score']
            if sentiment_label == 'Positive' and sentiment_score < 0.7:
                sentiment_label = 'Neutral'
            elif sentiment_label == 'Negative' and sentiment_score < 0.7:
                sentiment_label = 'Neutral'
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            sentiment_label = 'Unknown'
            sentiment_score = 0.0
    feedback_entry = {
        'id': str(uuid.uuid4()),
        'feedback': feedback,
        'sentiment': sentiment_label,
        'sentiment_score': sentiment_score
    }
    with open('feedback.json', 'a') as f:
        json.dump(feedback_entry, f)
        f.write('\n')
    return jsonify({'status': 'success'})

@app.route('/get_feedback')
def get_feedback():
    feedback_list = []
    if os.path.exists('feedback.json'):
        with open('feedback.json', 'r') as f:
            for line in f:
                feedback_list.append(json.loads(line))
    return jsonify(feedback_list)

@app.route('/get_tutorial_step/<int:step>')
def get_tutorial_step(step):
    global tutorial_progress
    if step >= len(TUTORIAL_STEPS):
        progress = {
            'completed': tutorial_progress['completed_steps'],
            'total': len(TUTORIAL_STEPS),
            'accuracy': (tutorial_progress['correct_attempts'] / tutorial_progress['total_attempts'] * 100) if tutorial_progress['total_attempts'] > 0 else 0
        }
        logger.info(f"Tutorial complete: {progress}")
        return jsonify({'done': True, 'progress': progress})
    # Check current prediction
    with lock:
        current_pred = current_prediction.split(': ')[1][0] if 'Prediction: ' in current_prediction and current_prediction.split(': ')[1][0] in ISL_LABELS else ''
        confidence = float(current_prediction.split('(')[1].split('%')[0]) if 'Prediction: No Hand' not in current_prediction and 'Low Confidence' not in current_prediction else 0
    tutorial_progress['total_attempts'] += 1
    feedback = ''
    if current_pred == TUTORIAL_STEPS[step]['letter'] and confidence > 65:  # Lowered threshold
        feedback = FEEDBACK_RULES[TUTORIAL_STEPS[step]['letter']]['correct']
        tutorial_progress['completed_steps'] = max(tutorial_progress['completed_steps'], step + 1)
        tutorial_progress['correct_attempts'] += 1
    elif current_pred and confidence > 50:
        feedback = FEEDBACK_RULES[TUTORIAL_STEPS[step]['letter']]['incorrect']
    elif confidence > 0:
        feedback = 'Low confidence. Try positioning your hand clearly in the green box.'
    else:
        feedback = 'No hand detected. Show your hand in the green box.'
    progress = {
        'completed': tutorial_progress['completed_steps'],
        'total': len(TUTORIAL_STEPS),
        'accuracy': (tutorial_progress['correct_attempts'] / tutorial_progress['total_attempts'] * 100) if tutorial_progress['total_attempts'] > 0 else 0
    }
    logger.info(f"Step {step}: Prediction={current_pred}, Confidence={confidence}, Feedback={feedback}, Progress={progress}")
    return jsonify({
        'letter': TUTORIAL_STEPS[step]['letter'],
        'prompt': TUTORIAL_STEPS[step]['prompt'],
        'feedback': feedback,
        'progress': progress
    })

@app.route('/reset_tutorial', methods=['POST'])
def reset_tutorial():
    global tutorial_progress
    tutorial_progress = {'completed_steps': 0, 'correct_attempts': 0, 'total_attempts': 0}
    logger.info("Tutorial progress reset")
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=False, threaded=True, host='0.0.0.0', port=8080)