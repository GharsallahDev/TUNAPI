import io
import os
import cv2
import torch
import shutil
import base64
from PIL import Image
from flask import request, jsonify, url_for, render_template
from werkzeug.utils import secure_filename
from app import app, socketio, celery

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/identification.pt', trust_repo=True)

@app.route('/test')
def test():
    return render_template('test_socket.html')

@app.route('/test_socket')
def test_socket():
    socketio.emit('test_message', {'message': 'Hello, world!'}, namespace='/')
    return jsonify(success=True, message="Message sent!")



@app.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'mp4'}

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify(success=False, message="No file part in the request")
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify(success=False, message="No selected file")
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if filename.lower().endswith(('.mp4', '.mov')):
            process_video(filepath)
            return jsonify(success=True, message="Video is being processed")
        else:
            clear_path = 'runs/detect/exp'
            if os.path.exists(clear_path):
                shutil.rmtree(clear_path)

            results = model(filepath)
            results.save()

            processed_dir = os.path.join(app.root_path, 'static', 'processed')
            if not os.path.exists(processed_dir):
                os.makedirs(processed_dir)

            jpg_filename = os.path.splitext(filename)[0] + '.jpg'
            rename_this_too = os.path.join('runs/detect/exp', jpg_filename)

            if os.path.exists(rename_this_too):
                processed_filepath = os.path.join(processed_dir, jpg_filename)
                if os.path.exists(processed_filepath):
                    os.remove(processed_filepath)
                shutil.move(rename_this_too, processed_filepath)
                shutil.rmtree('runs/detect/exp', ignore_errors=True)
                return jsonify(success=True, filename=filename, imagePath=url_for('static', filename=os.path.join('processed', jpg_filename)))

            print(f"Expected file not found: {rename_this_too}")
            return jsonify(success=False, message="Processed file could not be found")

    return jsonify(success=False, message="File not allowed")


def process_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames in the video
    if total_frames == 0:
        print("Error: Total frames returned as zero.")
        return
    
    frame_number = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            # Process each detection and draw boxes
            for det in results.xyxy[0]:  # detections for each image
                if det[4] >= 0.5:  # det[4] is the confidence score
                    x1, y1, x2, y2, conf, cls = int(det[0]), int(det[1]), int(det[2]), int(det[3]), det[4], int(det[5])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw rectangle on the frame
                    cv2.putText(frame, f'{model.names[cls]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
            img = Image.fromarray(processed_frame)
            file_object = io.BytesIO()
            img.save(file_object, 'JPEG')
            base64_image = base64.b64encode(file_object.getvalue()).decode('utf-8')

            status = 'Processing'
            if frame_number == total_frames - 1:
                status = 'Completed'

            socketio.emit('frame', {
                'frame_data': base64_image,
                'frame_number': frame_number,
                'total_frames': total_frames,
                'status': status
            }, namespace='/')

            frame_number += 1
    finally:
        cap.release()
        # Ensure to emit the completion status if not yet done
        if frame_number != total_frames or status != 'Completed':
            socketio.emit('frame', {
                'frame_data': '',  # No frame data needed for final message
                'frame_number': frame_number,
                'total_frames': total_frames,
                'status': 'Completed'
            }, namespace='/')