from flask import Flask, render_template, request
import os
from yolov4 import detect_cars
from algo import optimize_traffic

app = Flask(__name__)
print("✅ Flask app started — routes are loading...")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    print("✅ Index route loaded!")
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('videos')
    if len(files) != 4:
        return "Please upload exactly 4 videos", 400

    video_paths = []
    for i, file in enumerate(files):
        path = os.path.join(UPLOAD_FOLDER, f'video_{i}.mp4')
        file.save(path)
        video_paths.append(path)

    num_cars_list = [detect_cars(v) for v in video_paths]
    result = optimize_traffic(num_cars_list)
    return render_template('results.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8080)
