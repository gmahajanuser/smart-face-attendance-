from flask import Flask, send_from_directory
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/start-face-detection')
def start_face_detection():
    try:
        # Execute the add_faces.py script
        subprocess.Popen(['python', 'add_faces.py'])
        return "Face detection process initiated!"

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
