### README for Speech-to-Text, Text-to-Text, and Text-to-Speech Processing with Flask

#### Overview

This Python script is a Flask web application that allows users to upload a `.wav` audio file. The application performs the following tasks:

1. **Speech-to-Text (STT)**: Converts speech from the audio file to text using a pre-trained model.
2. **Text-to-Text Translation (TTT)**: Translates the text obtained from the STT process to another language.
3. **Text-to-Speech (TTS)**: Converts the translated text back to speech and saves it as an audio file.

The progress of these tasks is displayed in real-time using Socket.IO.

#### Prerequisites

Ensure you have Python 3.7 or later installed. You also need to install `pip` if it is not already installed.

#### Installation

1. **Clone the repository**:
   ```sh
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Create and activate a virtual environment**:
   ```sh
   python3 -m venv myenv
   source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
   ```

3. **Install required libraries**:
   ```sh
   pip install flask flask_socketio torch transformers soundfile librosa werkzeug
   pip install safetensors xformers
   ```

#### Running the Application

1. **Start the Flask application**:
   ```sh
   python voiceToVoiceWeb.py
   ```

2. **Access the application**:
   Open your web browser and navigate to `http://127.0.0.1:5000`.

#### Application Workflow

1. **Upload an Audio File**:
   - Navigate to the homepage.
   - Upload a `.wav` audio file.

2. **Processing**:
   - The server processes the uploaded file by performing STT, TTT, and TTS.
   - Progress updates are displayed in real-time on the web page.

3. **Results**:
   - Once processing is complete, the results page displays the STT result and TTT translation.
   - A link is provided to download the synthesized audio file.

#### Summary of the Code

- **Configuration**:
  - The script sets up Flask, Socket.IO, and the file upload folder.
  - Ensures necessary folders exist and defines allowed file types.

- **Routes**:
  - `upload_file`: Handles the file upload.
  - `progress`: Displays the processing progress.
  - `results`: Displays the STT and TTT results and provides a link to download the TTS audio file.
  - `download_file`: Allows downloading the processed audio file.

- **Socket.IO Events**:
  - `handle_connect`: Handles client connection.
  - `process_file`: Background task to process the uploaded file:
    - Loads models for STT, TTT, and TTS.
    - Performs STT, translates the text, and performs TTS.
    - Emits progress updates to the client.



### Dependencies and Their Uses

- **Flask**: Web framework to handle requests and serve web pages.
- **Flask-SocketIO**: Real-time communication between the server and clients.
- **Torch**: PyTorch library for loading and running machine learning models.
- **Transformers**: Hugging Face library for working with pre-trained models.
- **SoundFile**: Library for reading and writing sound files.
- **Librosa**: Library for audio processing.
- **Werkzeug**: Utility library for handling secure file uploads.
- **Safetensors**: Library for safe handling of tensors.

