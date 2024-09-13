# Football players tracking with deep learning techniques

[![Football Tracking Demo](https://img.youtube.com/vi/pV0spkNN5EE/0.jpg)](https://youtu.be/pV0spkNN5EE)


Web application to automate football players tracking.  
The goal of this project is to create a system dedicated to acquiring football players' tracking data and visualizing it automatically. The user is able to upload a video with a fragment of a football match and, in return, receive analytical data based on that footage. These data include tracking data and the percentage of teams' ball possession throughout the video clip.

## Object detection models weights 

Models weights for keypoints and players detections 
need to be downloaded and put into the *models* directory (adjust paths in config files).
Weights are avaliable for download [here](https://drive.google.com/drive/folders/1Gd394woIxfGoTMwppeBoTMg1DnQxrxfj?usp=sharing).


## Installation & How to use?

### Prerequisites

1. **Python 3.11** or later: Ensure Python is installed. You can download it from [here](https://www.python.org/downloads/).
2. **Virtual Environment**: It is recommended to create a virtual environment to manage dependencies.
   ```bash
   python -m venv venv
   source venv/bin/activate   # For MacOS/Linux
   venv\Scripts\activate      # For Windows
   ```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/wwoszczek/football-players-tracking
   cd football-players-tracking
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the credentials in `Creds.yaml` file (if needed).

4. Configure the paths in `tactical_config.yaml` and `tv_config.yaml`.

### How to Use?

1. **Run the main application** without the user interface: 
   ```bash
   python main.py
   ```

2. **Run the web application** to try the system with user interface:
   ```bash
   streamlit run web_app.py
   ```

   This will launch the web app where you can upload your video clip.

3. Once uploaded, the system will process the video and display:
   - Tracking data for players, referees, and ball movements.
   - A tactical map with player positions.
   - Percentage ball possession for both teams.
