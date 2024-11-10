from flask import Flask, render_template, request, Response, redirect, url_for, jsonify, send_file, flash, session
import os
import json
import shutil
import zipfile
import cv2
import numpy as np
import pandas as pd
import face_recognition
from datetime import datetime


# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# File paths and folders
UPLOAD_FOLDER = "uploads/"
ENCODINGS_FOLDER = "encodings/"
STATUS_FILE = "upload_status.json"
ATTENDANCE_RECORDS_FOLDER = "attendance_records/"

# Ensure directories and status file are set up
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ENCODINGS_FOLDER, exist_ok=True)
os.makedirs(ATTENDANCE_RECORDS_FOLDER, exist_ok=True)
if not os.path.exists(STATUS_FILE):
    with open(STATUS_FILE, 'w') as f:
        json.dump({"1st Year": False, "2nd Year": False, "3rd Year": False, "4th Year": False}, f)

# Get today's date to create attendance file
today = datetime.now().strftime('%d_%m_%Y')
attendance_path = f'attendance_{today}.csv'

# Helper functions for encoding and attendance

def load_encodings(encodings_path):
    """Load encodings from a JSON file."""
    if os.path.isfile(encodings_path):
        with open(encodings_path, 'r') as f:
            encodings = json.load(f)
            return {key: np.array(val) for key, val in encodings.items()}
    return {}

def save_encodings(encodings, encodings_path):
    """Save encodings to a JSON file."""
    with open(encodings_path, 'w') as f:
        json.dump({key: val.tolist() for key, val in encodings.items()}, f)

def generate_encodings(images_folder, encodings_path):
    """Generate and save encodings for student images in the specified folder."""
    encodings = {}
    for image_name in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_name)

        # Read the image using OpenCV
        image = cv2.imread(image_path)
        if image is not None:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Get face encodings
            encode = face_recognition.face_encodings(img_rgb)
            if encode:
                register_number = os.path.splitext(image_name)[0]  # Extract register number from file name
                encodings[register_number] = encode[0].tolist()  # Convert numpy array to list for JSON serialization

    # Save the encodings as a JSON file
    try:
        with open(encodings_path, 'w') as json_file:
            json.dump(encodings, json_file, indent=4)  # Use indent for pretty printing
    except Exception as e:
        print(f"Error saving encodings to {encodings_path}: {e}")


def markAttendance(register_number, student_name, year):
    """Mark attendance in the CSV file."""
    today = datetime.now().strftime('%d_%m_%Y')  # Get today's date
    attendance_path = os.path.join(ATTENDANCE_RECORDS_FOLDER, f'attendance_{year}_{today}.csv')

    # Check if the file exists; if not, create it with the header row
    if not os.path.isfile(attendance_path):
        with open(attendance_path, 'w') as f:
            f.write('Register Number,Student Name,Time Stamp,STATUS\n')

    # Check if the student has already been marked as present
    with open(attendance_path, 'r') as f:
        if any(register_number in line for line in f.readlines()):
            return  # Already marked

    timestamp = datetime.now().strftime('%H:%M:%S')
    current_time = datetime.now().strftime('%H:%M')
    status = "Absent"  # Default to absent after 9:00 AM
    if "08:00" <= current_time <= "08:30":
        status = "Present"
    elif "08:31" <= current_time <= "09:00":
        status = "Late"

    # Append the attendance record
    with open(attendance_path, 'a') as f:
        f.write(f'{register_number},{student_name},{timestamp},{status}\n')


def markAbsentStudents(register_numbers, student_names, year):
    """Automatically mark unmarked students as absent after 9:00 AM."""
    today = datetime.now().strftime('%d_%m_%Y')  # Get today's date
    attendance_path = os.path.join(ATTENDANCE_RECORDS_FOLDER, f'attendance_{year}_{today}.csv')

    # Check if the attendance file exists, otherwise create it with headers
    if not os.path.exists(attendance_path):
        with open(attendance_path, 'w') as f:
            f.write('Register Number,Student Name,Time Stamp,STATUS\n')

    # Read marked students to avoid duplicate entries
    with open(attendance_path, 'r') as f:
        marked_students = [line.split(',')[0] for line in f.readlines()[1:]]

    # Mark unmarked students as absent
    with open(attendance_path, 'a') as f:
        for register_number, student_name in zip(register_numbers, student_names):
            if register_number not in marked_students:
                f.write(f'{register_number},{student_name},{datetime.now().strftime("%H:%M:%S")},Absent\n')


# Routes for core attendance functionality

@app.route('/')
def index():
    return render_template('login_page.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    # Check admin credentials
    if username == 'it_admin' and password == 'itadmin':
        return redirect(url_for('admin_panel'))
    # Check year user credentials
    elif username == 'it_4thyear' and password == 'it4thyear':
        session['year'] = '4th Year'
        return redirect(url_for('welcome_page'))
    elif username == 'it_3rdyear' and password == 'it3rdyear':
        session['year'] = '3rd Year'
        return redirect(url_for('welcome_page'))
    elif username == 'it_2ndyear' and password == 'it2ndyear':
        session['year'] = '2nd Year'
        return redirect(url_for('welcome_page'))
    elif username == 'it_1styear' and password == 'it1styear':
        session['year'] = '1st Year'
        return redirect(url_for('welcome_page'))
    else:
        flash("Invalid username or password.")
        return redirect(url_for('index'))

@app.route('/welcome')
def welcome_page():
    return render_template('welcome.html')

@app.route('/take_attendance')
def take_attendance():
    return render_template('attendance.html')



def generate_frames(year):
    """Stream video frames with face detection and attendance marking."""
    cap = cv2.VideoCapture(0)
    absent_marked = False

    encodings_path = os.path.join(ENCODINGS_FOLDER, f"encoding_{year}.json")
    encodings = load_encodings(encodings_path)

    # Define paths for the student data file
    student_data_path = os.path.join(UPLOAD_FOLDER, year, f"{year}_dataset.csv")

    # Check if the student data CSV exists
    if not os.path.exists(student_data_path):
        print(f"Error: Student data file {student_data_path} does not exist.")
        return  # Exit the generator if the file is not found

    # Load student data outside of the loop for performance reasons
    student_data = pd.read_csv(student_data_path)

    # Clean column names to remove extra whitespaces
    student_data.columns = student_data.columns.str.strip()
    student_data['REGISTER NUMBER'] = student_data['REGISTER NUMBER'].astype(str).str.strip()

    while True:
        success, img = cap.read()
        if not success:
            break
        imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

        facesInCurrentFrame = face_recognition.face_locations(imgSmall)
        encodingsOfCurrentFrame = face_recognition.face_encodings(imgSmall, facesInCurrentFrame)

        for encodeFace, faceLoc in zip(encodingsOfCurrentFrame, facesInCurrentFrame):
            matches = face_recognition.compare_faces(list(encodings.values()), encodeFace)
            faceDistance = face_recognition.face_distance(list(encodings.values()), encodeFace)
            if any(matches):
                matchIndex = np.argmin(faceDistance)
                if matches[matchIndex] and faceDistance[matchIndex] < 0.5:
                    register_number = list(encodings.keys())[matchIndex]

                    if register_number in student_data['REGISTER NUMBER'].values:
                        student_name = student_data.loc[
                            student_data['REGISTER NUMBER'] == register_number, 'STUDENT NAME'
                        ].values[0]
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, student_name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                        markAttendance(register_number, student_name, year)
                    else:
                        print(f"No matching student found for register number: {register_number}")
                    break

        current_time = datetime.now().strftime('%H:%M')
        if current_time >= "09:00" and not absent_marked:
            # Pass `year` to markAbsentStudents instead of `attendance_path`
            markAbsentStudents(
                student_data['REGISTER NUMBER'].astype(str).tolist(),
                student_data['STUDENT NAME'].tolist(),
                year
            )
            absent_marked = True

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




@app.route('/video_feed')
def video_feed():
    year = session.get('year', None)
    # Ensure that year is valid
    if year is None:
        return "Year not found in session", 400  # or handle it as needed

    return Response(generate_frames(year), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/attendance_data')
def attendance_data():
    year = session.get('year')  # Use session data to determine year
    today = datetime.now().strftime('%d_%m_%Y')
    attendance_path = os.path.join("attendance_records", f"attendance_{year}_{today}.csv")

    if os.path.exists(attendance_path):
        # Read attendance file and convert to JSON
        attendance_df = pd.read_csv(attendance_path, header=None)  # No header in the CSV
        # Skip the first row (header) by using iloc[1:] and assign new column names
        attendance_df.columns = ['Register Number', 'Student Name', 'Time', 'Status']
        attendance_json = attendance_df.iloc[1:].to_dict(orient='records')  # Skip the header row
    else:
        attendance_json = []  # Empty list if no file found

    return jsonify(attendance_json)




@app.route('/download_attendance')
def download_attendance():
    year = session.get('year')
    today = datetime.now().strftime('%d_%m_%Y')  # Get today's date
    attendance_file = os.path.join(ATTENDANCE_RECORDS_FOLDER, f'attendance_{year}_{today}.csv')

    if os.path.isfile(attendance_file):
        return send_file(attendance_file, as_attachment=True)
    else:
        return "No attendance record found for today.", 404



@app.route('/logout', methods=['POST'])
def logout():
    session.pop('year', None)  # Remove the year from session
    return redirect(url_for('index'))  # Redirect to the login page

# Admin panel routes

@app.route('/admin')
def admin_panel():
    with open(STATUS_FILE) as f:
        upload_status = json.load(f)
    return render_template('admin.html', upload_status=upload_status)

@app.route('/upload-year-dataset', methods=['POST'])
def upload_year_dataset():
    year = request.form['year']
    year_folder = os.path.join(UPLOAD_FOLDER, year)
    images_folder = os.path.join(year_folder, "images")
    os.makedirs(images_folder, exist_ok=True)

    # Save files in year-specific folders
    image_zip = request.files['image_zip']
    csv_file = request.files['csv_file']
    image_zip_path = os.path.join(year_folder, f"{year}_images.zip")
    csv_file.save(os.path.join(year_folder, f"{year}_dataset.csv"))
    image_zip.save(image_zip_path)

    # Unzip images directly into the "images" folder
    with zipfile.ZipFile(image_zip_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            filename = os.path.basename(member)
            if filename:  # Only extract actual files, not directories
                source = zip_ref.open(member)
                target = open(os.path.join(images_folder, filename), "wb")
                with source, target:
                    shutil.copyfileobj(source, target)

    # Delete the zip file after extraction
    os.remove(image_zip_path)

    # Generate encodings and save them for the year
    encodings_path = os.path.join(ENCODINGS_FOLDER, f"encoding_{year}.json")
    generate_encodings(images_folder, encodings_path)  # Pass the encodings path

    # Update upload status
    with open(STATUS_FILE) as f:
        upload_status = json.load(f)
    upload_status[year] = True
    with open(STATUS_FILE, 'w') as f:
        json.dump(upload_status, f)

    flash(f"{year} files uploaded and encodings generated successfully.")
    return redirect(url_for('admin_panel'))

@app.route('/delete-year-dataset/<year>', methods=['POST'])
def delete_year_dataset(year):
    year_folder = os.path.join(UPLOAD_FOLDER, year)
    encoding_file = os.path.join(ENCODINGS_FOLDER, f"encoding_{year}.json")
    if os.path.exists(year_folder):
        shutil.rmtree(year_folder)
    if os.path.exists(encoding_file):
        os.remove(encoding_file)

    with open(STATUS_FILE) as f:
        upload_status = json.load(f)
    upload_status[year] = False
    with open(STATUS_FILE, 'w') as f:
        json.dump(upload_status, f)

    flash(f"{year} dataset deleted successfully.")
    return redirect(url_for('admin_panel'))

# Start the application
if __name__ == '__main__':
    app.run(debug=True)
