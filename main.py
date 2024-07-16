import streamlit as st
import cv2
from PIL import Image
from detect import detect_objects
from tracker import *
import cvzone
from gpt3_chatbot import GPT3Assistant

# Define color constants
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (192, 192, 192)
YELLOW = (0, 255, 255)

# Instantiate the GPT-3 assistant
assistant = GPT3Assistant()

# Enable wide mode for the entire app
st.set_page_config(layout="wide")

def draw_overlay(frame, x, y, x2, y2):
        """
        Draw a semi-transparent overlay on the bounding box.

        Args:
            frame (ndarray): The video frame.
            x (int): Top-left x coordinate of the bounding box.
            y (int): Top-left y coordinate of the bounding box.
            x2 (int): Bottom-right x coordinate of the bounding box.
            y2 (int): Bottom-right y coordinate of the bounding box.
        """
        overlay = frame.copy()
        alpha = 0.5
        cv2.rectangle(overlay, (x, y), (x2, y2), GREY, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_bounding_box(frame, x, y, x2, y2):
    """
    Draw the bounding box with corner rectangles.

    Args:
        frame (ndarray): The video frame.
        x (int): Top-left x coordinate of the bounding box.
        y (int): Top-left y coordinate of the bounding box.
        x2 (int): Bottom-right x coordinate of the bounding box.
        y2 (int): Bottom-right y coordinate of the bounding box.
    """
    cvzone.cornerRect(frame, (x, y, x2-x, y2-y), l=15, rt=2, colorR=BLUE)

def draw_id_circle(frame, cx, cy, id):
    """
    Draw the ID in a circle at the center of the bounding box.

    Args:
        frame (ndarray): The video frame.
        cx (int): Center x coordinate of the bounding box.
        cy (int): Center y coordinate of the bounding box.
        id (int): ID of the detected object.
    """
    cvzone.putTextRect(frame, f"{id}", (cx - 20, cy + 5), offset=10, scale=1, thickness=2, colorR=WHITE, colorT=BLACK)

def draw_confidence(frame, x, y, conf):
    """
    Display the confidence score near the bounding box.

    Args:
        frame (ndarray): The video frame.
        x (int): Top-left x coordinate of the bounding box.
        y (int): Top-left y coordinate of the bounding box.
        conf (float): Confidence score of the detection.
    """
    cvzone.putTextRect(frame, f"Goat {int(conf*100)}%", (x, y-10), offset=10, scale=1, thickness=2)

def draw_table(frame, trackers):
    """
    Draw a transparent table with the analysis of detected objects.

    Args:
        frame (ndarray): The video frame.
        trackers (list): List of tracked objects.
    """
    overlay = frame.copy()
    table_x, table_y = 10, 10
    table_width, table_height = 350, 50 + (len(trackers) * 30)
    col1_x, col2_x, col3_x = table_x + 10, table_x + 60, table_x + 170

    # Draw the transparent table background
    alpha = 0.7
    cv2.rectangle(overlay, (table_x, table_y), (table_x + table_width, table_y + table_height),  WHITE, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw table header
    cv2.putText(frame, "ID", (col1_x, table_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,  BLACK, 2)
    cv2.putText(frame, "Type", (col2_x, table_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,  BLACK, 2)
    cv2.putText(frame, "Height (px)", (col3_x, table_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,  BLACK, 2)
    
    # Draw table lines (horizontal)
    cv2.line(frame, (table_x, table_y + 40), (table_x + table_width, table_y + 40),  BLACK, 2)
    
    # Draw table lines (vertical)
    cv2.line(frame, (col2_x - 10, table_y), (col2_x - 10, table_y + table_height),  BLACK, 2)
    cv2.line(frame, (col3_x - 10, table_y), (col3_x - 10, table_y + table_height),  BLACK, 2)
    
    for i, result in enumerate(trackers):
        id = int(result[4])
        y_pos = table_y + 70 + i * 30
        x, y, w, h, id = result
        height = h
        cv2.putText(frame, str(id), (col1_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1,  BLACK, 2)
        cv2.putText(frame, "Goat", (col2_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1,  BLACK, 2)
        cv2.putText(frame, f"{height}", (col3_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, BLACK, 2)



# Function to perform YOLO object detection
def run_yolo_detection(video_url, frame_container):
    cap = cv2.VideoCapture(video_url)
    tracker = EuclideanDistTracker()
    total_count = []

    if not cap.isOpened():
        st.error("Error: Unable to open video stream.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform YOLO detection on 'frame'
        conf, detections, detections_list = detect_objects(frame)
        #trackers = update_tracker(detections, tracker)
        #print(f"Number of animals detected: {len(trackers)}")

        # Object Tracking
        boxes_ids = tracker.update(detections_list)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            x2, y2 = x + w, y + h
            cx, cy = (x + x2) // 2, (y + y2) // 2

            draw_overlay(frame, x, y, x2, y2)
            draw_bounding_box(frame, x, y, x2, y2)
            draw_id_circle(frame, cx, cy, id)
            draw_confidence(frame, x, y, conf)

            if id not in total_count:
                total_count.append(id)

        # Sort the trackers by ID
        trackers_sorted = sorted(boxes_ids, key=lambda x: x[4])
        #trackers_sorted = sorted(trackers, key=lambda x: x[4])

        draw_table(frame, trackers_sorted)
        cvzone.putTextRect(frame, f"Count: {len(total_count)}", (450, 50), offset=10, scale=3, thickness=2, colorR=WHITE, colorT=BLACK)
            

        # Convert the frame to an image
        frame_image = Image.fromarray(frame)

        frame_container.image(frame_image, channels="BGR")

    cap.release()

# CSS styles for video container
video_styles = """
<style>
.video-container {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
    height: auto;
}
</style>
"""

# Header
st.markdown("""
    <div style='background-color: #4CAF50; padding: 10px; text-align: center;'>
        <h2 style='color: white;'>Clinton's Pen Monitoring System</h2>
    </div>
    <br><br>
    """, unsafe_allow_html=True)

# Introduction with increased font size, line breaks, and spaces
st.markdown("""
    <div style='font-size: 18px; line-height: 1.6;'>
        <h6>Welcome to <b>Clinton's Pen Monitoring System</b>. This application is designed to help farmers easily monitor 
        the health and activity of their animals. By using this system, you can get real-time updates from your 
        livestock pen, including a live stream of the pen and analysis of the animals' behavior and health metrics.</h6>
    </div>
    """, unsafe_allow_html=True)

st.subheader("Features")
st.markdown("""

    <div style='font-size: 18px; line-height: 1.6;'>
        <b></b>
        <ul>
            <li><b>Pen Livestream:</b> View a live video feed from your livestock pen.</li>
            <li><b>Pen Analysis:</b> Get insights and analysis on the health and behavior of your animals using YOLO object detection.</li>
            <li><b>AI Bot:</b> Ask Clinton's AI Bot any questions about the animals.</li>
        </ul>
    </div>
    <div style='font-size: 16px;'>
        <h6>This tool aims to provide farmers with a comprehensive solution to ensure the wellbeing of their animals 
        and optimize farm operations.</h6>
        <h6>Click the checkboxes below to view the pen livestream and analysis.</h6>
    </div>
    <br>
    """, unsafe_allow_html=True)

# Add two columns, for the pen livestream and the analysis
col1, col2 = st.columns(2)

# Placeholder image URLs
livestream_image_url = "./images/goats.png"
analysis_image_url = "./images/analysis.png"

# Create checkboxes and placeholders for video and analysis
livestream_button = col1.checkbox("Pen Livestream")
livestream_placeholder = col1.empty()

analysis_button = col2.checkbox("Pen Analysis")
analysis_placeholder = col2.empty()

# Handle livestream content
if livestream_button:
    livestream_placeholder.markdown(video_styles, unsafe_allow_html=True)
    livestream_placeholder.video("./videos/pen_recording.mp4", autoplay=True, format='video/mp4')
else:
    col1.image(livestream_image_url, use_column_width=True)

# Handle analysis content
if analysis_button:
    analysis_placeholder.markdown(video_styles, unsafe_allow_html=True)
    video_url = "./videos/pen_recording.mp4"  # Replace with actual video URL
    run_yolo_detection(video_url, analysis_placeholder)
else:
    col2.image(analysis_image_url, use_column_width=True)


# Add ChatGPT chatbot to answer animal health questions
st.subheader("Clinton's Animals AI Bot")
st.markdown("""<h6>Ask Clinton's Farm Monitoring System AI Bot any questions about the animals. It will provide you with the general answers. For now it has no access to the pen, so you just need to ask general questions.</h6>""", unsafe_allow_html=True)
user_question = st.text_input("Ask anything about the animals")
if st.button("ASK", use_container_width=400):
    response = assistant.g_chat(user_question)
    st.write(response)

st.subheader("Clinton's Profile")
st.write("- I am a machine learning engineer with a passion for AI and ML")
st.write("- I have a background in computer science and statistics")
st.write("- I am a certified machine learning engineer with 2 years experience")
st.write("- LinkedIn: [Click link to LinkedIn profile](https://www.linkedin.com/in/clinton-nyaore-0a7590215/)")
st.write("- GitHub: [Click link to GitHub](https://github.com/Clinton-Nyaore)")
st.write("- Email: cnyaore@gmail.com")
st.write("- Phone: +254 745 504 421")
st.write("- WhatsApp: [Click link to Whatsapp](https://wa.me/254745504421?text=Hello%20am%20from%20cvzone%20bootcamp)")

# Footer
st.markdown("""
    <br><br><br>
    <div style='background-color: #4CAF50; padding: 10px; text-align: center;'>
        <p style='color: white;'>© 2024 Clinton's Farm Monitoring System. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)
