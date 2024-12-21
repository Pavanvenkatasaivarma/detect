from django.http import StreamingHttpResponse
from django.shortcuts import render
import cv2
from ultralytics import YOLO
import random
from django.http import JsonResponse
from django.views.decorators import gzip
from django.http import HttpResponseRedirect

# Global variable to control camera state
camera_running = False
videoCap = None
camera_url = "http://192.168.1.20:4747/video"

# Start camera stream
def start_camera():
    global videoCap, camera_running
    if not camera_running:
        videoCap = cv2.VideoCapture(camera_url)  # Open the webcam
        camera_running = True

# Stop camera stream
def stop_camera():
    global videoCap, camera_running
    if camera_running:
        videoCap.release()  # Release the camera
        camera_running = False

# Initialize YOLO model
yolo = YOLO('yolov8s.pt')

# Known parameters
KNOWN_WIDTH = 0.15  # Known width of the object in meters
KNOWN_DISTANCE = 1.00  # Known distance to the object in meters
KNOWN_PIXEL_WIDTH = 100
FOCAL_LENGTH = (KNOWN_PIXEL_WIDTH * KNOWN_DISTANCE) / KNOWN_WIDTH

def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Function to calculate distance
def calculate_distance(known_width, focal_length, perceived_width):
    if perceived_width == 0:
        return 0
    return (known_width * focal_length) / perceived_width

# Function to convert pixels to meters
def pixels_to_meters(perceived_width, perceived_height, known_width, focal_length):
    width_in_meters = (perceived_width * known_width) / KNOWN_PIXEL_WIDTH
    height_in_meters = (perceived_height * known_width) / KNOWN_PIXEL_WIDTH
    return width_in_meters, height_in_meters

# Function to generate video frames with object detection
def generate_frames(request):
    class_color = (255, 255, 0)  # Yellow for class name
    confidence_color = (255, 0, 0)  # Red for confidence
    distance_color = (0, 255, 0)  # Green for distance
    dimensions_color = (0, 0, 255)
    global videoCap, camera_running

    # Start the camera if it's not already running
        
    while camera_running:
        ret, frame = videoCap.read()
        if not ret:
            break

        results = yolo.track(frame, stream=True)
        for result in results:
            classes_names = result.names
            for box in result.boxes:
                if box.conf[0] > 0.4:
                    [x1, y1, x2, y2] = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    perceived_width = x2 - x1
                    perceived_height = y2 - y1
                    distance = calculate_distance(KNOWN_WIDTH, FOCAL_LENGTH, perceived_width)
                    width_in_meters, height_in_meters = pixels_to_meters(
                        perceived_width, perceived_height, KNOWN_WIDTH, FOCAL_LENGTH
                    )
                    rectangle_color = get_random_color()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), rectangle_color, 2)
                    
                    cls = int(box.cls[0])

                # get the class name
                    class_name = classes_names[cls]
                # create the label parts with specific colors for each part
                    label_class = f'{class_name} '
                    label_confidence = f'{box.conf[0]:.2f} '
                    label_distance = f'Distance: {distance:.2f}m '
                    label_width = f'W: {width_in_meters:.2f}m '
                    label_height = f'H: {height_in_meters:.2f}m'

                    # put the class name, confidence, distance, width, and height on the image
                    cv2.putText(frame, label_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, class_color, 2)
                    cv2.putText(frame, label_confidence, (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, confidence_color, 2)
                    cv2.putText(frame, label_distance, (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, distance_color, 2)
                    cv2.putText(frame, label_width, (x1, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dimensions_color, 2)
                    cv2.putText(frame, label_height, (x1, y1 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dimensions_color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    videoCap.release()

# Stream video feed
def video_feed(request):
    return StreamingHttpResponse(generate_frames(request), content_type='multipart/x-mixed-replace; boundary=frame')

# Handle start camera button
def start_camera_view(request):
    start_camera()
    return HttpResponseRedirect('/')

# Handle stop camera button
def stop_camera_view(request):
    stop_camera()
    return HttpResponseRedirect('/')

# Render homepage
def index(request):
    return render(request, 'index.html')
