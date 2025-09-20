# continuous_attendance_system.py - True real-time continuous video stream
import gradio as gr
import cv2
import numpy as np
import face_recognition
import dlib
import os
import time
import random
import threading
from PIL import Image

# Import local modules
import database
import face_utils
from config import config

class ContinuousAttendanceSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.predictor = None
        self.user_states = {}
        self.video_capture = None
        self.is_streaming = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Performance optimization
        self.frame_skip_counter = 0
        self.process_every_nth_frame = 3  # Process face detection every 3rd frame (optimized for 120 FPS)
        self.last_face_locations = []  # Store last known face locations
        self.last_face_names = []  # Store last known face names
        
        # Challenge settings
        self.challenges = ['blink', 'turn_left', 'turn_right', 'open_mouth']
        self.challenge_timeout = 8  # seconds
        
        # Initialize system
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the attendance system"""
        try:
            database.initialize_db()
            
            # Load facial recognition models
            if os.path.exists("shape_predictor_68_face_landmarks.dat"):
                self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            else:
                return "Facial landmark predictor not found!"
            
            # Load known faces
            self.load_known_faces()
            
            return f"System ready - {len(self.known_face_names)} known faces loaded"
            
        except Exception as e:
            return f"System initialization failed: {e}"
    
    def load_known_faces(self):
        """Load known faces from directory"""
        self.known_face_encodings = []
        self.known_face_names = []
        
        if not os.path.exists("known_faces"):
            os.makedirs("known_faces")
            return
        
        for filename in os.listdir("known_faces"):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                try:
                    image_path = os.path.join("known_faces", filename)
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    
                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        name = os.path.splitext(filename)[0].replace("_", " ").title()
                        self.known_face_names.append(name)
                        
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    def start_continuous_stream(self):
        """Start continuous video streaming"""
        if self.is_streaming:
            return "Stream already running"
        
        # Clear user states on new session start
        self.user_states.clear()
        
        try:
            # Try different camera backends for better performance
            self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow for Windows
            if not self.video_capture.isOpened():
                # Fallback to default
                self.video_capture = cv2.VideoCapture(0)
                if not self.video_capture.isOpened():
                    return "Failed to open camera. Please check camera connection."
            
            # Set optimal camera properties for faster startup and better performance
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.video_capture.set(cv2.CAP_PROP_FPS, 120)  # Request high FPS from camera
            self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for real-time
            self.video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))  # Use MJPEG for speed
            
            # Warm up camera with a test frame
            ret, _ = self.video_capture.read()
            if not ret:
                self.video_capture.release()
                return "Camera initialization failed. Please try again."
            
            self.is_streaming = True
            
            # Start continuous processing thread
            self.stream_thread = threading.Thread(target=self._continuous_stream_loop, daemon=True)
            self.stream_thread.start()
            
            return "Camera started successfully. Face detection is now active."
            
        except Exception as e:
            if self.video_capture:
                self.video_capture.release()
            return f"Error starting camera: {e}"
    
    def stop_continuous_stream(self):
        """Stop continuous video streaming"""
        self.is_streaming = False
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        with self.frame_lock:
            self.current_frame = None
        
        return "Continuous stream stopped"
    
    def _continuous_stream_loop(self):
        """Continuous stream processing loop"""
        while self.is_streaming and self.video_capture:
            try:
                ret, frame = self.video_capture.read()
                if not ret:
                    continue
                
                # Process frame with optimized face detection
                processed_frame = self.process_continuous_frame_optimized(frame)
                
                # Update current frame (thread-safe)
                with self.frame_lock:
                    self.current_frame = processed_frame.copy()
                
                # Control frame rate for smooth video - optimized for ultra-smooth performance
                time.sleep(1/120)  # 120 FPS processing for maximum smoothness
                
            except Exception as e:
                time.sleep(0.1)
    
    def get_current_stream_frame(self):
        """Get the current processed frame"""
        if not self.is_streaming:
            # Create a placeholder frame
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera Not Started", (150, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            cv2.putText(placeholder, "Click 'Start Camera' to begin", (120, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            rgb_frame = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb_frame)
        
        with self.frame_lock:
            if self.current_frame is not None:
                # Convert BGR to RGB for display
                rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb_frame)
            else:
                # Loading frame
                loading = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(loading, "Loading camera...", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                rgb_frame = cv2.cvtColor(loading, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb_frame)
    
    def process_continuous_frame_optimized(self, frame):
        """Optimized frame processing with selective face detection"""
        try:
            # Make a copy for processing
            display_frame = frame.copy()
            current_time = time.time()
            
            # Add header with timestamp
            cv2.putText(display_frame, f"Live Attendance - {time.strftime('%H:%M:%S')}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Skip face detection on some frames for better performance
            self.frame_skip_counter += 1
            
            if self.frame_skip_counter % self.process_every_nth_frame == 0:
                # Full face detection every nth frame
                result_frame = self.process_continuous_frame(frame)
                return result_frame
            else:
                # Use cached face locations from previous detection
                if hasattr(self, 'last_face_locations') and self.last_face_locations:
                    # Draw previous face locations with lighter processing
                    for i, (face_location, name) in enumerate(zip(self.last_face_locations, self.last_face_names)):
                        top, right, bottom, left = face_location
                        
                        if name == "Unknown":
                            color = (0, 0, 255)  # Red
                            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                            cv2.putText(display_frame, "UNKNOWN", (left, top - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        else:
                            color = (0, 255, 0)  # Green
                            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                            cv2.putText(display_frame, f"{name}", (left, top - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Light processing - just show the frame with timestamp
                cv2.putText(display_frame, "Real-time video stream active", (10, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                return display_frame
                
        except Exception as e:
            cv2.putText(display_frame, f"Error: {str(e)[:50]}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return display_frame

    def process_continuous_frame(self, frame):
        """Process each frame continuously"""
        try:
            # Make a copy for processing
            display_frame = frame.copy()
            current_time = time.time()
            
            # Add header with timestamp
            cv2.putText(display_frame, f"Live Attendance - {time.strftime('%H:%M:%S')}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Resize for faster face detection - balanced approach
            small_frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)  # Better balance of speed and accuracy
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces with optimized settings
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog", number_of_times_to_upsample=1)
            
            if not face_locations:
                cv2.putText(display_frame, "Looking for faces...", (10, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display_frame, "Position yourself clearly in camera view", (10, 110), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                return display_frame
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            # Cache for performance optimization
            self.last_face_locations = []
            self.last_face_names = []
            
            # Process each detected face
            for face_encoding, face_location in zip(face_encodings, face_locations):
                # Scale coordinates back up (3.33x because we used 0.3 scale)
                top, right, bottom, left = [int(v * 3.33) for v in face_location]
                
                # Cache this face location
                self.last_face_locations.append((top, right, bottom, left))
                
                # Recognize face with better tolerance
                name = "Unknown"
                if len(self.known_face_encodings) > 0:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)  # More lenient
                    
                    if True in matches:
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index] and face_distances[best_match_index] < 0.5:  # More lenient
                            name = self.known_face_names[best_match_index]
                
                # Cache the name
                self.last_face_names.append(name)
                
                # Handle face recognition result
                if name == "Unknown":
                    self._draw_unknown_face(display_frame, left, top, right, bottom)
                else:
                    self._handle_known_face_continuous(display_frame, name, left, top, right, bottom, current_time)
            
            return display_frame
            
        except Exception as e:
            cv2.putText(display_frame, f"Error: {str(e)[:50]}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return display_frame
    
    def _draw_unknown_face(self, frame, left, top, right, bottom):
        """Draw unknown face detection"""
        color = (0, 0, 255)  # Red
        cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
        cv2.putText(frame, "UNKNOWN", (left, top - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, "Please register first", (left, bottom + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _handle_known_face_continuous(self, frame, name, left, top, right, bottom, current_time):
        """Handle known face with continuous challenge system"""
        # Initialize user state if needed
        if name not in self.user_states:
            # Check database only once during initialization
            has_attended = database.has_attended_today(name)
            self.user_states[name] = {
                'attended': has_attended,
                'challenge': None,
                'challenge_start_time': 0,
                'verified_session': has_attended,
                'last_seen': current_time,
                'challenge_assigned': False
            }
            if has_attended:
                pass  # User already attended today
        
        state = self.user_states[name]
        state['last_seen'] = current_time
        
        # IMPORTANT: Always verify attendance status with database to prevent false positives
        if state['attended']:
            # Double-check with database to ensure attendance is actually marked
            actual_attended = database.has_attended_today(name)
            if not actual_attended:
                # State was wrong, reset it
                state['attended'] = False
                state['verified_session'] = False
        
        if state['attended']:
            # Already attended today - show GREEN box
            color = (0, 255, 0)  # Bright Green
            cv2.rectangle(frame, (left, top), (right, bottom), color, 4)
            cv2.putText(frame, f"{name}", (left, top - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, "ATTENDANCE MARKED", (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, "Welcome back!", (left, bottom + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return
        
        # User needs to mark attendance - show ORANGE box for challenge
        challenge_color = (0, 165, 255)  # Orange (BGR format)
        
        # Assign challenge if none exists and none has been assigned yet
        if state['challenge'] is None and not state['challenge_assigned']:
            state['challenge'] = random.choice(self.challenges)
            state['challenge_start_time'] = current_time
            state['challenge_assigned'] = True
        
        # If we have a challenge, process it
        if state['challenge'] is not None:
            challenge = state['challenge']
            time_elapsed = current_time - state['challenge_start_time']
            time_remaining = self.challenge_timeout - time_elapsed
            
            if time_remaining <= 0:
                # Challenge timeout - show RED for timeout
                state['challenge'] = None
                state['challenge_assigned'] = False
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
                cv2.putText(frame, f"{name}", (left, top - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "CHALLENGE TIMEOUT", (left, top - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, "New challenge coming...", (left, bottom + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                return
            
            # Active challenge - show ORANGE box
            cv2.rectangle(frame, (left, top), (right, bottom), challenge_color, 4)
            
            # Display challenge instruction with better positioning
            challenge_text = challenge.replace('_', ' ').upper()
            
            # Name at the top
            cv2.putText(frame, f"{name}", (left, top - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Challenge status
            cv2.putText(frame, "CHALLENGE ACTIVE", (left, top - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Main challenge instruction - use bright yellow for visibility
            cv2.putText(frame, f"PLEASE {challenge_text}", (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Timer with orange color
            cv2.putText(frame, f"Time: {time_remaining:.1f}s", (left, bottom + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, challenge_color, 2)
            
            # Specific instruction in white
            instruction = self._get_challenge_instruction(challenge)
            cv2.putText(frame, instruction, (left, bottom + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Verify challenge continuously
            if self._verify_challenge_continuous(frame, name, dlib.rectangle(left, top, right, bottom)):
                # Challenge completed! Show SUCCESS with bright green
                try:
                    success, message = database.mark_attendance(name)
                    if success:
                        # Update state
                        state['attended'] = True
                        state['verified_session'] = True
                        state['challenge'] = None
                        state['challenge_assigned'] = False
                        
                        # Show success with BRIGHT GREEN and thicker border
                        success_color = (0, 255, 0)  # Bright Green
                        cv2.rectangle(frame, (left, top), (right, bottom), success_color, 6)
                        cv2.putText(frame, f"{name}", (left, top - 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        cv2.putText(frame, "SUCCESS!", (left, top - 35), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, success_color, 3)
                        cv2.putText(frame, "ATTENDANCE MARKED", (left, top - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, success_color, 2)
                        cv2.putText(frame, "Thank you!", (left, bottom + 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    else:
                        # Attendance already marked (shouldn't happen with our state management)
                        state['attended'] = True
                        state['challenge'] = None
                        state['challenge_assigned'] = False
                except Exception as e:
                    cv2.putText(frame, "Error - Try again", (left, bottom + 75), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    def _get_challenge_instruction(self, challenge):
        """Get instruction text for challenge"""
        instructions = {
            'blink': "Close your eyes completely",
            'turn_left': "Turn your head to the left",
            'turn_right': "Turn your head to the right", 
            'open_mouth': "Open your mouth wide"
        }
        return instructions.get(challenge, "Follow the instruction")
    
    def _verify_challenge_continuous(self, frame, name, face_rect):
        """Verify challenge completion continuously"""
        if not self.predictor or name not in self.user_states:
            return False
        
        state = self.user_states[name]
        challenge = state['challenge']
        
        try:
            # Get facial landmarks
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            landmarks = self.predictor(gray, face_rect)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
            
            # Verify specific challenge with stricter thresholds for reliability
            if challenge == 'blink':
                left_eye = landmarks[36:42]
                right_eye = landmarks[42:48]
                left_ear = face_utils.get_eye_aspect_ratio(left_eye)
                right_ear = face_utils.get_eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2
                return avg_ear < 0.22  # Stricter threshold
                
            elif challenge == 'turn_left':
                pose = face_utils.get_head_pose(landmarks)
                return pose == "left"
                
            elif challenge == 'turn_right':
                pose = face_utils.get_head_pose(landmarks)
                return pose == "right"
                
            elif challenge == 'open_mouth':
                mouth = landmarks[48:68]
                mar = face_utils.get_mouth_aspect_ratio(mouth)
                return mar > 0.6  # Stricter threshold
                
        except Exception as e:
            return False
        
        return False
    
    def get_attendance_data(self):
        """Get attendance data for today"""
        try:
            summary = database.get_daily_attendance_summary()
            today_str = summary['date']
            
            # Build detailed attendance report
            report_lines = [
                f"Daily Attendance Report - {today_str}",
                "=" * 50,
                f"Summary:",
                f"   • Total Attendees: {summary['total_attendees']}",
                "",
                "Attendance Details:",
                "-" * 30
            ]
            
            if summary['attendees']:
                for name, timestamp in summary['attendees']:
                    report_lines.append(f"{name} - {timestamp}")
                
                report_lines.append("")
                
                # Add CSV file info
                csv_path = f"Attendance_Reports/Attendance_{today_str}.csv"
                if os.path.exists(csv_path):
                    report_lines.extend([
                        "-" * 30,
                        f"Exported to: {csv_path}",
                        f"Last updated: {time.strftime('%H:%M:%S')}"
                    ])
            else:
                report_lines.append("No attendance recorded today.")
            
            return "\n".join(report_lines)
                
        except Exception as e:
            return f"Error loading attendance data: {e}"
    
    def register_new_user(self, name, reg_image, reg_webcam):
        """Register new user"""
        if not name or not name.strip():
            return "Please enter a valid name"
        
        # Use webcam image if available, otherwise use uploaded image
        image_to_use = reg_webcam if reg_webcam is not None else reg_image
        
        if image_to_use is None:
            return "Please provide an image either by uploading or taking a photo"
        
        try:
            name = name.strip()
            
            # Convert PIL to OpenCV
            cv_image = cv2.cvtColor(np.array(image_to_use), cv2.COLOR_RGB2BGR)
            
            # Check for face
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            
            if not face_locations:
                return "No face detected in the image. Please ensure your face is clearly visible."
            
            if len(face_locations) > 1:
                return "Multiple faces detected. Please ensure only one person is in the image."
            
            # Save the image
            filename = f"{name.replace(' ', '_').lower()}.jpg"
            filepath = os.path.join("known_faces", filename)
            cv2.imwrite(filepath, cv_image)
            
            # Reload known faces
            self.load_known_faces()
            
            return f"{name} registered successfully! You can now use the Live Attendance system."
            
        except Exception as e:
            return f"Registration failed: {e}"

# Create the application instance
app = ContinuousAttendanceSystem()

# Create Gradio interface with automatic streaming
def create_interface():
    with gr.Blocks(title="Real-Time Attendance System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Real-Time Face Recognition Attendance System
        **Advanced biometric attendance tracking with facial recognition**
        
        This system provides secure, automated attendance recording using facial recognition technology.
        The real-time video stream processes faces continuously with challenge-based verification for enhanced security.
        """)
        
        # System status
        init_status = gr.Textbox(
            value=app.initialize_system(),
            label="System Status",
            interactive=False
        )
        
        with gr.Tabs():
            with gr.Tab("Live Attendance"):
                with gr.Row():
                    with gr.Column(scale=2):
                        # Control buttons
                        with gr.Row():
                            start_btn = gr.Button("Start Camera", variant="primary", size="lg")
                            stop_btn = gr.Button("Stop Camera", variant="secondary", size="lg")
                        
                        # Automatic continuous video display - updates every 33ms
                        video_stream = gr.Image(
                            # label="Live Video Stream - Auto-updating at 30 FPS",
                            type="pil",
                            height=500
                        )

                        # Auto-update timer - triggers every 8ms for ultra-smooth video
                        timer = gr.Timer(value=0.008)  # 8ms = ~120 FPS
                        
                        gr.Markdown("""
                        ### System Operations
                        
                        **Camera Controls:**
                        - Start Camera: Activates the video stream and face detection
                        - Stop Camera: Terminates the session and releases camera resources
                        
                        **Daily Attendance Process:**
                        - Video stream updates automatically at 120 FPS
                        - Face detection operates intelligently with enhanced accuracy
                        - **Once Per Day**: Each user can mark attendance once per day
                        - Improved face recognition from normal distance
                        - Verified users receive challenge prompts for security
                        - Attendance is recorded automatically upon successful challenge completion
                        
                        **Status Indicators:**
                        - **Green border**: Attendance already marked for today
                        - **Orange border**: Ready to mark attendance - complete challenge
                        - **Red border**: Unregistered user (registration required)
                        
                        **Attendance Information:**
                        - Each person can attend once per day
                        - Challenge verification ensures security
                        - All attendance is tracked and exported to daily CSV reports
                        """)
                    
                    with gr.Column(scale=1):
                        attendance_log = gr.Textbox(
                            label="Daily Attendance Records",
                            lines=25,
                            interactive=False,
                            value=app.get_attendance_data()
                        )
                        
                        refresh_attendance_btn = gr.Button("Refresh Records", variant="secondary")
            
            with gr.Tab("User Registration"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### User Registration Portal")
                        
                        reg_name = gr.Textbox(
                            label="Full Name",
                            placeholder="Enter the person's full name"
                        )
                        
                        with gr.Tabs():
                            with gr.Tab("Upload Photo"):
                                reg_image = gr.Image(
                                    label="Profile Photo",
                                    type="pil",
                                    height=400
                                )
                            
                            with gr.Tab("Take Photo"):
                                reg_webcam = gr.Image(
                                    label="Webcam Capture",
                                    sources=["webcam"],
                                    type="pil",
                                    height=400
                                )
                        
                        register_btn = gr.Button("Register User", variant="primary", size="lg")
                        
                        reg_status = gr.Textbox(
                            label="Registration Status",
                            interactive=False
                        )
                    
                    with gr.Column():
                        gr.Markdown("""
                        ### Registration Requirements
                        
                        **Photo Specifications:**
                        - High-quality, well-illuminated facial image
                        - Subject facing directly toward camera
                        - Single individual per photograph
                        - Clear facial features without obstructions
                        
                        **Photo Options:**
                        - **Upload Photo**: Select an existing image file
                        - **Take Photo**: Use webcam to capture live photo
                        
                        **Registration Process:**
                        - Enter the person's full name in the text field
                        - Choose either "Upload Photo" or "Take Photo" tab
                        - Provide a clear facial image using your preferred method
                        - Click "Register User" to complete registration
                        - System will process and store the facial data
                        - Registration confirmation will appear below
                        
                        **After Registration:**
                        - User can then use the Live Attendance tab
                        - Camera will recognize the registered face
                        - Security challenges will verify identity
                        """)
        
        # Event handlers
        start_btn.click(
            app.start_continuous_stream,
            outputs=[init_status]
        )
        
        stop_btn.click(
            app.stop_continuous_stream,
            outputs=[init_status]
        )
        
        # Auto-update video stream using timer
        timer.tick(
            fn=app.get_current_stream_frame,
            outputs=[video_stream]
        )
        
        refresh_attendance_btn.click(
            app.get_attendance_data,
            outputs=[attendance_log]
        )
        
        register_btn.click(
            app.register_new_user,
            inputs=[reg_name, reg_image, reg_webcam],
            outputs=[reg_status]
        )
        
        return demo

if __name__ == "__main__":
    print("Starting Real-Time Face Recognition Attendance System...")
    print("Video Stream: 120 FPS ultra-smooth processing")
    print("Face Detection: Enhanced accuracy with intelligent optimization")
    print("Recognition Distance: Improved for normal camera distance")
    print("Web Interface: http://127.0.0.1:7866")
    print("System Status: Ready for operation")
    
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7866,
        share=False,
        show_error=True
    )