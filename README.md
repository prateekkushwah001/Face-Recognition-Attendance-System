# Face Recognition Attendance System

A production-ready face recognition attendance system featuring real-time video processing, anti-spoofing security measures, and automated attendance tracking.

## Overview

This system provides secure, automated attendance recording using advanced facial recognition technology. The application features real-time video streaming at 120 FPS, challenge-based liveness detection to prevent spoofing, and comprehensive attendance management with automated reporting.

## Features

### Core Functionality
- Real-time face recognition with 120 FPS video processing
- Anti-spoofing protection through challenge-response verification
- Automated attendance logging with SQLite database storage
- Daily CSV report generation with timestamp tracking
- User registration with photo capture capabilities

### Security Features
- Liveness detection with multiple challenge types (blink, head movement, mouth opening)
- Session management to prevent duplicate entries
- Configurable recognition tolerance and thresholds
- Comprehensive audit trail and logging system

### Technical Specifications
- Optimized performance with intelligent frame processing (every 3rd frame)
- Thread-safe video streaming with frame synchronization
- Professional web interface built with Gradio framework
- Configurable system parameters through JSON configuration

## System Requirements

### Prerequisites
- Python 3.8 or higher
- Webcam or camera device
- Windows, macOS, or Linux operating system

### Required Files
- `shape_predictor_68_face_landmarks.dat` - Facial landmark detection model

## Installation

1. Clone or download the project repository
2. Navigate to the project directory
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download the facial landmark predictor:
   - Download `shape_predictor_68_face_landmarks.dat.bz2` from [dlib models](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
   - Extract the file to the project root directory

## Quick Start

### Installation and Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/prateekkushwah001/face-recognition-attendance-system.git
   cd face-recognition-attendance-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the facial landmark predictor**
   - Download `shape_predictor_68_face_landmarks.dat.bz2`
   - Extract the file to the project root directory

### Running the Application

```bash
python attendance_system_application.py
```

The system will start the web interface accessible at `http://127.0.0.1:7866`

### Initial Setup
1. Access the web interface through your browser
2. Navigate to the "User Registration" tab
3. Register users by entering their name and capturing a clear facial photo
4. Return to the "Live Attendance" tab to begin attendance tracking

## Usage

### User Registration
1. Enter the user's full name in the registration form
2. Capture a clear, well-lit facial photograph
3. Ensure only one person is visible in the frame
4. Click "Register User" to complete the process

### Attendance Tracking
1. Click "Start Camera" to begin video streaming
2. Position users in front of the camera for face detection
3. Registered users will receive challenge prompts for verification
4. Complete the displayed challenge (blink, head turn, or mouth opening)
5. Attendance is automatically recorded upon successful verification

### Status Indicators
- **Green Border**: Attendance already marked for the current day
- **Orange Border**: Challenge active - complete the displayed action
- **Red Border**: Unregistered user - registration required

## Configuration

The system uses `config.json` for customization:

```json
{
    "camera": {
        "width": 640,
        "height": 480,
        "fps": 30,
        "camera_index": 0
    },
    "recognition": {
        "tolerance": 0.6,
        "face_detection_model": "hog"
    },
    "challenges": {
        "enabled": true,
        "types": ["blink", "turn_left", "turn_right", "open_mouth"],
        "duration": 5
    }
}
```

## Project Structure

```
attendance-system/
├── attendance_system_application.py    # Main application file
├── database.py                        # Database operations
├── face_utils.py                     # Face recognition utilities
├── config.py                         # Configuration management
├── requirements.txt                  # Python dependencies
├── config.json                       # System configuration
├── attendance.db                     # SQLite database (auto-generated)
├── known_faces/                      # Registered user photos
├── Attendance_Reports/               # Daily CSV reports
└── shape_predictor_68_face_landmarks.dat  # Facial landmark model
```

## Database Schema

The system uses SQLite with the following schema:

```sql
CREATE TABLE attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    attendance_date TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    UNIQUE(name, attendance_date)
);
```

## API Reference

### Core Classes

#### ContinuousAttendanceSystem
Main application class handling video streaming, face recognition, and attendance management.

**Key Methods:**
- `start_continuous_stream()`: Initialize camera and begin video processing
- `stop_continuous_stream()`: Terminate video streaming
- `register_new_user()`: Add new user to the system
- `get_attendance_data()`: Retrieve daily attendance summary

### Database Functions

#### database.py
- `initialize_db()`: Create database tables
- `mark_attendance(name)`: Record attendance entry
- `has_attended_today(name)`: Check attendance status
- `export_to_csv()`: Generate daily reports

### Face Recognition Utilities

#### face_utils.py
- `get_eye_aspect_ratio()`: Calculate eye closure measurement
- `get_mouth_aspect_ratio()`: Calculate mouth opening measurement
- `get_head_pose()`: Determine head orientation
- `verify_challenge()`: Validate challenge completion

## Performance Optimization

### Video Processing
- Frame processing optimized to every 3rd frame for 120 FPS performance
- Intelligent caching of face locations between frames
- Thread-safe video streaming with minimal latency

### Face Recognition
- HOG model for fast detection with balanced accuracy
- Configurable tolerance levels for recognition sensitivity
- Optimized image scaling (0.3x) for faster processing

## Troubleshooting

### Common Issues

**Camera Access Problems**
- Verify camera permissions in system settings
- Test different camera indices (0, 1, 2) in configuration
- Ensure no other applications are using the camera

**Face Recognition Issues**
- Confirm adequate lighting conditions
- Verify `shape_predictor_68_face_landmarks.dat` file exists
- Check face images are clear and properly positioned

**Installation Problems**
- Update pip: `pip install --upgrade pip`
- Install cmake if dlib installation fails
- On Windows, install Visual C++ Build Tools

### Performance Tuning

**For Enhanced Performance**
- Reduce camera resolution in configuration
- Adjust frame processing intervals
- Use GPU-accelerated OpenCV if available

**Memory Management**
- Monitor system resources with large user databases
- Adjust recognition tolerance for accuracy vs speed balance

## Security Considerations

- The system implements liveness detection to prevent photo spoofing
- Challenge verification ensures real human presence
- Session management prevents duplicate attendance entries
- All attendance data is logged with timestamps for audit purposes

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## Issues and Bug Reports

If you encounter any issues:
1. Check the troubleshooting section above
2. Review system logs in `attendance_system.log`
3. Open an issue on GitHub with detailed information about the problem

## Repository Structure

This repository contains all necessary files for local deployment:
- Core application files
- Database management modules
- Configuration system
- Documentation and setup guides




## Technical Specifications

- **Framework**: Gradio 4.0+ for web interface
- **Computer Vision**: OpenCV with face_recognition library
- **Database**: SQLite with automatic CSV export
- **Performance**: 120 FPS video processing with intelligent optimization
- **Security**: Multi-factor liveness detection and challenge verification