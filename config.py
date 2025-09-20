# config.py - Configuration management for the attendance system
import json
import os
from typing import Dict, Any

class Config:
    """Configuration manager for the attendance system"""
    
    DEFAULT_CONFIG = {
        "camera": {
            "width": 640,
            "height": 480,
            "fps": 30,
            "camera_index": 0
        },
        "recognition": {
            "tolerance": 0.6,
            "face_detection_model": "hog",  # or "cnn" for better accuracy but slower
            "num_jitters": 1,
            "recognition_threshold": 0.6
        },
        "challenges": {
            "enabled": True,
            "types": ["blink", "turn_left", "turn_right", "open_mouth"],
            "duration": 5,
            "verification_threshold": {
                "eye_ar_thresh": 0.22,
                "mouth_ar_thresh": 0.60,
                "head_pose_thresh": 12
            }
        },
        "ui": {
            "theme": "arc",
            "window_width": 1300,
            "window_height": 720,
            "video_panel_width": 640,
            "video_panel_height": 480
        },
        "database": {
            "path": "attendance.db",
            "export_csv": True,
            "csv_export_path": "Attendance_Reports"
        },
        "logging": {
            "level": "INFO",
            "file": "attendance_system.log",
            "max_file_size": 10485760,  # 10MB
            "backup_count": 5
        },
        "deployment": {
            "mode": "local",  # local, gradio, web
            "gradio_share": False,
            "gradio_auth": None,  # [("username", "password")] or None
            "server_name": "127.0.0.1",
            "server_port": 7860
        }
    }
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self.DEFAULT_CONFIG.copy()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                self._update_config(self.config, loaded_config)
            else:
                self.save_config()
        except Exception as e:
            pass  # Use default configuration on error
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            pass  # Silent fail on save error
    
    def _update_config(self, base_config: Dict, new_config: Dict):
        """Recursively update configuration"""
        for key, value in new_config.items():
            if key in base_config:
                if isinstance(value, dict) and isinstance(base_config[key], dict):
                    self._update_config(base_config[key], value)
                else:
                    base_config[key] = value
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation"""
        try:
            keys = key_path.split('.')
            value = self.config
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation"""
        try:
            keys = key_path.split('.')
            config = self.config
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            config[keys[-1]] = value
            self.save_config()
        except Exception as e:
            pass  # Silent fail on configuration error
    
    def get_camera_config(self):
        """Get camera configuration"""
        return self.config.get("camera", {})
    
    def get_recognition_config(self):
        """Get face recognition configuration"""
        return self.config.get("recognition", {})
    
    def get_ui_config(self):
        """Get UI configuration"""
        return self.config.get("ui", {})
    
    def get_deployment_config(self):
        """Get deployment configuration"""
        return self.config.get("deployment", {})

# Global config instance
config = Config()