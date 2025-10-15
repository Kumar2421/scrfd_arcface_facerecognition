# Smart Face Recognition System - Configuration Guide

## Overview

The Smart Face Recognition System now uses a centralized JSON configuration file (`config.json`) to manage all thresholds, parameters, and settings. This makes it easy to adjust the system behavior without modifying the source code.

## Configuration File Structure

The `config.json` file is organized into logical categories:

### 1. System Configuration (`system`)
- `database_path`: Path to SQLite database file
- `model_name`: InsightFace model name (e.g., "buffalo_l")
- `gpu_id`: GPU ID (0 for GPU, -1 for CPU)
- `image_cache_dir`: Directory for cached images

### 2. Face Detection (`face_detection`)
- `detection_size`: Face detection input size [width, height]
- `confidence_threshold`: Minimum confidence for face detection
- `quality_threshold`: Minimum face quality for registration
- `min_quality_threshold`: Minimum quality to avoid rejecting good images
- `pose_angle_threshold`: Maximum pose angle in degrees
- `yaw_threshold`: Maximum yaw angle in degrees
- `pitch_threshold`: Maximum pitch angle in degrees

### 3. Face Recognition (`face_recognition`)
- `similarity_threshold`: Base similarity threshold for matching
- `grouping_threshold_file`: Similarity threshold for file processing
- `grouping_threshold_json`: Similarity threshold for JSON processing
- `duplicate_similarity_threshold`: Threshold for detecting duplicate images
- `merge_duplicate_threshold`: Threshold for merging duplicate persons

### 4. Face Quality Assessment (`face_quality`)
- `weights`: Quality score weights for different factors
  - `detection_score`: Weight for detection confidence
  - `size_score`: Weight for face size
  - `blur_score`: Weight for blur assessment
  - `pose_score`: Weight for pose assessment
  - `lighting_score`: Weight for lighting assessment
- `size_normalization`: Normalization factor for face size scoring
- `min_overall_score`: Minimum overall quality score

### 5. Side Face Detection (`side_face_detection`)
- `aspect_ratio_thresholds`: Thresholds for aspect ratio analysis
- `area_thresholds`: Thresholds for face area analysis
- `compactness_thresholds`: Thresholds for compactness analysis
- `confidence_thresholds`: Thresholds for detection confidence
- `decision_threshold`: Final decision threshold for side face detection
- `edge_position_threshold`: Threshold for edge position analysis

### 6. Image Processing (`image_processing`)
- `web_max_size`: Maximum size for web display [width, height]
- `jpeg_quality`: JPEG compression quality (1-100)
- `download_timeout`: Timeout for image downloads in seconds
- `max_workers`: Maximum number of processing threads

### 7. Web Interface (`web_interface`)
- `host`: Web server host address
- `port`: Web server port
- `cache_control_max_age`: Cache control max age in seconds

### 8. Processing (`processing`)
- `max_visits_fallback`: Fallback max visits for processing
- `max_visits_default`: Default max visits for processing
- `save_images_default`: Default setting for saving images
- `clear_existing_default`: Default setting for clearing existing data

### 9. HTTP Headers (`http_headers`)
- `user_agent`: User agent string for HTTP requests
- `accept`: Accept header for HTTP requests
- `accept_language`: Accept-Language header
- `cache_control`: Cache-Control header

## Usage Examples

### 1. Using Default Configuration
```python
from smart_face_recognition import SmartFaceRecognition

# Uses config.json automatically
face_recognition = SmartFaceRecognition()
```

### 2. Overriding Specific Parameters
```python
# Override specific parameters while using config.json for others
face_recognition = SmartFaceRecognition(
    confidence_thresh=0.6,  # Override confidence threshold
    similarity_thresh=0.5   # Override similarity threshold
)
```

### 3. Using Custom Configuration File
```python
# Use a different configuration file
face_recognition = SmartFaceRecognition(config_file="my_config.json")
```

### 4. Modifying Configuration at Runtime
```python
# Load system with default config
face_recognition = SmartFaceRecognition()

# Modify configuration at runtime
face_recognition.config['face_recognition']['grouping_threshold_file'] = 0.7
face_recognition.config['image_processing']['max_workers'] = 8
```

## Configuration Best Practices

### 1. Backup Your Configuration
Always backup your `config.json` file before making changes:
```bash
cp config.json config.json.backup
```

### 2. Test Changes Incrementally
Make small changes and test the system to ensure it works as expected.

### 3. Document Custom Settings
If you modify the configuration, document the changes and their purpose.

### 4. Use Version Control
Keep your configuration files in version control to track changes.

## Common Configuration Adjustments

### Increasing Face Detection Sensitivity
```json
{
  "face_detection": {
    "confidence_threshold": 0.3,
    "quality_threshold": 0.2
  }
}
```

### Making Recognition More Strict
```json
{
  "face_recognition": {
    "similarity_threshold": 0.6,
    "grouping_threshold_file": 0.7,
    "grouping_threshold_json": 0.8
  }
}
```

### Improving Side Face Detection
```json
{
  "side_face_detection": {
    "decision_threshold": 3,
    "aspect_ratio_thresholds": {
      "strong_profile": 0.7
    }
  }
}
```

### Optimizing Performance
```json
{
  "image_processing": {
    "max_workers": 8,
    "download_timeout": 60
  }
}
```

## Troubleshooting

### Configuration File Not Found
If `config.json` is not found, the system will use default values and print a warning message.

### Invalid JSON Format
If the JSON file has syntax errors, the system will raise an exception. Check the JSON syntax using a JSON validator.

### Missing Configuration Keys
If a required configuration key is missing, the system will raise a KeyError. Ensure all required keys are present in your configuration file.

## Migration from Hardcoded Values

The system has been updated to use the configuration file instead of hardcoded values. All previous hardcoded thresholds and parameters are now configurable through the JSON file.

### Before (Hardcoded)
```python
# Old hardcoded approach
confidence_thresh = 0.5
similarity_thresh = 0.4
quality_thresh = 0.3
```

### After (Configuration-based)
```python
# New configuration-based approach
face_recognition = SmartFaceRecognition()  # Uses config.json
# or
face_recognition = SmartFaceRecognition(
    confidence_thresh=0.5,  # Override specific values
    similarity_thresh=0.4,
    quality_thresh=0.3
)
```

## Example Configuration File

See `config.json` for a complete example with all available configuration options and their default values.
