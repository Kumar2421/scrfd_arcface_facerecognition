#!/usr/bin/env python3
"""
Smart Face Recognition System with NumPy-based Similarity Search + Database
Solves classical face recognition problems: face mismatch, false positives, scalability
No FAISS dependency - uses NumPy for fast similarity search
"""

import os
import cv2
import sqlite3
import numpy as np
import math
# import faiss  # Using NumPy-based similarity search instead
import logging
import json
import time
import requests
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from insightface.app import FaceAnalysis
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from json_storage import save_clustering_results
import hashlib
from urllib.parse import urlparse
import tempfile
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from compare_face_from_api import FaceComparisonFromAPI
from qdrant_manager import QdrantManager
import uvicorn
import base64
from PIL import Image
import io

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")


def load_api_config():
    """Load API configuration from api_config.txt file"""
    config = {
        'api_url': 'https://live.thefusionapps.com/api/v2/retail/event/all',
        'auth_token': '',
        'api_key': '',
        'default_start_date': '2025-10-13',
        'default_end_date': '2025-10-13',
        'default_start_time': '12:33:33',
        'default_end_time': '13:03:33',
        'default_page': 0,
        'default_limit': 100,
        'default_all_branch': True,
        'default_max_visits': 100
    }
    
    try:
        if os.path.exists('api_config.txt'):
            with open('api_config.txt', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        
                        if key == 'api_url':
                            config['api_url'] = value
                        elif key == 'auth_token':
                            config['auth_token'] = value
                        elif key == 'api_key':
                            config['api_key'] = value
                        elif key == 'default_start_date':
                            config['default_start_date'] = value
                        elif key == 'default_end_date':
                            config['default_end_date'] = value
                        elif key == 'default_start_time':
                            config['default_start_time'] = value
                        elif key == 'default_end_time':
                            config['default_end_time'] = value
                        elif key == 'default_page':
                            config['default_page'] = int(value)
                        elif key == 'default_limit':
                            config['default_limit'] = int(value)
                        elif key == 'default_all_branch':
                            config['default_all_branch'] = value.lower() == 'true'
                        elif key == 'default_max_visits':
                            config['default_max_visits'] = int(value)
                        elif key == 'api_key':
                            config['api_key'] = value
    except Exception as e:
        print(f"Warning: Could not load api_config.txt: {e}")
    
    return config


class SmartFaceRecognition:
    def __init__(self, 
                 database_path: str = None,
                 model_name: str = None,
                 gpu_id: int = None,
                 confidence_thresh: float = None,
                 similarity_thresh: float = None,
                 quality_thresh: float = None,
                 config_file: str = "config.json"):
        """
        Initialize Smart Face Recognition System
        
        Args:
            database_path: Path to SQLite database (overrides config)
            model_name: InsightFace model name (overrides config)
            gpu_id: GPU ID (0 for GPU, -1 for CPU) (overrides config)
            confidence_thresh: Face detection confidence threshold (overrides config)
            similarity_thresh: Face similarity threshold for matching (overrides config)
            quality_thresh: Face quality threshold for registration (overrides config)
            config_file: Path to configuration JSON file
        """
        # Load configuration from JSON file
        self.config = self.load_config(config_file)
        
        # Initialize logger after config is loaded
        self.setup_logging()
        
        # Set configuration values (command line args override config file)
        self.database_path = database_path or self.config['system']['database_path']
        self.model_name = model_name or self.config['system']['model_name']
        self.gpu_id = gpu_id if gpu_id is not None else self.config['system']['gpu_id']
        self.confidence_thresh = confidence_thresh if confidence_thresh is not None else self.config['face_detection']['confidence_threshold']
        self.similarity_thresh = similarity_thresh if similarity_thresh is not None else self.config['face_recognition']['similarity_threshold']
        self.quality_thresh = quality_thresh if quality_thresh is not None else self.config['face_detection']['quality_threshold']
        
        # Initialize components
        self.app = None
        self.face_quality_cache = {}
        self.image_cache = {}  # Cache for processed images
        self.image_cache_dir = self.config['system']['image_cache_dir']  # Directory for cached images
        
        # Initialize vector database
        self.vector_db = QdrantManager(self.config)
        
        # Initialize webhook manager
        # JSON storage manager is imported globally
        
        # Setup
        self.setup_database()
        self.run_database_migrations()
        self.setup_image_cache()
        self.initialize_model()
        self.load_embeddings()
    
    def load_config(self, config_file: str) -> dict:
        """
        Load configuration from JSON file
        
        Args:
            config_file: Path to configuration JSON file
            
        Returns:
            Dictionary containing configuration
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"Configuration loaded from {config_file}")
            return config
        except FileNotFoundError:
            print(f"Configuration file {config_file} not found, using defaults")
            # Return default configuration if file not found
            return {
                'system': {
                    'database_path': 'face_database.db',
                    'model_name': 'buffalo_l',
                    'gpu_id': 0,
                    'image_cache_dir': 'image_cache'
                },
                'face_detection': {
                    'confidence_threshold': 0.5,
                    'quality_threshold': 0.3
                },
                'face_recognition': {
                    'similarity_threshold': 0.4
                }
            }
        except json.JSONDecodeError as e:
            print(f"Error parsing configuration file {config_file}: {e}")
            raise
        except Exception as e:
            print(f"Error loading configuration file {config_file}: {e}")
            raise
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_database(self):
        """Setup SQLite database for person metadata"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Create persons table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                image_path TEXT,
                face_quality REAL,
                face_hash TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                match_count INTEGER DEFAULT 0
            )
        ''')
        
        # Create face_quality table for quality assessment
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_quality (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                quality_score REAL,
                blur_score REAL,
                pose_score REAL,
                lighting_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons (id)
            )
        ''')
        
        # Create person_visits table for web interface
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS person_visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                visit_id TEXT,
                customer_id TEXT,
                entry_time TEXT,
                image_url TEXT,
                saved_image_path TEXT,
                similarity REAL,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        self.logger.info("Database setup completed")
    
    def run_database_migrations(self):
        """Run database migrations to update schema"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Migration 1: Add reason column to low_similarity_images table
            try:
                cursor.execute('ALTER TABLE low_similarity_images ADD COLUMN reason TEXT')
                conn.commit()
                self.logger.info("Added reason column to low_similarity_images table")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    self.logger.info("Reason column already exists in low_similarity_images table")
                else:
                    self.logger.warning(f"Could not add reason column: {e}")
            
            # Migration 2: Remove embedding column from persons table (moved to Qdrant)
            try:
                # Check if embedding column exists
                cursor.execute("PRAGMA table_info(persons)")
                columns = [column[1] for column in cursor.fetchall()]
                
                if 'embedding' in columns:
                    # SQLite doesn't support DROP COLUMN directly, so we need to recreate the table
                    self.logger.info("Migrating persons table to remove embedding column...")
                    
                    # Create new table without embedding column
                    cursor.execute('''
                        CREATE TABLE persons_new (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL,
                            image_path TEXT,
                            face_quality REAL,
                            face_hash TEXT UNIQUE,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            match_count INTEGER DEFAULT 0
                        )
                    ''')
                    
                    # Copy data from old table to new table (excluding embedding)
                    cursor.execute('''
                        INSERT INTO persons_new (id, name, image_path, face_quality, face_hash, created_at, last_seen, match_count)
                        SELECT id, name, image_path, face_quality, face_hash, created_at, last_seen, match_count
                        FROM persons
                    ''')
                    
                    # Drop old table and rename new table
                    cursor.execute('DROP TABLE persons')
                    cursor.execute('ALTER TABLE persons_new RENAME TO persons')
                    
                    conn.commit()
                    self.logger.info("Successfully migrated persons table - embedding column removed")
                else:
                    self.logger.info("Persons table already migrated - no embedding column found")
                    
            except Exception as e:
                self.logger.warning(f"Could not migrate persons table: {e}")
            
            conn.close()
        except Exception as e:
            self.logger.error(f"Error running database migrations: {e}")
    
    def setup_image_cache(self):
        """Setup image cache directory"""
        os.makedirs(self.image_cache_dir, exist_ok=True)
        self.logger.info(f"Image cache directory: {self.image_cache_dir}")
    
    def clear_database(self):
        """Clear all data from the database"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            # Clear all tables
            cursor.execute('DELETE FROM person_visits')
            cursor.execute('DELETE FROM face_quality')
            cursor.execute('DELETE FROM persons')
            
            # Reset auto-increment counters
            cursor.execute('DELETE FROM sqlite_sequence WHERE name IN ("persons", "face_quality", "person_visits")')
            
            conn.commit()
            self.logger.info("SQLite database cleared successfully")
            
            # Clear Qdrant vector database
            try:
                self.vector_db.clear_all()
                self.logger.info("Qdrant vector database cleared successfully")
            except Exception as e:
                self.logger.error(f"Error clearing Qdrant database: {e}")
            
        except Exception as e:
            self.logger.error(f"Error clearing database: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def initialize_model(self):
        """Initialize InsightFace model"""
        self.logger.info(f"Loading {self.model_name} model...")
        self.app = FaceAnalysis(name=self.model_name)
        det_size = tuple(self.config['face_detection']['detection_size'])
        self.app.prepare(ctx_id=self.gpu_id, det_size=det_size)
        self.logger.info("Model loaded successfully")
    
    def compute_face_hash(self, embedding: np.ndarray) -> str:
        """Compute hash for face embedding to detect duplicates"""
        return hashlib.md5(embedding.tobytes()).hexdigest()
    
    def get_cached_image_path(self, image_url: str) -> str:
        """Get cached image path or create cache entry"""
        # Create a hash of the URL for cache filename
        url_hash = hashlib.md5(image_url.encode()).hexdigest()
        cached_path = os.path.join(self.image_cache_dir, f"{url_hash}.jpg")
        
        # If cached image doesn't exist, download and cache it
        if not os.path.exists(cached_path):
            try:
                image = self.download_image_from_url(image_url, save_path=cached_path)
                if image is not None:
                    self.logger.info(f"Cached image: {image_url} -> {cached_path}")
                else:
                    # If download failed, return None
                    return None
            except Exception as e:
                self.logger.error(f"Error caching image {image_url}: {e}")
                return None
        
        return cached_path
    
    def process_image_for_web(self, image_path: str, max_size: tuple = None) -> Optional[str]:
        """
        Process image for web display with proper resizing and format
        
        Args:
            image_path: Path to the image file
            max_size: Maximum size (width, height) for the processed image
            
        Returns:
            Base64 encoded image string or None if processing failed
        """
        try:
            if not os.path.exists(image_path):
                return None
            
            # Use config default if max_size not provided
            if max_size is None:
                max_size = tuple(self.config['image_processing']['web_max_size'])
            
            # Open and process image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize while maintaining aspect ratio
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Convert to base64
                buffer = io.BytesIO()
                jpeg_quality = self.config['image_processing']['jpeg_quality']
                img.save(buffer, format='JPEG', quality=jpeg_quality, optimize=True)
                buffer.seek(0)
                
                # Encode to base64
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                return f"data:image/jpeg;base64,{img_base64}"
                
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            return None
    
    def download_image_from_url(self, url: str, timeout: int = None, save_path: str = None) -> Optional[np.ndarray]:
        """
        Download image from URL and return as OpenCV image
        
        Args:
            url: Image URL
            timeout: Request timeout in seconds
            save_path: Optional path to save the downloaded image
            
        Returns:
            OpenCV image array or None if failed
        """
        try:
            self.logger.info(f"Downloading image from: {url}")
            
            # Use config default if timeout not provided
            if timeout is None:
                timeout = self.config['image_processing']['download_timeout']
            
            # Set headers to request image content
            headers = {
                'User-Agent': self.config['http_headers']['user_agent'],
                'Accept': self.config['http_headers']['accept'],
                'Accept-Language': self.config['http_headers']['accept_language'],
                'Cache-Control': self.config['http_headers']['cache_control']
            }
            
            # Download image
            response = requests.get(url, timeout=timeout, stream=True, headers=headers)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            self.logger.info(f"Content-Type: {content_type}")
            
            # Check if response is JSON (error response)
            if 'application/json' in content_type:
                self.logger.warning(f"Received JSON response instead of image from: {url}")
                try:
                    json_response = response.json()
                    self.logger.warning(f"JSON response: {json_response}")
                except:
                    self.logger.warning(f"Could not parse JSON response")
                return None
            
            # Check if response is actually an image
            if not any(img_type in content_type for img_type in ['image/', 'application/octet-stream']):
                self.logger.warning(f"Unexpected content type: {content_type}")
                # Try to decode anyway in case it's an image with wrong headers
            
            # Convert to OpenCV format
            image_array = np.frombuffer(response.content, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                self.logger.warning(f"Failed to decode image from URL: {url}")
                # Try to save raw content for debugging
                if save_path:
                    raw_path = save_path.replace('.jpg', '_raw.bin')
                    with open(raw_path, 'wb') as f:
                        f.write(response.content)
                    self.logger.info(f"Saved raw content to: {raw_path}")
                return None
            
            # Save image if path provided
            if save_path:
                success = cv2.imwrite(save_path, image)
                if success:
                    self.logger.info(f"Saved image to: {save_path}")
                else:
                    self.logger.warning(f"Failed to save image to: {save_path}")
                
            self.logger.info(f"Successfully downloaded image: {image.shape}")
            return image
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error downloading {url}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error downloading image from {url}: {e}")
            return None
    
    def load_visit_data(self, json_file_path: str) -> List[Dict]:
        """
        Load visit data from JSON file
        
        Args:
            json_file_path: Path to JSON file containing visit data
            
        Returns:
            List of visit records with image URLs
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            visits = data.get('visits', [])
            self.logger.info(f"Loaded {len(visits)} visits from JSON file")
            
            # Filter visits with valid image URLs
            valid_visits = []
            for visit in visits:
                if visit.get('image') and visit.get('image').startswith('http'):
                    valid_visits.append(visit)
            
            self.logger.info(f"Found {len(valid_visits)} visits with valid image URLs")
            return valid_visits
            
        except FileNotFoundError:
            self.logger.error(f"JSON file not found: {json_file_path}")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON format in {json_file_path}: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error loading visit data: {e}")
            return []
    
    def fetch_face_comparison_data_from_api(self, api_url: str, start_date: str = None, end_date: str = None, 
                                           page: int = 0, limit: int = 100, start_time: str = None, 
                                           end_time: str = None, all_branch: bool = True, api_key: str = None, 
                                           auth_token: str = None) -> List[Dict]:
        """
        Fetch face comparison data from external API - New structure with url1 and url2
        
        Args:
            api_url: Base API URL
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            page: Page number for pagination
            limit: Number of records per page
            start_time: Start time in HH:MM:SS format
            end_time: End time in HH:MM:SS format
            all_branch: Whether to include all branches
            api_key: API key for authentication (optional)
            auth_token: Bearer token for authentication (optional)
            
        Returns:
            List of face comparison records
        """
        try:
            # Build query parameters for new analytics API
            params = {
                'page': page,
                'limit': limit,
                'allBranch': str(all_branch).lower()
            }
            
            # Add date parameter (new API uses single date parameter)
            if start_date:
                params['date'] = start_date
            
            # Add time parameters if provided
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            # Add other optional parameters
            params.update({
                'nolimit': 'false',
                'isZone': 'false',
                'BlackListed': 'false',
                'Vip': 'false',
                'Vendor': 'false',
                'isDeleted': 'false'
            })
            
            self.logger.info(f"Fetching face comparison data from API: {api_url}")
            self.logger.info(f"Parameters: {params}")
            
            # Prepare headers for authentication
            headers = {}
            if api_key:
                headers['X-API-Key'] = api_key
            if auth_token:
                headers['Authorization'] = f'Bearer {auth_token}'
            
            # Make API request
            response = requests.get(api_url, params=params, headers=headers, timeout=30)
            
            # Handle different HTTP status codes
            if response.status_code == 401:
                self.logger.error("API request failed: 401 Unauthorized. Please check your API credentials.")
                return []
            elif response.status_code == 403:
                self.logger.error("API request failed: 403 Forbidden. You don't have permission to access this resource.")
                return []
            elif response.status_code == 404:
                self.logger.error("API request failed: 404 Not Found. The API endpoint may be incorrect.")
                return []
            elif response.status_code == 429:
                self.logger.error("API request failed: 429 Too Many Requests. Rate limit exceeded.")
                return []
            elif not response.ok:
                self.logger.error(f"API request failed: {response.status_code} {response.reason}")
                return []
            
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            self.logger.info(f"API response received: {len(data) if isinstance(data, list) else 'object'} records")
            
            # Debug: Log the structure of the response (only in debug mode)
            if self.logger.level <= 10:  # DEBUG level
                if isinstance(data, list) and len(data) > 0:
                    self.logger.debug(f"First record structure: {list(data[0].keys()) if data[0] else 'Empty record'}")
                elif isinstance(data, dict):
                    self.logger.debug(f"Response keys: {list(data.keys())}")
                    if 'data' in data:
                        self.logger.debug(f"Data field type: {type(data['data'])}, length: {len(data['data']) if isinstance(data['data'], list) else 'N/A'}")
            
            # Extract visits from response - Updated for new API structure
            raw_visits = data if isinstance(data, list) else data.get('visits', data.get('data', []))
            
            if not raw_visits:
                self.logger.warning("No visits found in API response")
                return []
            
            # Transform visits to face comparison format
            comparison_records = []
            for i, visit in enumerate(raw_visits):
                try:
                    # Extract image URLs from API structure
                    # The API returns 'image' (current visit) and 'refImage' (reference from database)
                    image1_url = visit.get('image')  # Current visit image
                    image2_url = visit.get('refImage')  # Reference image from database
                    
                    if not image1_url or not image2_url:
                        self.logger.debug(f"No image URLs found for visit {visit.get('id', 'unknown')}")
                        continue
                    
                    # Map fields from API structure
                    comparison_record = {
                        'comparison_id': visit.get('id', f"comparison_{len(comparison_records)}"),
                        'event_id': visit.get('entryEventIds', [None])[0] if visit.get('entryEventIds') else None,
                        'approve': visit.get('isConverted', False),  # Using isConverted as approval indicator
                        'image1_url': image1_url,
                        'image2_url': image2_url,
                        'branch_id': visit.get('branchId'),
                        'created_at': visit.get('entryTime'),
                        'customer_info': [visit.get('customerId')] if visit.get('customerId') else [],
                        'matched_info': [visit.get('refImage')] if visit.get('refImage') else [],
                        'message': f"Visit comparison for customer {visit.get('customerId', 'unknown')}",
                        'is_first_visit': visit.get('isFirstVisit', False),
                        'is_vip': visit.get('isVip', False),
                        'is_blacklisted': visit.get('isBlackListed', False),
                        'raw_data': visit
                    }
                    
                    comparison_records.append(comparison_record)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing visit: {e}")
                    continue
            
            self.logger.info(f"Transformed {len(comparison_records)} face comparison records")
            return comparison_records
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error fetching face comparison data from API: {e}")
            return []

    def fetch_visit_data_from_api(self, api_url: str, start_date: str = None, end_date: str = None, 
                                 page: int = 0, limit: int = 100, start_time: str = None, 
                                 end_time: str = None, all_branch: bool = True, api_key: str = None, 
                                 auth_token: str = None) -> List[Dict]:
        """
        Fetch visit data from external API
        
        Args:
            api_url: Base API URL
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            page: Page number for pagination
            limit: Number of records per page
            start_time: Start time in HH:MM:SS format
            end_time: End time in HH:MM:SS format
            all_branch: Whether to include all branches
            api_key: API key for authentication (optional)
            auth_token: Bearer token for authentication (optional)
            
        Returns:
            List of visit records with image URLs
        """
        try:
            # Build query parameters for new analytics API
            params = {
                'page': page,
                'limit': limit,
                'allBranch': str(all_branch).lower()
            }
            
            # Add date parameter (new API uses single date parameter)
            if start_date:
                params['date'] = start_date
            
            # Add time parameters if provided
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            # Add optional parameters only if they make sense
            # Remove restrictive parameters that might cause 0 results
            if not all_branch:
                params['allBranch'] = 'false'
            
            # Add other optional parameters
            params.update({
                'nolimit': 'false',
                'isZone': 'false',
                'BlackListed': 'false',
                'Vip': 'false',
                'Vendor': 'false',
                'isDeleted': 'false'
            })
            
            self.logger.info(f"Fetching data from API: {api_url}")
            self.logger.info(f"Parameters: {params}")
            
            # Prepare headers for authentication
            headers = {}
            if api_key:
                headers['X-API-Key'] = api_key
            if auth_token:
                headers['Authorization'] = f'Bearer {auth_token}'
            
            # Make API request
            response = requests.get(api_url, params=params, headers=headers, timeout=30)
            
            # Handle different HTTP status codes
            if response.status_code == 401:
                self.logger.error("API request failed: 401 Unauthorized. Please check your API credentials.")
                return []
            elif response.status_code == 403:
                self.logger.error("API request failed: 403 Forbidden. You don't have permission to access this resource.")
                return []
            elif response.status_code == 404:
                self.logger.error("API request failed: 404 Not Found. The API endpoint may be incorrect.")
                return []
            elif response.status_code == 429:
                self.logger.error("API request failed: 429 Too Many Requests. Rate limit exceeded.")
                return []
            elif not response.ok:
                self.logger.error(f"API request failed: {response.status_code} {response.reason}")
                return []
            
            response.raise_for_status()
            
            data = response.json()
            
            # Get the actual count of records
            if isinstance(data, list):
                record_count = len(data)
            elif isinstance(data, dict):
                record_count = len(data.get('list', data.get('data', data.get('visits', data.get('results', [])))))
            else:
                record_count = 0
            
            self.logger.info(f"API response received: {record_count} records")
            
            # Debug: Log response structure if no records found
            if record_count == 0:
                self.logger.warning("No records found in API response. Response structure:")
                self.logger.warning(f"Response type: {type(data)}")
                if isinstance(data, dict):
                    self.logger.warning(f"Response keys: {list(data.keys())}")
                    for key, value in data.items():
                        if isinstance(value, (list, dict)):
                            self.logger.warning(f"  {key}: {type(value)} with {len(value) if hasattr(value, '__len__') else 'unknown'} items")
                        else:
                            self.logger.warning(f"  {key}: {value}")
                else:
                    self.logger.warning(f"Response content: {str(data)[:500]}...")
            
            # Transform API data to match expected format
            visits = []
            if isinstance(data, list):
                # If API returns a list directly
                raw_visits = data
            elif isinstance(data, dict):
                # If API returns an object with data array
                raw_visits = data.get('list', data.get('data', data.get('visits', data.get('results', []))))
            else:
                self.logger.error(f"Unexpected API response format: {type(data)}")
                return []
            
            # Transform each visit record
            for visit in raw_visits:
                # Extract image URL from new API structure
                image_url = None
                
                # Try different possible locations for image URL in new API
                if 'faceResponse' in visit and visit['faceResponse']:
                    face_response = visit['faceResponse']
                    if isinstance(face_response, dict):
                        # Try different nested structures
                        image_url = (face_response.get('boxData', {}).get('imageUrl') or
                                   face_response.get('faceResponse', {}).get('imageUrl') or
                                   face_response.get('imageUrl') or
                                   face_response.get('image'))
                elif 'imageUrl' in visit:
                    image_url = visit['imageUrl']
                elif 'image' in visit:
                    image_url = visit['image']
                elif 'faceImage' in visit:
                    image_url = visit['faceImage']
                elif 'face_image' in visit:
                    image_url = visit['face_image']
                elif 'photo' in visit:
                    image_url = visit['photo']
                elif 'photoUrl' in visit:
                    image_url = visit['photoUrl']
                
                # Map API fields to expected format
                transformed_visit = {
                    'visit_id': visit.get('id', visit.get('visitId', visit.get('visit_id'))),
                    'customer_id': visit.get('customerId', visit.get('customer_id', visit.get('customerId'))),
                    'image': image_url,
                    'entry_time': visit.get('timestamp', visit.get('entryTime', visit.get('entry_time'))),
                    'event': 'entry' if visit.get('isEntry', False) else 'exit',
                    'camera': visit.get('camera', visit.get('cameraName', 'Unknown')),
                    'branch_id': visit.get('branchId', visit.get('branch_id', 'Unknown')),
                    'age': visit.get('faceResponse', {}).get('age', {}).get('low') if visit.get('faceResponse') else None,
                    'gender': visit.get('faceResponse', {}).get('gender', {}).get('value') if visit.get('faceResponse') else None,
                    'similarity': visit.get('confidence', visit.get('similarity', 1.0))
                }
                
                # Only include visits with valid image URLs
                if transformed_visit['image'] and transformed_visit['image'].startswith('http'):
                    visits.append(transformed_visit)
            
            self.logger.info(f"Transformed {len(visits)} visits with valid image URLs")
            return visits
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON response from API: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error fetching visit data from API: {e}")
            return []
    
    def compare_face_images(self, image1_url: str, image2_url: str) -> Dict:
        """
        Compare two face images and return comparison result
        
        Args:
            image1_url: URL of first image
            image2_url: URL of second image
            
        Returns:
            Dictionary with comparison results
        """
        try:
            self.logger.info(f"Comparing faces: {image1_url} vs {image2_url}")
            
            # Download both images
            img1 = self.download_image_from_url(image1_url)
            img2 = self.download_image_from_url(image2_url)
            
            if img1 is None or img2 is None:
                return {
                    'same_person': False,
                    'confidence': 0.0,
                    'error': 'Could not download one or both images',
                    'image1_url': image1_url,
                    'image2_url': image2_url
                }
            
            # Workaround for InsightFace library issue - detect faces directly
            try:
                # Convert BGR to RGB for InsightFace
                img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                
                # Detect faces in both images
                faces1 = self.app.get(img1_rgb)
                faces2 = self.app.get(img2_rgb)
                
                if len(faces1) == 0 or len(faces2) == 0:
                    return {
                        'same_person': False,
                        'confidence': 0.0,
                        'error': 'Could not detect faces in one or both images',
                        'image1_url': image1_url,
                        'image2_url': image2_url
                    }
                
                # Get face embeddings directly from detected faces
                face1_embedding = faces1[0].embedding
                face2_embedding = faces2[0].embedding
                
                # Calculate similarity
                similarity = self.calculate_face_similarity(face1_embedding, face2_embedding)
                
                # Determine if same person based on threshold
                same_person = similarity > self.similarity_thresh
                
                self.logger.info(f"Face comparison result: {same_person} (confidence: {similarity:.4f}, threshold: {self.similarity_thresh})")
                
                return {
                    'same_person': same_person,
                    'confidence': float(similarity),
                    'threshold_used': self.similarity_thresh,
                    'image1_url': image1_url,
                    'image2_url': image2_url,
                    'error': None
                }
                
            except Exception as face_error:
                self.logger.error(f"Error in face detection/embedding: {face_error}")
                return {
                    'same_person': False,
                    'confidence': 0.0,
                    'error': f'Face detection error: {str(face_error)}',
                    'image1_url': image1_url,
                    'image2_url': image2_url
                }
            
        except Exception as e:
            self.logger.error(f"Error comparing faces: {e}")
            return {
                'same_person': False,
                'confidence': 0.0,
                'error': str(e),
                'image1_url': image1_url,
                'image2_url': image2_url
            }
    
    def calculate_face_similarity(self, face1, face2) -> float:
        """
        Calculate similarity between two face embeddings
        
        Args:
            face1: First face embedding
            face2: Second face embedding
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            # Calculate cosine similarity
            similarity = np.dot(face1, face2) / (np.linalg.norm(face1) * np.linalg.norm(face2))
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Error calculating face similarity: {e}")
            return 0.0
    
    def extract_face_embedding(self, image) -> np.ndarray:
        """
        Extract face embedding from image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Face embedding vector or None if no face detected
        """
        try:
            # Ensure image is in the correct format for InsightFace
            if not isinstance(image, np.ndarray):
                self.logger.error(f"Invalid image format: {type(image)}")
                return None
            
            # Convert BGR to RGB if needed (InsightFace expects RGB)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Detect faces in the image
            faces = self.app.get(image_rgb)
            
            if len(faces) == 0:
                self.logger.warning("No faces detected in image")
                return None
            
            # Get the first (largest) face
            face = faces[0]
            
            # Return the embedding
            return face.embedding
            
        except Exception as e:
            self.logger.error(f"Error extracting face embedding: {e}")
            return None

    def process_face_comparisons(self, comparison_records: List[Dict], max_comparisons: int = None) -> Dict:
        """
        Process face comparisons from API data
        
        Args:
            comparison_records: List of comparison records from API
            max_comparisons: Maximum number of comparisons to process
            
        Returns:
            Dictionary with processing results
        """
        try:
            if not comparison_records:
                return {
                    'total_comparisons': 0,
                    'processed': 0,
                    'same_person': 0,
                    'different_person': 0,
                    'errors': 0,
                    'results': []
                }
            
            # Limit comparisons if specified
            if max_comparisons and len(comparison_records) > max_comparisons:
                comparison_records = comparison_records[:max_comparisons]
            
            self.logger.info(f"Processing {len(comparison_records)} face comparisons")
            
            results = []
            same_person_count = 0
            different_person_count = 0
            error_count = 0
            
            for i, record in enumerate(comparison_records):
                try:
                    self.logger.info(f"Processing comparison {i+1}/{len(comparison_records)}: {record['comparison_id']}")
                    
                    # Compare the two faces
                    comparison_result = self.compare_face_images(
                        record['image1_url'], 
                        record['image2_url']
                    )
                    
                    # Add metadata to result
                    result = {
                        'comparison_id': record['comparison_id'],
                        'event_id': record['event_id'],
                        'branch_id': record['branch_id'],
                        'created_at': record['created_at'],
                        'customer_info': record['customer_info'],
                        'matched_info': record['matched_info'],
                        'api_approve': record['approve'],
                        'our_result': comparison_result['same_person'],
                        'confidence': comparison_result['confidence'],
                        'threshold_used': comparison_result.get('threshold_used', self.similarity_thresh),
                        'image1_url': comparison_result['image1_url'],
                        'image2_url': comparison_result['image2_url'],
                        'error': comparison_result['error'],
                        'match_status': 'SAME' if comparison_result['same_person'] else 'DIFFERENT',
                        'api_vs_our_match': record['approve'] == comparison_result['same_person'],
                        'raw_data': record.get('raw_data', {})
                    }
                    
                    results.append(result)
                    
                    # Update counters
                    if comparison_result['error']:
                        error_count += 1
                    elif comparison_result['same_person']:
                        same_person_count += 1
                    else:
                        different_person_count += 1
                    
                    self.logger.info(f"Comparison {i+1} result: {result['match_status']} (confidence: {comparison_result['confidence']:.4f})")
                    
                except Exception as e:
                    self.logger.error(f"Error processing comparison {record['comparison_id']}: {e}")
                    error_count += 1
                    results.append({
                        'comparison_id': record['comparison_id'],
                        'error': str(e),
                        'match_status': 'ERROR'
                    })
            
            # Calculate accuracy if we have API approval data
            api_matches = sum(1 for r in results if r.get('api_vs_our_match') is True)
            total_with_api_data = sum(1 for r in results if 'api_vs_our_match' in r and r['api_vs_our_match'] is not None)
            accuracy = (api_matches / total_with_api_data * 100) if total_with_api_data > 0 else 0
            
            processing_results = {
                'total_comparisons': len(comparison_records),
                'processed': len(results),
                'same_person': same_person_count,
                'different_person': different_person_count,
                'errors': error_count,
                'accuracy_vs_api': accuracy,
                'api_matches': api_matches,
                'total_with_api_data': total_with_api_data,
                'results': results
            }
            
            self.logger.info(f"Face comparison processing completed:")
            self.logger.info(f"  Total: {processing_results['total_comparisons']}")
            self.logger.info(f"  Same Person: {processing_results['same_person']}")
            self.logger.info(f"  Different Person: {processing_results['different_person']}")
            self.logger.info(f"  Errors: {processing_results['errors']}")
            self.logger.info(f"  Accuracy vs API: {processing_results['accuracy_vs_api']:.2f}%")
            
            return processing_results
            
        except Exception as e:
            self.logger.error(f"Error processing face comparisons: {e}")
            return {
                'total_comparisons': 0,
                'processed': 0,
                'same_person': 0,
                'different_person': 0,
                'errors': 1,
                'error_message': str(e),
                'results': []
            }

    def assess_face_quality(self, face) -> Dict[str, float]:
        """
        Assess face quality to prevent false positives and mismatches
        
        Args:
            face: InsightFace face object
            
        Returns:
            Dictionary with quality scores
        """
        quality_scores = {
            'overall': 0.0,
            'blur': 0.0,
            'pose': 0.0,
            'lighting': 0.0,
            'size': 0.0
        }
        
        try:
            # Detection confidence
            det_score = getattr(face, 'det_score', 0.0)
            
            # Face size assessment
            bbox = face.bbox
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            face_area = face_width * face_height
            
            # Size score (prefer larger faces)
            size_normalization = self.config['face_quality']['size_normalization']
            size_score = min(1.0, face_area / size_normalization)
            
            # Blur assessment (simplified)
            blur_score = min(1.0, det_score * 1.2)  # Higher detection score = less blur
            
            # Pose assessment (using keypoints if available)
            pose_score = 1.0
            if hasattr(face, 'kps') and face.kps is not None:
                kps = face.kps
                # Check if keypoints are well distributed
                if len(kps) >= 5:
                    # Simple pose assessment based on keypoint distribution
                    x_coords = kps[:, 0]
                    y_coords = kps[:, 1]
                    x_range = np.max(x_coords) - np.min(x_coords)
                    y_range = np.max(y_coords) - np.min(y_coords)
                    pose_score = min(1.0, (x_range + y_range) / 100)
            
            # Lighting assessment (simplified)
            lighting_score = min(1.0, det_score * 1.1)
            
            # Overall quality score using config weights
            weights = self.config['face_quality']['weights']
            overall_score = (det_score * weights['detection_score'] + 
                           size_score * weights['size_score'] + 
                           blur_score * weights['blur_score'] + 
                           pose_score * weights['pose_score'] + 
                           lighting_score * weights['lighting_score'])
            
            quality_scores = {
                'overall': float(overall_score),
                'blur': float(blur_score),
                'pose': float(pose_score),
                'lighting': float(lighting_score),
                'size': float(size_score)
            }
            
        except Exception as e:
            self.logger.warning(f"Error assessing face quality: {e}")
            quality_scores['overall'] = self.config['face_quality']['min_overall_score']
        
        return quality_scores
    
    def get_face_pose_angles(self, face) -> Dict[str, float]:
        """
        Extract face pose angles from InsightFace detection
        
        Args:
            face: InsightFace face object
            
        Returns:
            Dict with yaw, pitch, roll angles in degrees
        """
        try:
            # InsightFace provides pose information in radians
            yaw = getattr(face, 'yaw', 0)  # Left/right rotation
            pitch = getattr(face, 'pitch', 0)  # Up/down rotation
            roll = getattr(face, 'roll', 0)  # Rotation around face axis
            
            # Convert from radians to degrees
            yaw_deg = math.degrees(yaw) if yaw else 0
            pitch_deg = math.degrees(pitch) if pitch else 0
            roll_deg = math.degrees(roll) if roll else 0
            
            return {
                'yaw': yaw_deg,      # -90 to +90 (left to right)
                'pitch': pitch_deg,  # -90 to +90 (up to down)  
                'roll': roll_deg     # -180 to +180 (rotation)
            }
        except Exception as e:
            self.logger.warning(f"Error extracting pose angles: {e}")
            return {'yaw': 0, 'pitch': 0, 'roll': 0}
    
    def is_side_face(self, face) -> bool:
        """
        Check if face is too side-facing to be useful for recognition
        Uses both pose angles and advanced bbox analysis
        
        Args:
            face: InsightFace face object
            
        Returns:
            True if face should be rejected (side-facing)
        """
        try:
            # Method 1: Try pose angles if available
            pose_angles = self.get_face_pose_angles(face)
            yaw_angle = abs(pose_angles.get('yaw', 0))
            pitch_angle = abs(pose_angles.get('pitch', 0))
            
            if yaw_angle > 0 or pitch_angle > 0:  # If pose data is available
                yaw_threshold = self.config['face_detection']['yaw_threshold']
                pitch_threshold = self.config['face_detection']['pitch_threshold']
                if yaw_angle > yaw_threshold:
                    self.logger.info(f"Side face detected: yaw={yaw_angle:.1f}° (threshold: {yaw_threshold}°)")
                    return True
                if pitch_angle > pitch_threshold:
                    self.logger.info(f"Extreme angle detected: pitch={pitch_angle:.1f}° (threshold: {pitch_threshold}°)")
                    return True
                return False
            
            # Method 2: Advanced bbox analysis using InsightFace bbox
            bbox = getattr(face, 'bbox', None)
            det_score = getattr(face, 'det_score', 0.0)
            
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                bbox_data = {
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'top': y1,
                    'left': x1
                }
                
                is_side, reason, score = self.analyze_bbox_for_side_face(bbox_data, det_score)
                if is_side:
                    self.logger.info(f"Side face detected: {reason} (score: {score})")
                    return True
                
            return False
        except Exception as e:
            self.logger.warning(f"Error checking side face: {e}")
            return False
    
    def analyze_bbox_for_side_face(self, bbox_data, det_score=None):
        """
        Analyze bbox data from JSON to detect side faces using advanced methods
        Based on research on side face detection techniques
        
        Args:
            bbox_data: Dictionary with 'width', 'height', 'top', 'left' keys
            det_score: Detection confidence score (optional)
            
        Returns:
            Tuple of (is_side_face, reason, side_face_score)
        """
        if not bbox_data:
            return False, "No bbox data", 0
        
        width = bbox_data.get('width', 0)
        height = bbox_data.get('height', 0)
        top = bbox_data.get('top', 0)
        left = bbox_data.get('left', 0)
        
        if width <= 0 or height <= 0:
            return False, "Invalid bbox dimensions", 0
        
        # Calculate metrics
        aspect_ratio = width / height
        area = width * height
        perimeter = 2 * (width + height)
        compactness = (4 * 3.14159 * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Advanced scoring system based on research
        side_face_score = 0
        reasons = []
        
        # 1. Aspect ratio analysis using config thresholds
        aspect_thresholds = self.config['side_face_detection']['aspect_ratio_thresholds']
        if aspect_ratio < aspect_thresholds['extreme_profile']:
            side_face_score += 4
            reasons.append(f"Extreme profile (ratio: {aspect_ratio:.2f})")
        elif aspect_ratio < aspect_thresholds['very_strong_profile']:
            side_face_score += 3
            reasons.append(f"Very strong profile (ratio: {aspect_ratio:.2f})")
        elif aspect_ratio < aspect_thresholds['strong_profile']:
            side_face_score += 2
            reasons.append(f"Strong profile (ratio: {aspect_ratio:.2f})")
        elif aspect_ratio > aspect_thresholds['very_wide']:
            side_face_score += 3
            reasons.append(f"Very wide face (ratio: {aspect_ratio:.2f})")
        elif aspect_ratio > aspect_thresholds['wide']:
            side_face_score += 2
            reasons.append(f"Wide face (ratio: {aspect_ratio:.2f})")
        elif aspect_ratio > aspect_thresholds['moderately_wide']:
            side_face_score += 1
            reasons.append(f"Moderately wide (ratio: {aspect_ratio:.2f})")
        
        # 2. Size analysis using config thresholds
        area_thresholds = self.config['side_face_detection']['area_thresholds']
        if area < area_thresholds['extremely_small']:
            side_face_score += 3
            reasons.append(f"Extremely small area: {area}")
        elif area < area_thresholds['very_small']:
            side_face_score += 2
            reasons.append(f"Very small area: {area}")
        elif area < area_thresholds['small']:
            side_face_score += 1
            reasons.append(f"Small area: {area}")
        elif area > area_thresholds['very_large']:
            side_face_score += 2
            reasons.append(f"Very large area: {area}")
        elif area > area_thresholds['large']:
            side_face_score += 1
            reasons.append(f"Large area: {area}")
        
        # 3. Compactness analysis using config thresholds
        compactness_thresholds = self.config['side_face_detection']['compactness_thresholds']
        if compactness < compactness_thresholds['very_low']:
            side_face_score += 2
            reasons.append(f"Very low compactness: {compactness:.2f}")
        elif compactness < compactness_thresholds['low']:
            side_face_score += 1
            reasons.append(f"Low compactness: {compactness:.2f}")
        
        # 4. Detection confidence analysis using config thresholds
        confidence_thresholds = self.config['side_face_detection']['confidence_thresholds']
        if det_score and det_score < confidence_thresholds['very_low']:
            side_face_score += 2
            reasons.append(f"Very low confidence: {det_score:.3f}")
        elif det_score and det_score < confidence_thresholds['low']:
            side_face_score += 1
            reasons.append(f"Low confidence: {det_score:.3f}")
        
        # 5. Position analysis (faces at edges might be side views)
        edge_threshold = self.config['side_face_detection']['edge_position_threshold']
        if left < edge_threshold or top < edge_threshold:
            side_face_score += 1
            reasons.append(f"Face very near edge (left: {left}, top: {top})")
        
        # Decision threshold using config
        decision_threshold = self.config['side_face_detection']['decision_threshold']
        is_side_face = side_face_score >= decision_threshold
        reason = "; ".join(reasons) if reasons else "Normal face"
        
        return is_side_face, reason, side_face_score
    
    def check_side_face_from_json_bbox(self, visit_data) -> tuple:
        """
        Check for side faces using bbox data from JSON before processing
        
        Args:
            visit_data: Visit data dictionary from JSON
            
        Returns:
            Tuple of (is_side_face, reason, bbox_data)
        """
        try:
            # Extract bbox data from entryEventIds
            entry_events = visit_data.get('entryEventIds', [])
            if not entry_events:
                return False, "No entry events", None
            
            # Use the first entry event's bbox
            first_event = entry_events[0]
            bbox_data = first_event.get('box', {})
            
            if not bbox_data:
                return False, "No bbox data in entry event", None
            
            # Analyze the bbox for side face characteristics
            is_side, reason, score = self.analyze_bbox_for_side_face(bbox_data)
            
            return is_side, reason, bbox_data
            
        except Exception as e:
            self.logger.warning(f"Error checking side face from JSON bbox: {e}")
            return False, f"Error: {e}", None
    
    def extract_face_embedding(self, image_source: str, save_image: bool = False, output_dir: str = None) -> Optional[Dict]:
        """
        Extract face embedding from image with quality assessment
        
        Args:
            image_source: Path to local image file or URL to image
            save_image: Whether to save downloaded images locally
            output_dir: Directory to save images (if save_image is True)
            
        Returns:
            Dictionary with embedding and quality info, or None if no face found
        """
        try:
            # Determine if it's a URL or local file path
            if image_source.startswith('http'):
                # Generate save path if needed
                save_path = None
                if save_image and output_dir:
                    # Extract filename from URL
                    url_parts = image_source.split('/')
                    filename = url_parts[-1] if url_parts else f"image_{int(time.time())}.jpg"
                    # Ensure it has proper extension
                    if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                        filename += '.jpg'
                    save_path = os.path.join(output_dir, filename)
                
                # Download image from URL
                image = self.download_image_from_url(image_source, save_path=save_path)
                if image is None:
                    self.logger.warning(f"Could not download image from URL: {image_source}")
                    return None
            else:
                # Load local image file
                image = cv2.imread(image_source)
                if image is None:
                    self.logger.warning(f"Could not load local image: {image_source}")
                    return None
            
            # Detect faces
            faces = self.app.get(image)
            if not faces:
                self.logger.warning(f"No faces detected in: {image_source}")
                return None
            
            # Select best face
            best_face = max(faces, key=lambda f: getattr(f, 'det_score', 0.0))
            
            # Check confidence threshold
            if getattr(best_face, 'det_score', 0.0) < self.confidence_thresh:
                self.logger.warning(f"Face confidence too low in: {image_source}")
                return None
            
            # Check if face is side-facing (reject side faces)
            if self.is_side_face(best_face):
                self.logger.warning(f"Side face rejected in: {image_source}")
                return None
            
            # Get embedding
            embedding = getattr(best_face, 'normed_embedding', None)
            if embedding is None:
                embedding = getattr(best_face, 'embedding', None)
                if embedding is not None:
                    embedding = embedding / np.linalg.norm(embedding)
            
            if embedding is None:
                self.logger.warning(f"Could not extract embedding from: {image_source}")
                return None
            
            # Assess face quality
            quality_scores = self.assess_face_quality(best_face)
            
            # Check quality threshold - only reject extremely poor quality faces
            min_quality = self.config['face_detection']['min_quality_threshold']
            if quality_scores['overall'] < min_quality:
                self.logger.warning(f"Face quality extremely low in: {image_source}")
                return None
            
            result = {
                'embedding': embedding.astype(np.float32),
                'quality': quality_scores,
                'bbox': best_face.bbox,
                'det_score': getattr(best_face, 'det_score', 0.0),
                'face_confidence': getattr(best_face, 'det_score', 0.0),  # Add face_confidence for easier access
                'face_hash': self.compute_face_hash(embedding),
                'image_source': image_source
            }
            
            # Add saved image path if image was saved
            if save_image and save_path:
                result['saved_image_path'] = save_path
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {image_source}: {e}")
            return None
    
    def add_person(self, name: str, image_source: str, embedding_data: Dict) -> int:
        """
        Add a new person to the database
        
        Args:
            name: Person name
            image_source: Path to person's image or URL
            embedding_data: Face embedding data
            
        Returns:
            Person ID
        """
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            # Check for duplicate face hash
            cursor.execute('SELECT id FROM persons WHERE face_hash = ?', 
                         (embedding_data['face_hash'],))
            if cursor.fetchone():
                self.logger.warning(f"Duplicate face detected for: {name}")
                return -1
            
            # Store person data (without embedding BLOB - now stored in Qdrant)
            quality_score = embedding_data['quality']['overall']
            
            cursor.execute('''
                INSERT INTO persons (name, image_path, face_quality, face_hash)
                VALUES (?, ?, ?, ?)
            ''', (name, image_source, quality_score, embedding_data['face_hash']))
            
            person_id = cursor.lastrowid
            
            # Store detailed quality scores
            cursor.execute('''
                INSERT INTO face_quality (person_id, quality_score, blur_score, pose_score, lighting_score)
                VALUES (?, ?, ?, ?, ?)
            ''', (person_id, quality_score, 
                  embedding_data['quality']['blur'],
                  embedding_data['quality']['pose'],
                  embedding_data['quality']['lighting']))
            
            conn.commit()
            
            # Store embedding in Qdrant
            success = self.vector_db.add_embedding(
                person_id=person_id,
                embedding=embedding_data['embedding'],
                metadata={
                    'name': name,
                    'quality': quality_score,
                    'image_path': image_source,
                    'face_hash': embedding_data['face_hash']
                }
            )
            
            if success:
                self.logger.info(f"Added person: {name} (ID: {person_id}) to both SQLite and Qdrant")
            else:
                self.logger.error(f"Failed to add embedding to Qdrant for person {person_id}")
                # Rollback SQLite transaction
                conn.rollback()
                return -1
            
            return person_id
            
        except Exception as e:
            self.logger.error(f"Error adding person {name}: {e}")
            conn.rollback()
            return -1
        finally:
            conn.close()
    
    def load_embeddings(self):
        """Initialize vector database connection (embeddings are stored in Qdrant)"""
        try:
            # Check if Qdrant is working
            embedding_count = self.vector_db.get_embedding_count()
            self.logger.info(f"Vector database initialized with {embedding_count} embeddings")
            
            # Note: We no longer load all embeddings into memory
            # Qdrant handles the storage and retrieval efficiently
            self.logger.info("Using Qdrant for vector storage and similarity search")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector database: {e}")
            raise
    
    def search_person(self, query_embedding: np.ndarray, k=5) -> List[Dict]:
        """
        Search for similar persons using Qdrant vector database
        
        Args:
            query_embedding: Query face embedding
            k: Number of top results to return
            
        Returns:
            List of similar persons with scores
        """
        try:
            # Use Qdrant for similarity search
            results = self.vector_db.search_similar(
                query_embedding=query_embedding,
                k=k,
                threshold=self.similarity_thresh
            )
            
            self.logger.debug(f"Found {len(results)} similar persons")
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching for similar persons: {e}")
            return []
    
    def update_person_stats(self, person_id: int):
        """Update person statistics (last seen, match count)"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE persons 
            SET last_seen = CURRENT_TIMESTAMP, match_count = match_count + 1
            WHERE id = ?
        ''', (person_id,))
        conn.commit()
        conn.close()
    
    def store_visit_info(self, person_id: int, visit_id: str, customer_id: str, 
                        entry_time: str, image_url: str, saved_image_path: str, similarity: float):
        """Store visit information in database"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO person_visits 
                (person_id, visit_id, customer_id, entry_time, image_url, saved_image_path, similarity)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (person_id, visit_id, customer_id, entry_time, image_url, saved_image_path, similarity))
            
            conn.commit()
        except Exception as e:
            self.logger.error(f"Error storing visit info: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def store_low_similarity_image(self, visit_id: str, customer_id: str, entry_time: str, 
                                 image_url: str, saved_image_path: str, similarity: float, 
                                 best_match_name: str = None, reason: str = None):
        """Store low similarity images in a separate table"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            # Create low similarity table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS low_similarity_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    visit_id TEXT,
                    customer_id TEXT,
                    entry_time TEXT,
                    image_url TEXT,
                    saved_image_path TEXT,
                    similarity REAL,
                    best_match_name TEXT,
                    reason TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add reason column if it doesn't exist (for existing databases)
            try:
                cursor.execute('ALTER TABLE low_similarity_images ADD COLUMN reason TEXT')
            except sqlite3.OperationalError:
                # Column already exists, ignore error
                pass
            
            cursor.execute('''
                INSERT INTO low_similarity_images 
                (visit_id, customer_id, entry_time, image_url, saved_image_path, similarity, best_match_name, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (visit_id, customer_id, entry_time, image_url, saved_image_path, similarity, best_match_name, reason))
            
            conn.commit()
        except Exception as e:
            self.logger.error(f"Error storing low similarity image: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def process_visit_data(self, json_file_path: str, output_folder: str = None, max_visits: int = None, save_images: bool = True):
        """
        Process visit data from JSON file with image URLs - main function
        
        Args:
            json_file_path: Path to JSON file containing visit data
            output_folder: Folder to organize recognized persons (optional)
            max_visits: Maximum number of visits to process (for testing)
            save_images: Whether to save downloaded images locally
        """
        self.logger.info(f"Processing visit data from: {json_file_path}")
        
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            # Create images subfolder if saving images
            if save_images:
                images_folder = os.path.join(output_folder, "downloaded_images")
                os.makedirs(images_folder, exist_ok=True)
        
        # Load visit data
        visits = self.load_visit_data(json_file_path)
        if not visits:
            self.logger.warning("No valid visits found in JSON file")
            return
        
        # Limit visits for testing if specified
        if max_visits and max_visits < len(visits):
            visits = visits[:max_visits]
            self.logger.info(f"Processing first {max_visits} visits for testing")
        
        self.logger.info(f"Processing {len(visits)} visits")
        
        # Process each visit
        results = {
            'processed': 0,
            'recognized': 0,
            'new_persons': 0,
            'no_faces': 0,
            'low_quality': 0,
            'download_failed': 0,
            'duplicate_faces': 0,
            'low_similarity': 0
        }
        
        # Thread-safe aggregation for batch webhook
        results_lock = threading.Lock()
        batch_groups = []  # Will store all person groups for final webhook
        
        def process_single_visit(visit_data):
            """Process a single visit - thread-safe function"""
            i, visit = visit_data
            visit_results = {
                'processed': 0,
                'recognized': 0,
                'new_persons': 0,
                'no_faces': 0,
                'low_quality': 0,
                'download_failed': 0,
                'duplicate_faces': 0,
                'low_similarity': 0
            }
            person_group = None
            
            try:
                visit_id = visit.get('id', f'visit_{i}')
                image_url = visit.get('image')
                customer_id = visit.get('customerId', f'customer_{i}')
                entry_time = visit.get('entryTime', '')
                
                self.logger.info(f"Processing visit {visit_id} ({i+1}/{len(visits)})")
                self.logger.info(f"Customer: {customer_id}, Time: {entry_time}")
                
                # Extract face embedding from image URL
                images_dir = os.path.join(output_folder, "downloaded_images") if (output_folder and save_images) else None
                embedding_data = self.extract_face_embedding(image_url, save_image=save_images, output_dir=images_dir)
                if embedding_data is None:
                    # Store in low similarity table as unusable image
                    self.store_low_similarity_image(visit_id, customer_id, entry_time, 
                                                  image_url, None, 0.0, "No face detected, low confidence, or side face")
                    visit_results['no_faces'] += 1
                    return visit_results, person_group
            except Exception as e:
                self.logger.error(f"Error processing visit {i}: {e}")
                visit_results['no_faces'] += 1
                return visit_results, person_group
            
            # Check for duplicates
            try:
                if self.is_duplicate_image(image_url, embedding_data['embedding']):
                    self.logger.info(f"Skipping duplicate image: {image_url}")
                    visit_results['duplicate_faces'] += 1
                    return visit_results, person_group
            except Exception as e:
                self.logger.error(f"Error checking duplicates for {image_url}: {e}")
                # Continue processing even if duplicate check fails
            
            visit_results['processed'] += 1
            
            # If no persons in DB, add the first image as a new person
            if self.vector_db.get_embedding_count() == 0:
                person_name = f"Person_{customer_id}_{int(time.time())}"
                person_id = self.add_person(person_name, image_url, embedding_data)
                if person_id > 0:
                    self.logger.info(f"First person added: {person_name} (ID: {person_id})")
                    self.store_visit_info(person_id, visit_id, customer_id, entry_time, image_url, embedding_data.get('saved_image_path'), 1.0)
                    
                    # Create person group for batch webhook
                    person_group = {
                        'person_id': person_id,
                        'person_name': person_name,
                        'visits': [{
                            'visit_id': visit_id,
                            'customer_id': customer_id,
                            'customerId': visit.get('customerId', customer_id),
                            'image_url': image_url,
                            'image': visit.get('image', image_url),
                            'entry_time': entry_time,
                            'entryTime': visit.get('entryTime', entry_time),
                            'similarity': 1.0,
                            'branchId': visit.get('branchId', ''),
                            'camera': visit.get('camera', ''),
                            'entryEventIds': visit.get('entryEventIds', []),
                            'customer': visit.get('customer', {}),
                            'results': visit.get('results', {})
                        }]
                    }
                    
                    visit_results['new_persons'] += 1
                else:
                    visit_results['duplicate_faces'] += 1
                return visit_results, person_group

            # Otherwise, do similarity search as usual
            search_results = self.search_person(embedding_data['embedding'], k=5)  # Get more results
            similarity = search_results[0]['similarity'] if search_results else 0.0
            best_match = search_results[0] if search_results else None
            
            # Use a more strict threshold for grouping
            grouping_threshold = self.config['face_recognition']['grouping_threshold_file']
            
            if search_results and similarity >= grouping_threshold:
                # Person recognized (group with best match)
                person_id = best_match['person_id']
                person_name = best_match['name']
                
                self.logger.info(f"Recognized: {person_name} (similarity: {similarity:.3f})")
                
                # Update statistics
                self.update_person_stats(person_id)
                
                # Store visit information in database
                self.store_visit_info(person_id, visit_id, customer_id, entry_time, 
                                    image_url, embedding_data.get('saved_image_path'), similarity)
                
                # Create person group for batch webhook
                person_group = {
                    'person_id': person_id,
                    'person_name': person_name,
                    'visits': [{
                        'visit_id': visit_id,
                        'customer_id': customer_id,
                        'customerId': visit.get('customerId', customer_id),
                        'image_url': image_url,
                        'image': visit.get('image', image_url),
                        'entry_time': entry_time,
                        'entryTime': visit.get('entryTime', entry_time),
                        'similarity': similarity,
                        'branchId': visit.get('branchId', ''),
                        'camera': visit.get('camera', ''),
                        'entryEventIds': visit.get('entryEventIds', []),
                        'customer': visit.get('customer', {}),
                        'results': visit.get('results', {})
                    }]
                }
                
                # Save visit info if output folder specified
                if output_folder:
                    person_folder = os.path.join(output_folder, f"{person_name}_{person_id}")
                    os.makedirs(person_folder, exist_ok=True)
                    
                    # Save visit metadata
                    visit_info = {
                        'visit_id': visit_id,
                        'customer_id': customer_id,
                        'entry_time': entry_time,
                        'image_url': image_url,
                        'saved_image_path': embedding_data.get('saved_image_path'),
                        'similarity': similarity,
                        'processed_at': datetime.now().isoformat()
                    }
                    
                    visit_file = os.path.join(person_folder, f"visit_{visit_id}.json")
                    with open(visit_file, 'w') as f:
                        json.dump(visit_info, f, indent=2)
                
                visit_results['recognized'] += 1
            else:
                # Not recognized (low similarity) or no match - create new person
                self.logger.info(f"Low similarity: {similarity:.3f} (grouping threshold: {grouping_threshold:.3f}) - creating new person.")
                
                person_name = f"Person_{customer_id}_{int(time.time())}"
                person_id = self.add_person(person_name, image_url, embedding_data)
                if person_id > 0:
                    self.logger.info(f"New person added: {person_name} (ID: {person_id})")
                    self.store_visit_info(person_id, visit_id, customer_id, entry_time, 
                                        image_url, embedding_data.get('saved_image_path'), similarity)
                    
                    # Create person group for batch webhook
                    person_group = {
                        'person_id': person_id,
                        'person_name': person_name,
                        'visits': [{
                            'visit_id': visit_id,
                            'customer_id': customer_id,
                            'image_url': image_url,
                            'entry_time': entry_time,
                            'similarity': similarity,
                            'branchId': visit.get('branchId', ''),
                            'camera': visit.get('camera', ''),
                            'entryEventIds': visit.get('entryEventIds', []),
                            'customer': visit.get('customer', {}),
                            'results': visit.get('results', {})
                        }]
                    }
                    
                    visit_results['new_persons'] += 1
                else:
                    self.logger.warning(f"Failed to add new person: {person_name}")
                    visit_results['duplicate_faces'] += 1
            
            return visit_results, person_group
        
        # Process visits using multi-threading
        max_workers = min(self.config['image_processing']['max_workers'], len(visits))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all visits for processing
            future_to_visit = {executor.submit(process_single_visit, (i, visit)): (i, visit) for i, visit in enumerate(visits)}
            
            # Collect results as they complete
            for future in as_completed(future_to_visit):
                try:
                    visit_results, person_group = future.result()
                    
                    # Thread-safe aggregation of results
                    with results_lock:
                        for key in results:
                            results[key] += visit_results[key]
                        
                        # Add person group to batch webhook data
                        if person_group:
                            batch_groups.append(person_group)
                            
                except Exception as e:
                    i, visit = future_to_visit[future]
                    self.logger.error(f"Error processing visit {i}: {e}")
                    with results_lock:
                        results['no_faces'] += 1
        
        # Print summary
        self.logger.info("Processing completed!")
        self.logger.info(f"Results: {results}")
        
        # Save clustering results to JSON file
        if batch_groups:
            try:
                self.logger.info(f"Saving clustering results with {len(batch_groups)} groups")
                
                # Save clustering results to JSON
                success = save_clustering_results(
                    groups=batch_groups,
                    total_processed=results['processed'],
                    results=results
                )
                
                if success:
                    self.logger.info("Clustering results saved successfully")
                else:
                    self.logger.warning("Failed to save clustering results")
                    
            except Exception as e:
                self.logger.error(f"Error saving clustering results: {e}")
        else:
            self.logger.info("No groups to save in clustering results")
        
        return results
    
    def process_visit_data_from_json(self, json_data: dict, output_folder: str = None, max_visits: int = None, save_images: bool = True, clear_existing: bool = False):
        """
        Process visit data from JSON data (not file) - main function
        
        Args:
            json_data: Dictionary containing visit data
            output_folder: Folder to organize recognized persons (optional)
            max_visits: Maximum number of visits to process (for testing)
            save_images: Whether to save downloaded images locally
            clear_existing: Whether to clear existing data before processing
        """
        self.logger.info(f"Processing visit data from JSON data")
        
        # Clear existing data if requested
        if clear_existing:
            self.logger.info("Clearing existing data before processing new JSON...")
            self.clear_all_data()
        
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            # Create images subfolder if saving images
            if save_images:
                images_folder = os.path.join(output_folder, "downloaded_images")
                os.makedirs(images_folder, exist_ok=True)
        
        # Extract visits from JSON data
        visits = json_data.get('visits', [])
        if not visits:
            self.logger.warning("No visits found in JSON data")
            return {
                'processed': 0,
                'recognized': 0,
                'new_persons': 0,
                'no_faces': 0,
                'low_quality': 0,
                'download_failed': 0,
                'duplicate_faces': 0
            }
        
        # Filter visits with valid image URLs
        valid_visits = []
        for visit in visits:
            if visit.get('image') and visit.get('image').startswith('http'):
                valid_visits.append(visit)
        
        self.logger.info(f"Found {len(valid_visits)} visits with valid image URLs")
        
        # Limit visits for testing if specified
        if max_visits and max_visits < len(valid_visits):
            valid_visits = valid_visits[:max_visits]
            self.logger.info(f"Processing first {max_visits} visits for testing")
        
        self.logger.info(f"Processing {len(valid_visits)} visits")
        
        # Process each visit
        results = {
            'processed': 0,
            'recognized': 0,
            'new_persons': 0,
            'no_faces': 0,
            'low_quality': 0,
            'download_failed': 0,
            'duplicate_faces': 0,
            'low_similarity': 0
        }
        
        # Thread-safe aggregation for batch webhook
        results_lock = threading.Lock()
        batch_groups = []  # Will store all person groups for final webhook
        
        def process_single_visit_json(visit_data):
            """Process a single visit from JSON - thread-safe function"""
            i, visit = visit_data
            visit_results = {
                'processed': 0,
                'recognized': 0,
                'new_persons': 0,
                'no_faces': 0,
                'low_quality': 0,
                'download_failed': 0,
                'duplicate_faces': 0,
                'low_similarity': 0
            }
            person_group = None
            
            visit_id = visit.get('id', f'visit_{i}')
            image_url = visit.get('image')
            customer_id = visit.get('customerId', f'customer_{i}')
            entry_time = visit.get('entryTime', '')
            
            self.logger.info(f"Processing visit {visit_id} ({i+1}/{len(valid_visits)})")
            self.logger.info(f"Customer: {customer_id}, Time: {entry_time}")
            
            # Check for side face using JSON bbox data first
            is_side_face, side_reason, bbox_data = self.check_side_face_from_json_bbox(visit)
            if is_side_face:
                self.logger.info(f"Side face detected from JSON bbox: {side_reason}")
                # Store in low similarity table as unusable image
                self.store_low_similarity_image(visit_id, customer_id, entry_time, 
                                              image_url, None, 0.0, f"Side face: {side_reason}")
                visit_results['no_faces'] += 1
                return visit_results, person_group
            
            # Extract face embedding from image URL
            images_dir = os.path.join(output_folder, "downloaded_images") if (output_folder and save_images) else None
            embedding_data = self.extract_face_embedding(image_url, save_image=save_images, output_dir=images_dir)
            if embedding_data is None:
                # Store in low similarity table as unusable image
                self.store_low_similarity_image(visit_id, customer_id, entry_time, 
                                              image_url, None, 0.0, "No face detected, low confidence, or side face")
                visit_results['no_faces'] += 1
                return visit_results, person_group
            
            # Check for duplicates
            try:
                if self.is_duplicate_image(image_url, embedding_data['embedding']):
                    self.logger.info(f"Skipping duplicate image: {image_url}")
                    visit_results['duplicate_faces'] += 1
                    return visit_results, person_group
            except Exception as e:
                self.logger.error(f"Error checking duplicates for {image_url}: {e}")
                # Continue processing even if duplicate check fails
            
            visit_results['processed'] += 1
            
            # If no persons in DB, add the first image as a new person
            if self.vector_db.get_embedding_count() == 0:
                person_name = f"Person_{customer_id}_{int(time.time())}"
                person_id = self.add_person(person_name, image_url, embedding_data)
                if person_id > 0:
                    self.logger.info(f"First person added: {person_name} (ID: {person_id})")
                    self.store_visit_info(person_id, visit_id, customer_id, entry_time, image_url, embedding_data.get('saved_image_path'), 1.0)
                    
                    # Create person group for batch webhook
                    person_group = {
                        'person_id': person_id,
                        'person_name': person_name,
                        'visits': [{
                            'visit_id': visit_id,
                            'customer_id': customer_id,
                            'customerId': visit.get('customerId', customer_id),
                            'image_url': image_url,
                            'image': visit.get('image', image_url),
                            'entry_time': entry_time,
                            'entryTime': visit.get('entryTime', entry_time),
                            'similarity': 1.0,
                            'branchId': visit.get('branchId', ''),
                            'camera': visit.get('camera', ''),
                            'entryEventIds': visit.get('entryEventIds', []),
                            'customer': visit.get('customer', {}),
                            'results': visit.get('results', {})
                        }]
                    }
                    
                    visit_results['new_persons'] += 1
                else:
                    visit_results['duplicate_faces'] += 1
                return visit_results, person_group

            # Otherwise, do similarity search as usual
            search_results = self.search_person(embedding_data['embedding'], k=5)  # Get more results
            similarity = search_results[0]['similarity'] if search_results else 0.0
            best_match = search_results[0] if search_results else None
            
            # Use a more strict threshold for grouping
            grouping_threshold = self.config['face_recognition']['grouping_threshold_json']
            
            if search_results and similarity >= grouping_threshold:
                # Person recognized (group with best match)
                person_id = best_match['person_id']
                person_name = best_match['name']
                
                self.logger.info(f"Recognized: {person_name} (similarity: {similarity:.3f})")
                
                # Update statistics
                self.update_person_stats(person_id)
                
                # Store visit information in database
                self.store_visit_info(person_id, visit_id, customer_id, entry_time, 
                                    image_url, embedding_data.get('saved_image_path'), similarity)
                
                # Create person group for batch webhook
                person_group = {
                    'person_id': person_id,
                    'person_name': person_name,
                    'visits': [{
                        'visit_id': visit_id,
                        'customer_id': customer_id,
                        'customerId': visit.get('customerId', customer_id),
                        'image_url': image_url,
                        'image': visit.get('image', image_url),
                        'entry_time': entry_time,
                        'entryTime': visit.get('entryTime', entry_time),
                        'similarity': similarity,
                        'branchId': visit.get('branchId', ''),
                        'camera': visit.get('camera', ''),
                        'entryEventIds': visit.get('entryEventIds', []),
                        'customer': visit.get('customer', {}),
                        'results': visit.get('results', {})
                    }]
                }
                
                # Save visit info if output folder specified
                if output_folder:
                    person_folder = os.path.join(output_folder, f"{person_name}_{person_id}")
                    os.makedirs(person_folder, exist_ok=True)
                    
                    # Save visit metadata
                    visit_info = {
                        'visit_id': visit_id,
                        'customer_id': customer_id,
                        'entry_time': entry_time,
                        'image_url': image_url,
                        'saved_image_path': embedding_data.get('saved_image_path'),
                        'similarity': similarity,
                        'processed_at': datetime.now().isoformat()
                    }
                    
                    visit_file = os.path.join(person_folder, f"visit_{visit_id}.json")
                    with open(visit_file, 'w') as f:
                        json.dump(visit_info, f, indent=2)
                
                visit_results['recognized'] += 1
            else:
                # Not recognized (low similarity) or no match - create new person
                self.logger.info(f"Low similarity: {similarity:.3f} (grouping threshold: {grouping_threshold:.3f}) - creating new person.")
                
                person_name = f"Person_{customer_id}_{int(time.time())}"
                person_id = self.add_person(person_name, image_url, embedding_data)
                if person_id > 0:
                    self.logger.info(f"New person added: {person_name} (ID: {person_id})")
                    self.store_visit_info(person_id, visit_id, customer_id, entry_time, 
                                        image_url, embedding_data.get('saved_image_path'), similarity)
                    
                    # Create person group for batch webhook
                    person_group = {
                        'person_id': person_id,
                        'person_name': person_name,
                        'visits': [{
                            'visit_id': visit_id,
                            'customer_id': customer_id,
                            'image_url': image_url,
                            'entry_time': entry_time,
                            'similarity': similarity,
                            'branchId': visit.get('branchId', ''),
                            'camera': visit.get('camera', ''),
                            'entryEventIds': visit.get('entryEventIds', []),
                            'customer': visit.get('customer', {}),
                            'results': visit.get('results', {})
                        }]
                    }
                    
                    visit_results['new_persons'] += 1
                else:
                    self.logger.warning(f"Failed to add new person: {person_name}")
                    visit_results['duplicate_faces'] += 1
            
            return visit_results, person_group
        
        # Process visits using multi-threading
        max_workers = min(self.config['image_processing']['max_workers'], len(valid_visits))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all visits for processing
            future_to_visit = {executor.submit(process_single_visit_json, (i, visit)): (i, visit) for i, visit in enumerate(valid_visits)}
            
            # Collect results as they complete
            for future in as_completed(future_to_visit):
                try:
                    visit_results, person_group = future.result()
                    
                    # Thread-safe aggregation of results
                    with results_lock:
                        for key in results:
                            results[key] += visit_results[key]
                        
                        # Add person group to batch webhook data
                        if person_group:
                            batch_groups.append(person_group)
                            
                except Exception as e:
                    i, visit = future_to_visit[future]
                    self.logger.error(f"Error processing visit {i}: {e}")
                    with results_lock:
                        results['no_faces'] += 1
        
        # Print summary
        self.logger.info("Processing completed!")
        self.logger.info(f"Results: {results}")
        
        # Save clustering results to JSON file
        if batch_groups:
            try:
                self.logger.info(f"Saving clustering results with {len(batch_groups)} groups")
                
                # Save clustering results to JSON
                success = save_clustering_results(
                    groups=batch_groups,
                    total_processed=results['processed'],
                    results=results
                )
                
                if success:
                    self.logger.info("Clustering results saved successfully")
                else:
                    self.logger.warning("Failed to save clustering results")
                    
            except Exception as e:
                self.logger.error(f"Error saving clustering results: {e}")
        else:
            self.logger.info("No groups to save in clustering results")
        
        return results
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Total persons
        cursor.execute('SELECT COUNT(*) FROM persons')
        total_persons = cursor.fetchone()[0]
        
        # Average quality
        cursor.execute('SELECT AVG(face_quality) FROM persons')
        avg_quality = cursor.fetchone()[0] or 0
        
        # Recent activity
        cursor.execute('''
            SELECT COUNT(*) FROM persons 
            WHERE last_seen > datetime('now', '-1 day')
        ''')
        recent_activity = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_persons': total_persons,
            'average_quality': float(avg_quality),
            'recent_activity': recent_activity,
            'embeddings_loaded': self.vector_db.get_embedding_count()
        }
    
    def get_person_groups_for_web(self) -> List[Dict]:
        """
        Get person groups with images for web interface
        
        Returns:
            List of person groups with visit information
        """
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Check if person_visits table exists and has data
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='person_visits'")
        visits_table_exists = cursor.fetchone() is not None
        
        if not visits_table_exists:
            # If no visits table, just return persons with their main image
            cursor.execute('''
                SELECT id, name, image_path, face_quality, match_count, last_seen
                FROM persons
                ORDER BY match_count DESC, last_seen DESC
            ''')
            
            persons = cursor.fetchall()
            person_groups = []
            
            for person in persons:
                person_id, name, image_path, face_quality, match_count, last_seen = person
                
                person_groups.append({
                    'person_id': person_id,
                    'name': name,
                    'image_path': image_path,
                    'face_quality': face_quality,
                    'match_count': match_count,
                    'last_seen': last_seen,
                    'visit_count': 0,
                    'avg_quality': face_quality,
                    'images': [{
                        'visit_id': f'person_{person_id}',
                        'customer_id': name,
                        'entry_time': last_seen or '',
                        'image_url': image_path,
                        'image_path': image_path,
                        'similarity': 1.0
                    }] if image_path else []
                })
            
            conn.close()
            return person_groups
        
        # Get all persons with their visit information
        cursor.execute('''
            SELECT 
                p.id,
                p.name,
                p.image_path,
                p.face_quality,
                p.match_count,
                p.last_seen,
                COUNT(v.visit_id) as visit_count
            FROM persons p
            LEFT JOIN (
                SELECT DISTINCT 
                    person_id,
                    visit_id,
                    entry_time,
                    image_url,
                    saved_image_path
                FROM person_visits
            ) v ON p.id = v.person_id
            GROUP BY p.id, p.name, p.image_path, p.face_quality, p.match_count, p.last_seen
            ORDER BY p.match_count DESC, p.last_seen DESC
        ''')
        
        persons = cursor.fetchall()
        
        # Get visit details for each person
        person_groups = []
        for person in persons:
            person_id, name, image_path, face_quality, match_count, last_seen, visit_count = person
            
            # Get all visits for this person
            cursor.execute('''
                SELECT 
                    visit_id,
                    customer_id,
                    entry_time,
                    image_url,
                    saved_image_path,
                    similarity
                FROM person_visits
                WHERE person_id = ?
                ORDER BY entry_time DESC
            ''', (person_id,))
            
            visits = cursor.fetchall()
            
            # Prepare images list, include all images for this person
            images = []
            for visit in visits:
                visit_id, customer_id, entry_time, image_url, saved_image_path, similarity = visit
                if similarity is not None:
                    # Use saved image path if available, otherwise cache the URL
                    if saved_image_path and os.path.exists(saved_image_path):
                        display_path = saved_image_path
                    elif image_url and image_url.startswith('http'):
                        cached_path = self.get_cached_image_path(image_url)
                        display_path = cached_path if cached_path else image_url
                    else:
                        display_path = image_url
                    images.append({
                        'visit_id': visit_id,
                        'customer_id': customer_id,
                        'entry_time': entry_time,
                        'image_url': image_url,
                        'image_path': display_path,
                        'similarity': similarity
                    })
            
            # If no visits but person exists, add their main image
            if not images and image_path:
                images.append({
                    'visit_id': f'person_{person_id}',
                    'customer_id': name,
                    'entry_time': last_seen or '',
                    'image_url': image_path,
                    'image_path': image_path,
                    'similarity': 1.0
                })
            
            person_groups.append({
                'person_id': person_id,
                'name': name,
                'image_path': image_path,
                'face_quality': face_quality,
                'match_count': match_count,
                'last_seen': last_seen,
                'visit_count': visit_count,
                'avg_quality': face_quality,
                'images': images
            })
        
        conn.close()
        return person_groups
    
    def get_web_stats(self) -> Dict:
        """
        Get statistics for web interface
        
        Returns:
            Dictionary with web statistics
        """
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Total persons
        cursor.execute('SELECT COUNT(*) FROM persons')
        total_persons = cursor.fetchone()[0]
        
        # Check if person_visits table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='person_visits'")
        visits_table_exists = cursor.fetchone() is not None
        
        if visits_table_exists:
            # Total visits
            cursor.execute('SELECT COUNT(DISTINCT visit_id) FROM person_visits')
            total_visits = cursor.fetchone()[0]
            
            # Total images (unique image URLs)
            cursor.execute('SELECT COUNT(DISTINCT image_url) FROM person_visits')
            total_images = cursor.fetchone()[0]
        else:
            # If no visits table, count persons as images
            total_visits = 0
            cursor.execute('SELECT COUNT(*) FROM persons WHERE image_path IS NOT NULL')
            total_images = cursor.fetchone()[0]
        
        # Low similarity images count
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='low_similarity_images'")
        low_sim_table_exists = cursor.fetchone() is not None
        
        if low_sim_table_exists:
            cursor.execute('SELECT COUNT(*) FROM low_similarity_images')
            low_similarity_count = cursor.fetchone()[0]
        else:
            low_similarity_count = 0
        
        # Recent activity (last 24 hours)
        cursor.execute('''
            SELECT COUNT(*) FROM persons 
            WHERE last_seen > datetime('now', '-1 day')
        ''')
        recent_activity = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_persons': total_persons,
            'total_visits': total_visits,
            'total_images': total_images,
            'low_similarity_count': low_similarity_count,
            'recent_activity': recent_activity
        }
    
    def get_low_similarity_images(self) -> List[Dict]:
        """
        Get low similarity images for web interface
        
        Returns:
            List of low similarity images with details
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Check if low similarity table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='low_similarity_images'")
            if not cursor.fetchone():
                conn.close()
                return []
            
            # Get low similarity images
            cursor.execute('''
                SELECT 
                    visit_id,
                    customer_id,
                    entry_time,
                    image_url,
                    saved_image_path,
                    similarity,
                    best_match_name,
                    reason,
                    processed_at
                FROM low_similarity_images
                ORDER BY similarity DESC, processed_at DESC
            ''')
            
            images = []
            for row in cursor.fetchall():
                visit_id, customer_id, entry_time, image_url, saved_image_path, similarity, best_match_name, reason, processed_at = row
                
                # Use saved image path if available, otherwise cache the URL
                if saved_image_path and os.path.exists(saved_image_path):
                    display_path = saved_image_path
                elif image_url and image_url.startswith('http'):
                    # Try to get cached version of the URL
                    cached_path = self.get_cached_image_path(image_url)
                    display_path = cached_path if cached_path else image_url
                else:
                    display_path = image_url
                
                images.append({
                    'visit_id': visit_id,
                    'customer_id': customer_id,
                    'entry_time': entry_time,
                    'image_url': image_url,
                    'image_path': display_path,
                    'similarity': max(0, min(100, similarity * 100)) if similarity else 0,  # Convert to percentage and ensure 0-100 range
                    'best_match_name': best_match_name,
                    'reason': reason or 'Low similarity',
                    'processed_at': processed_at
                })
            
            conn.close()
            return images
        except Exception as e:
            self.logger.error(f"Error getting low similarity images: {e}")
            return []

    def is_duplicate_image(self, image_url: str, embedding: np.ndarray) -> bool:
        """Check if this image has already been processed"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            try:
                # Check if image URL already exists
                cursor.execute('SELECT COUNT(*) FROM person_visits WHERE image_url = ?', (image_url,))
                if cursor.fetchone()[0] > 0:
                    return True
                
                # Check if image URL exists in low similarity table
                cursor.execute('SELECT COUNT(*) FROM low_similarity_images WHERE image_url = ?', (image_url,))
                if cursor.fetchone()[0] > 0:
                    return True
                
                # Check if embedding is too similar to existing ones using Qdrant
                if self.vector_db.get_embedding_count() > 0:
                    duplicate_threshold = self.config['face_recognition']['duplicate_similarity_threshold']
                    similar_faces = self.vector_db.search_similar(
                        query_embedding=embedding,
                        k=1,  # Only need the most similar
                        threshold=duplicate_threshold
                    )
                    
                    if similar_faces and len(similar_faces) > 0:
                        return True
                
                return False
            finally:
                conn.close()
        except Exception as e:
            self.logger.error(f"Error checking for duplicate image: {e}")
            return False  # If there's an error, assume it's not a duplicate

    def clear_all_data(self):
        """Clear all data from database"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            # Clear all tables
            cursor.execute('DELETE FROM person_visits')
            cursor.execute('DELETE FROM low_similarity_images')
            cursor.execute('DELETE FROM persons')
            
            conn.commit()
            self.logger.info("All data cleared from database")
        except Exception as e:
            self.logger.error(f"Error clearing database: {e}")
            conn.rollback()
        finally:
            conn.close()
        
        # Reset in-memory data
        self.face_quality_cache = {}
        self.image_cache = {}
    
    # Webhook functionality removed - now using JSON storage

    def merge_duplicate_persons(self, person_id1: int, person_id2: int):
        """
        Merge two persons that are actually the same person
        
        Args:
            person_id1: ID of the person to keep
            person_id2: ID of the person to merge and delete
        """
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            # Update all visits from person_id2 to person_id1
            cursor.execute('''
                UPDATE person_visits 
                SET person_id = ? 
                WHERE person_id = ?
            ''', (person_id1, person_id2))
            
            # Update match count for person_id1
            cursor.execute('''
                UPDATE persons 
                SET match_count = match_count + (
                    SELECT match_count FROM persons WHERE id = ?
                )
                WHERE id = ?
            ''', (person_id2, person_id1))
            
            # Delete the duplicate person from SQLite
            cursor.execute('DELETE FROM persons WHERE id = ?', (person_id2,))
            
            conn.commit()
            self.logger.info(f"Merged person {person_id2} into person {person_id1} in SQLite")
            
            # Delete the duplicate person from Qdrant
            try:
                self.vector_db.delete_embedding(person_id2)
                self.logger.info(f"Deleted person {person_id2} from Qdrant")
            except Exception as e:
                self.logger.error(f"Error deleting person {person_id2} from Qdrant: {e}")
            
        except Exception as e:
            self.logger.error(f"Error merging persons {person_id1} and {person_id2}: {e}")
            conn.rollback()
        finally:
            conn.close()

    def find_and_merge_duplicates(self, similarity_threshold: float = None):
        """
        Find and merge duplicate persons based on high similarity using Qdrant
        
        Args:
            similarity_threshold: Threshold for considering persons as duplicates
        """
        try:
            # Use config default if threshold not provided
            if similarity_threshold is None:
                similarity_threshold = self.config['face_recognition']['merge_duplicate_threshold']
            
            # Get all person IDs from SQLite
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            cursor.execute('SELECT id, name FROM persons ORDER BY id')
            persons = cursor.fetchall()
            conn.close()
            
            if len(persons) < 2:
                self.logger.info("Not enough persons to find duplicates")
                return
            
            self.logger.info(f"Searching for duplicate persons among {len(persons)} persons...")
            
            # Check each person against others using Qdrant
            processed_pairs = set()
            duplicates_found = 0
            
            for i, (person_id1, name1) in enumerate(persons):
                # Get embedding for person1
                embedding1 = self.vector_db.get_embedding(person_id1)
                if embedding1 is None:
                    continue
                
                # Search for similar persons using Qdrant
                similar_persons = self.vector_db.search_similar(
                    query_embedding=embedding1,
                    k=len(persons),  # Get all persons
                    threshold=similarity_threshold
                )
                
                # Check each similar person
                for similar in similar_persons:
                    person_id2 = similar['person_id']
                    similarity = similar['similarity']
                    
                    # Skip self and already processed pairs
                    if (person_id1 >= person_id2 or 
                        (person_id1, person_id2) in processed_pairs or 
                        (person_id2, person_id1) in processed_pairs):
                        continue
                    
                    # Mark this pair as processed
                    processed_pairs.add((person_id1, person_id2))
                    
                    # Get name for person2
                    name2 = similar['name']
                    
                    self.logger.info(f"Found duplicate persons: {name1} (ID: {person_id1}) and {name2} (ID: {person_id2}) (similarity: {similarity:.3f})")
                    
                    # Merge person2 into person1
                    self.merge_duplicate_persons(person_id1, person_id2)
                    duplicates_found += 1
                    
                    # Update the persons list to reflect the merge
                    persons = [(pid, name) for pid, name in persons if pid != person_id2]
            
            self.logger.info(f"Duplicate detection completed. Found and merged {duplicates_found} duplicate pairs.")
            
        except Exception as e:
            self.logger.error(f"Error finding duplicates: {e}")


def main():
    # Load configuration from JSON file
    try:
        with open("config.json", 'r') as f:
            config = json.load(f)
        print("Configuration loaded from config.json")
    except FileNotFoundError:
        print("Configuration file config.json not found, using defaults")
        config = {
            'system': {'database_path': 'face_database.db', 'model_name': 'buffalo_l', 'gpu_id': 0},
            'face_detection': {'confidence_threshold': 0.5, 'quality_threshold': 0.3},
            'face_recognition': {'similarity_threshold': 0.4},
            'processing': {'max_visits_default': 500}
        }
    
    # Configuration - Update these paths as needed
    JSON_FILE_PATH = "visit-cluster.json"  # JSON file containing visit data with image URLs
    OUTPUT_FOLDER = "processed_visits"  # Folder to organize recognized persons
    DATABASE_PATH = config['system']['database_path']
    MODEL_NAME = config['system']['model_name']
    GPU_ID = config['system']['gpu_id']
    SIMILARITY_THRESH = config['face_recognition']['similarity_threshold']
    CONFIDENCE_THRESH = config['face_detection']['confidence_threshold']
    QUALITY_THRESH = config['face_detection']['quality_threshold']
    MAX_VISITS = config['processing']['max_visits_default']
    
    print("🌟 Smart Face Recognition System - JSON URL Processing")
    print("=" * 60)
    print(f"JSON file: {JSON_FILE_PATH}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Database: {DATABASE_PATH}")
    print(f"Model: {MODEL_NAME}")
    print(f"GPU ID: {GPU_ID}")
    print(f"Similarity threshold: {SIMILARITY_THRESH}")
    print(f"Confidence threshold: {CONFIDENCE_THRESH}")
    print(f"Quality threshold: {QUALITY_THRESH}")
    print(f"Max visits to process: {MAX_VISITS if MAX_VISITS else 'All'}")
    print("=" * 60)
    
    # Initialize system
    face_recognition = SmartFaceRecognition(
        database_path=DATABASE_PATH,
        model_name=MODEL_NAME,
        gpu_id=GPU_ID,
        confidence_thresh=CONFIDENCE_THRESH,
        similarity_thresh=SIMILARITY_THRESH,
        quality_thresh=QUALITY_THRESH
    )
    
    # Process visit data from JSON
    results = face_recognition.process_visit_data(JSON_FILE_PATH, OUTPUT_FOLDER, MAX_VISITS, save_images=True)
    
    # Print database stats
    stats = face_recognition.get_database_stats()
    print(f"\n📊 Database Statistics:")
    print(f"Total persons: {stats['total_persons']}")
    print(f"Average quality: {stats['average_quality']:.3f}")
    print(f"Recent activity: {stats['recent_activity']}")
    print(f"Embeddings loaded: {stats['embeddings_loaded']}")
    
    # Print processing results
    print(f"\n📈 Processing Results:")
    print(f"Visits processed: {results['processed']}")
    print(f"Persons recognized: {results['recognized']}")
    print(f"New persons added: {results['new_persons']}")
    print(f"No faces found: {results['no_faces']}")
    print(f"Low quality faces: {results['low_quality']}")
    print(f"Duplicate faces: {results['duplicate_faces']}")
    print(f"Download failures: {results['download_failed']}")


# FastAPI Web Interface
app = FastAPI(title="🌟 Smart Face Recognition System", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Global face recognition instance
face_recognition = None

@app.on_event("startup")
async def startup_event():
    """Initialize the face recognition system on startup"""
    global face_recognition
    try:
        # Load configuration for web server
        try:
            with open("config.json", 'r') as f:
                config = json.load(f)
            print("Configuration loaded from config.json for web server")
        except FileNotFoundError:
            print("Configuration file config.json not found, using defaults for web server")
            config = {
                'system': {'database_path': 'face_database.db', 'model_name': 'buffalo_l', 'gpu_id': 0},
                'face_detection': {'confidence_threshold': 0.4, 'quality_threshold': 0.3},
                'face_recognition': {'similarity_threshold': 0.7}
            }
        
        face_recognition = SmartFaceRecognition(
            database_path=config['system']['database_path'],
            model_name=config['system']['model_name'],
            gpu_id=config['system']['gpu_id'],
            confidence_thresh=config['face_detection']['confidence_threshold'],
            similarity_thresh=config['face_recognition']['similarity_threshold'],
            quality_thresh=config['face_detection']['quality_threshold']
        )
        print("🌟 Face Recognition System initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing face recognition system: {e}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        stats = face_recognition.get_web_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.get("/api/config")
async def get_api_config():
    """Get API configuration from config file"""
    try:
        config = load_api_config()
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading config: {str(e)}")

@app.get("/api/person-groups")
async def get_person_groups():
    """Get person groups with images"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        groups = face_recognition.get_person_groups_for_web()
        return groups
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting person groups: {str(e)}")

@app.get("/api/low-similarity-images")
async def get_low_similarity_images():
    """Get low similarity images"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        images = face_recognition.get_low_similarity_images()
        return images
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting low similarity images: {str(e)}")

@app.post("/api/merge-duplicates")
async def merge_duplicates():
    """Find and merge duplicate persons"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        face_recognition.find_and_merge_duplicates()
        return {"message": "Duplicate detection and merging completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error merging duplicates: {str(e)}")

@app.post("/api/clear-database")
async def clear_database():
    """Clear all data from database"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        face_recognition.clear_all_data()
        return {"message": "Database cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")

# Webhook API endpoint removed - now using JSON storage

@app.get("/api/person/{person_id}")
async def get_person_details(person_id: int):
    """Get detailed information about a specific person"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        conn = sqlite3.connect(face_recognition.database_path)
        cursor = conn.cursor()
        
        # Get person details
        cursor.execute('''
            SELECT id, name, image_path, face_quality, match_count, last_seen, created_at
            FROM persons WHERE id = ?
        ''', (person_id,))
        
        person = cursor.fetchone()
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        
        # Get all visits for this person
        cursor.execute('''
            SELECT visit_id, customer_id, entry_time, image_url, saved_image_path, similarity, processed_at
            FROM person_visits
            WHERE person_id = ?
            ORDER BY entry_time DESC
        ''', (person_id,))
        
        visits = cursor.fetchall()
        conn.close()
        
        return {
            'person_id': person[0],
            'name': person[1],
            'image_path': person[2],
            'face_quality': person[3],
            'match_count': person[4],
            'last_seen': person[5],
            'created_at': person[6],
            'visits': [
                {
                    'visit_id': visit[0],
                    'customer_id': visit[1],
                    'entry_time': visit[2],
                    'image_url': visit[3],
                    'saved_image_path': visit[4],
                    'similarity': visit[5],
                    'processed_at': visit[6]
                }
                for visit in visits
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting person details: {str(e)}")

@app.post("/api/process-visits")
async def process_visits(request_data: dict = None):
    """Process visits from JSON data or file"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        # Extract parameters from request
        json_data = request_data.get('json_data') if request_data else None
        # Auto-detect max visits from JSON data
        max_visits = request_data.get('max_visits', None) if request_data else None
        if max_visits is None and json_data:
            max_visits = len(json_data.get('visits', []))
        elif max_visits is None:
            max_visits = face_recognition.config['processing']['max_visits_fallback']
        save_images = request_data.get('save_images', True) if request_data else True
        clear_existing = request_data.get('clear_existing', False) if request_data else False
        
        if json_data:
            # Process JSON data from web interface
            results = face_recognition.process_visit_data_from_json(
                json_data=json_data,
                output_folder="processed_visits",
                max_visits=max_visits,
                save_images=save_images,
                clear_existing=clear_existing
            )
        else:
            # Process from file (fallback)
            results = face_recognition.process_visit_data(
                json_file_path="visit-cluster.json",
                output_folder="processed_visits",
                max_visits=max_visits,
                save_images=save_images
            )
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing visits: {str(e)}")

@app.post("/api/process-face-comparisons-from-api")
async def process_face_comparisons_from_api(request_data: dict):
    """Process face comparisons from external API - New structure with url1 and url2"""
    try:
        # Initialize the face comparison system
        face_comparison = FaceComparisonFromAPI()
        
        # Extract API parameters from request
        api_url = request_data.get('api_url')
        if not api_url:
            raise HTTPException(status_code=400, detail="API URL is required")
        
        start_date = request_data.get('start_date')
        end_date = request_data.get('end_date')
        page = request_data.get('page', 0)
        limit = request_data.get('limit', 100)
        start_time = request_data.get('start_time')
        end_time = request_data.get('end_time')
        all_branch = request_data.get('all_branch', True)
        max_comparisons = request_data.get('max_comparisons', limit)
        api_key = request_data.get('api_key')
        auth_token = request_data.get('auth_token')
        
        # Fetch face comparison data from API
        comparison_records = face_comparison.fetch_face_comparison_data_from_api(
            api_url=api_url,
            start_date=start_date,
            end_date=end_date,
            page=page,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            all_branch=all_branch,
            api_key=api_key,
            auth_token=auth_token
        )
        
        if not comparison_records:
            return {
                "message": "No face comparison records found from API",
                "total_comparisons": 0,
                "processed": 0,
                "same_person": 0,
                "different_person": 0,
                "errors": 0,
                "results": []
            }
        
        # Process the face comparisons
        results = face_comparison.process_face_comparisons(
            comparison_records=comparison_records,
            max_comparisons=max_comparisons
        )
        
        # Add API-specific information to results
        results['api_info'] = {
            'api_url': api_url,
            'fetched_records': len(comparison_records),
            'parameters': {
                'start_date': start_date,
                'end_date': end_date,
                'page': page,
                'limit': limit,
                'start_time': start_time,
                'end_time': end_time,
                'all_branch': all_branch
            }
        }
        
        # Save results to JSON file
        try:
            import json
            from datetime import datetime
            
            print(f"🔧 Starting JSON file creation...")
            print(f"🔧 Results keys: {list(results.keys())}")
            print(f"🔧 Results length: {len(results.get('results', []))}")
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"face_comparison_results_{timestamp}.json"
            print(f"🔧 Creating JSON file: {filename}")
            
            # Prepare data for JSON file (simplified format as requested)
            json_data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'total_comparisons': results.get('total_comparisons', 0),
                    'same_person': results.get('same_person', 0),
                    'different_person': results.get('different_person', 0),
                    'errors': results.get('errors', 0),
                    'accuracy_vs_api': results.get('accuracy_vs_api', 0)
                },
                'comparisons': []
            }
            
            # Add each comparison result in the requested format - only include specified fields
            for comparison in results.get('results', []):
                # Extract fileName, event, and camera from the raw_data if available
                raw_data = comparison.get('raw_data', {})
                entry_event_ids = raw_data.get('entryEventIds', [])
                
                # Get data from first entryEventId if available
                fileName = ''
                event = ''
                camera = ''
                eventId = ''
                
                if entry_event_ids and len(entry_event_ids) > 0:
                    event_data = entry_event_ids[0]
                    fileName = event_data.get('fileName', '')
                    event = event_data.get('event', '')
                    camera = event_data.get('camera', '')
                    eventId = event_data.get('eventId', '')
                else:
                    # Fallback: try to get eventId from the comparison's event_id if it's a string
                    event_id = comparison.get('event_id', '')
                    if isinstance(event_id, str):
                        eventId = event_id
                
                comparison_data = {
                    'fileName': fileName,
                    'event': event,
                    'camera': camera,
                    'eventId': eventId,
                    'approve': comparison.get('api_approve', False),
                    'match_status': comparison.get('match_status', 'UNKNOWN'),
                    'branch_id': comparison.get('branch_id', '')
                }
                json_data['comparisons'].append(comparison_data)
            
            # Write to JSON file
            print(f"🔧 Writing to JSON file: {filename}")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"🔧 JSON file written successfully")
            
            # Add file info to results
            results['json_file'] = {
                'filename': filename,
                'path': os.path.abspath(filename),
                'size': os.path.getsize(filename)
            }
            
            print(f"✅ Face comparison results saved to: {filename}")
            print(f"✅ File size: {os.path.getsize(filename)} bytes")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save results to JSON file: {e}")
            results['json_file'] = {'error': str(e)}
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing face comparisons from API: {str(e)}")

@app.post("/api/process-visits-from-api")
async def process_visits_from_api(request_data: dict):
    """Process visits from external API"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        # Extract API parameters from request
        api_url = request_data.get('api_url')
        if not api_url:
            raise HTTPException(status_code=400, detail="API URL is required")
        
        start_date = request_data.get('start_date')
        end_date = request_data.get('end_date')
        page = request_data.get('page', 0)
        limit = request_data.get('limit', 100)
        start_time = request_data.get('start_time')
        end_time = request_data.get('end_time')
        all_branch = request_data.get('all_branch', True)
        max_visits = request_data.get('max_visits', limit)
        save_images = request_data.get('save_images', True)
        clear_existing = request_data.get('clear_existing', False)
        api_key = request_data.get('api_key')
        auth_token = request_data.get('auth_token')
        
        # Fetch data from API
        visits = face_recognition.fetch_visit_data_from_api(
            api_url=api_url,
            start_date=start_date,
            end_date=end_date,
            page=page,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            all_branch=all_branch,
            api_key=api_key,
            auth_token=auth_token
        )
        
        if not visits:
            return {
                "message": "No visits found from API",
                "processed": 0,
                "recognized": 0,
                "new_persons": 0,
                "no_faces": 0,
                "low_quality": 0,
                "download_failed": 0,
                "duplicate_faces": 0,
                "low_similarity": 0
            }
        
        # Limit visits if max_visits is specified
        if max_visits and len(visits) > max_visits:
            visits = visits[:max_visits]
        
        # Create JSON data structure for processing
        json_data = {
            "visits": visits,
            "total_visits": len(visits),
            "api_url": api_url,
            "fetched_at": datetime.now().isoformat()
        }
        
        # Process the visits using existing logic
        results = face_recognition.process_visit_data_from_json(
            json_data=json_data,
            output_folder="processed_visits",
            max_visits=max_visits,
            save_images=save_images,
            clear_existing=clear_existing
        )
        
        # Add API-specific information to results
        results['api_info'] = {
            'api_url': api_url,
            'fetched_visits': len(visits),
            'parameters': {
                'start_date': start_date,
                'end_date': end_date,
                'page': page,
                'limit': limit,
                'start_time': start_time,
                'end_time': end_time,
                'all_branch': all_branch
            }
        }
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing visits from API: {str(e)}")

@app.get("/api/image/{image_path:path}")
async def serve_image(image_path: str):
    """Serve local images with proper caching and error handling"""
    try:
        # Decode the image path
        decoded_path = image_path.replace('%3A', ':').replace('%2F', '/').replace('%5C', '\\')
        
        # Check if file exists
        if not os.path.exists(decoded_path):
            # Try to serve the no-image placeholder
            no_image_path = "static/no-image.png"
            if os.path.exists(no_image_path):
                return FileResponse(no_image_path)
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Return the image file with proper headers for caching
        response = FileResponse(decoded_path)
        cache_max_age = face_recognition.config['web_interface']['cache_control_max_age']
        response.headers["Cache-Control"] = f"public, max-age={cache_max_age}"
        response.headers["Content-Type"] = "image/jpeg"
        return response
    except Exception as e:
        # Try to serve the no-image placeholder on error
        no_image_path = "static/no-image.png"
        if os.path.exists(no_image_path):
            return FileResponse(no_image_path)
        raise HTTPException(status_code=500, detail=f"Error serving image: {str(e)}")


@app.get("/api/image-base64/{image_path:path}")
async def serve_image_base64(image_path: str):
    """Serve images as base64 for better web performance"""
    try:
        # Decode the image path
        decoded_path = image_path.replace('%3A', ':').replace('%2F', '/').replace('%5C', '\\')
        
        # Check if file exists
        if not os.path.exists(decoded_path):
            return {"error": "Image not found", "base64": None}
        
        # Process image for web
        if face_recognition:
            base64_image = face_recognition.process_image_for_web(decoded_path)
            if base64_image:
                return {"base64": base64_image}
        
        # Fallback to regular file serving
        return {"error": "Could not process image", "base64": None}
    except Exception as e:
        return {"error": f"Error processing image: {str(e)}", "base64": None}

@app.post("/api/clear-cache")
async def clear_image_cache():
    """Clear the image cache"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        import shutil
        cache_dir = face_recognition.image_cache_dir
        
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            face_recognition.logger.info("Image cache cleared successfully")
            return {"message": "Cache cleared successfully"}
        else:
            return {"message": "Cache directory does not exist"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

@app.post("/api/clear-database")
async def clear_database():
    """Clear the face recognition database"""
    if not face_recognition:
        raise HTTPException(status_code=500, detail="Face recognition system not initialized")
    
    try:
        face_recognition.clear_database()
        return {"message": "Database cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")

def run_web_server(host: str = None, port: int = None):
    """Run the FastAPI web server"""
    # Load configuration for web server
    try:
        with open("config.json", 'r') as f:
            config = json.load(f)
        print("Configuration loaded from config.json for web server")
    except FileNotFoundError:
        print("Configuration file config.json not found, using defaults for web server")
        config = {
            'web_interface': {'host': '0.0.0.0', 'port': 8000}
        }
    
    # Use config defaults if not provided
    if host is None:
        host = config['web_interface']['host']
    if port is None:
        port = config['web_interface']['port']
    
    print(f"🌟 Starting Smart Face Recognition Web Server...")
    print(f"🌐 Server will be available at: http://{host}:{port}")
    print(f"📱 Web Interface: http://{host}:{port}")
    print(f"📊 API Documentation: http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        # Run web server
        run_web_server()
    else:
        # Run original main function
        main()
# pm2 start main.py --interpreter=python3 --name=boxmot-server