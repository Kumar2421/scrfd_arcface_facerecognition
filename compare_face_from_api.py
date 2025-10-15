#!/usr/bin/env python3
"""
Face Comparison from API Module

This module handles face comparison functionality specifically for API data.
It includes its own threshold configuration and excludes side face detection.
"""

import os
import json
import logging
import requests
import cv2
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import insightface
from insightface.app import FaceAnalysis

class FaceComparisonFromAPI:
    """
    Face comparison system specifically designed for API data processing.
    Includes separate threshold configuration and excludes side face detection.
    """
    
    def __init__(self, config_file: str = "config.json"):
        """
        Initialize the face comparison system
        
        Args:
            config_file: Path to configuration file
        """
        self.config = self.load_config(config_file)
        self.setup_logging()
        self.setup_face_recognition()
        
        # Face comparison specific thresholds
        self.similarity_threshold = self.config.get('face_comparison', {}).get('similarity_threshold', 0.4)
        self.confidence_threshold = self.config.get('face_comparison', {}).get('confidence_threshold', 0.3)
        
        self.logger.info(f"Face comparison thresholds - Similarity: {self.similarity_threshold}, Confidence: {self.confidence_threshold}")
    
    def load_config(self, config_file: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('face_comparison.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_face_recognition(self):
        """Initialize InsightFace model for face recognition"""
        try:
            self.logger.info("Loading InsightFace model for face comparison...")
            self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.logger.info("InsightFace model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading InsightFace model: {e}")
            raise
    
    def download_image_from_url(self, url: str, timeout: int = 30) -> Optional[np.ndarray]:
        """
        Download image from URL and return as OpenCV image
        
        Args:
            url: Image URL
            timeout: Request timeout in seconds
            
        Returns:
            OpenCV image array or None if failed
        """
        try:
            self.logger.info(f"Downloading image from: {url}")
            
            # Set headers to request image content
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Cache-Control': 'no-cache'
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
                return None
            
            # Check if response is actually an image
            if not any(img_type in content_type for img_type in ['image/', 'application/octet-stream']):
                self.logger.warning(f"Unexpected content type: {content_type}")
                # Try to decode anyway in case it's an image with wrong headers
            
            # Convert response to image
            image_data = response.content
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                self.logger.error(f"Failed to decode image from: {url}")
                return None
            
            self.logger.info(f"Successfully downloaded image: {image.shape}")
            return image
            
        except Exception as e:
            self.logger.error(f"Error downloading image from {url}: {e}")
            return None
    
    def detect_faces(self, image: np.ndarray) -> List:
        """
        Detect faces in an image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            List of detected faces
        """
        try:
            # Convert BGR to RGB for InsightFace
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Detect faces
            faces = self.app.get(image_rgb)
            return faces
            
        except Exception as e:
            self.logger.error(f"Error detecting faces: {e}")
            return []
    
    def calculate_face_similarity(self, face1_embedding: np.ndarray, face2_embedding: np.ndarray) -> float:
        """
        Calculate similarity between two face embeddings
        
        Args:
            face1_embedding: First face embedding
            face2_embedding: Second face embedding
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            # Calculate cosine similarity
            similarity = np.dot(face1_embedding, face2_embedding) / (np.linalg.norm(face1_embedding) * np.linalg.norm(face2_embedding))
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Error calculating face similarity: {e}")
            return 0.0
    
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
            
            # Detect faces in both images
            faces1 = self.detect_faces(img1)
            faces2 = self.detect_faces(img2)
            
            if len(faces1) == 0 or len(faces2) == 0:
                return {
                    'same_person': False,
                    'confidence': 0.0,
                    'error': 'Could not detect faces in one or both images',
                    'image1_url': image1_url,
                    'image2_url': image2_url
                }
            
            # Get face embeddings from the first (largest) face in each image
            face1_embedding = faces1[0].embedding
            face2_embedding = faces2[0].embedding
            
            # Calculate similarity
            similarity = self.calculate_face_similarity(face1_embedding, face2_embedding)
            
            # Determine if same person based on threshold
            same_person = similarity > self.similarity_threshold
            
            self.logger.info(f"Face comparison result: {same_person} (confidence: {similarity:.4f}, threshold: {self.similarity_threshold})")
            
            return {
                'same_person': same_person,
                'confidence': float(similarity),
                'threshold_used': self.similarity_threshold,
                'image1_url': image1_url,
                'image2_url': image2_url,
                'error': None
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
    
    def fetch_face_comparison_data_from_api(self, api_url: str, start_date: str = None, end_date: str = None, 
                                           page: int = 0, limit: int = 100, start_time: str = None, 
                                           end_time: str = None, all_branch: bool = True, api_key: str = None, 
                                           auth_token: str = None) -> List[Dict]:
        """
        Fetch face comparison data from external API
        
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
            # Build query parameters for analytics API
            params = {
                'page': page,
                'limit': limit,
                'allBranch': str(all_branch).lower()
            }
            
            # Add date parameter (API uses single date parameter)
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
            
            # Extract visits from response
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
                    
                    # Extract event data from entryEventIds
                    event_data = None
                    if visit.get('entryEventIds') and len(visit.get('entryEventIds', [])) > 0:
                        event_data = visit.get('entryEventIds')[0]
                        # Debug logging
                        self.logger.info(f"Event data for visit {visit.get('id')}: {event_data}")
                        self.logger.info(f"fileName: {event_data.get('fileName')}, event: {event_data.get('event')}, camera: {event_data.get('camera')}")
                    else:
                        self.logger.warning(f"No entryEventIds found for visit {visit.get('id')}")
                    
                    # Map fields from API structure - only include required fields
                    comparison_record = {
                        'comparison_id': visit.get('id', f"comparison_{len(comparison_records)}"),
                        'event_id': event_data.get('eventId') if event_data else None,
                        'approve': visit.get('isConverted', False),
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
                        # Additional fields for JSON output - extract from event_data
                        'fileName': event_data.get('fileName', '') if event_data else '',
                        'event': event_data.get('event', '') if event_data else '',
                        'camera': event_data.get('camera', '') if event_data else '',
                        'raw_data': visit  # Store complete visit object with entryEventIds
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
                        'our_result': comparison_result.get('same_person', False),
                        'confidence': comparison_result.get('confidence', 0.0),
                        'threshold_used': comparison_result.get('threshold_used', self.similarity_threshold),
                        'image1_url': comparison_result.get('image1_url', ''),
                        'image2_url': comparison_result.get('image2_url', ''),
                        'error': comparison_result.get('error', None),
                        'match_status': 'SAME' if comparison_result.get('same_person', False) else 'DIFFERENT',
                        'api_vs_our_match': record['approve'] == comparison_result.get('same_person', False),
                        'raw_data': record.get('raw_data', {})
                    }
                    
                    results.append(result)
                    
                    # Update counters
                    if comparison_result.get('error'):
                        error_count += 1
                    elif comparison_result.get('same_person', False):
                        same_person_count += 1
                    else:
                        different_person_count += 1
                    
                    self.logger.info(f"Comparison {i+1} result: {result['match_status']} (confidence: {comparison_result.get('confidence', 0.0):.4f})")
                    
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

# Example usage
if __name__ == "__main__":
    # Initialize the face comparison system
    face_comparison = FaceComparisonFromAPI()
    
    # Test with sample data
    print("Testing Face Comparison from API...")
    
    # Test single comparison
    result = face_comparison.compare_face_images(
        'https://cdn.analytics.thefusionapps.com/v11/191371ed-a3ab-4246-a26e-895c847a0fda-1032025-100451AM.jpg',
        'https://cdn.analytics.thefusionapps.com/v4/GT-ODISHA/visitor-120409.jpg'
    )
    
    print("Single comparison result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
