"""
JSON Storage System for Face Recognition Clustering
Saves clustering results to JSON files instead of webhooks
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import logging

class JSONStorageManager:
    def __init__(self, output_dir: str = "clustering_results"):
        """
        Initialize JSON Storage Manager
        
        Args:
            output_dir: Directory to save JSON files
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_job_id(self) -> str:
        """Create a unique job ID"""
        return str(uuid.uuid4())
    
    def format_groups_for_json(self, person_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format person groups data for JSON storage
        
        Args:
            person_groups: List of person groups from clustering
            
        Returns:
            Formatted groups for JSON storage
        """
        json_groups = []
        
        for group in person_groups:
            person_id = group.get('person_id')
            person_name = group.get('person_name', f'Person_{person_id}')
            visits = group.get('visits', [])
            
            # Calculate average similarity score for the group
            similarities = [visit.get('similarity', 0.0) for visit in visits if visit.get('similarity') is not None]
            group_score = sum(similarities) / len(similarities) if similarities else 0.0
            
            # Create group data
            group_data = self._create_group_data(
                person_id=person_id,
                person_name=person_name,
                visits=visits,
                group_score=group_score
            )
            
            if group_data:
                json_groups.append(group_data)
        
        return json_groups
    
    def _create_group_data(self, 
                          person_id: int,
                          person_name: str,
                          visits: List[Dict[str, Any]],
                          group_score: float = 0.0) -> Dict[str, Any]:
        """
        Create group data for a person cluster
        
        Args:
            person_id: Unique person identifier
            person_name: Person name/identifier
            visits: List of visits for this person
            group_score: Average similarity score for the group
            
        Returns:
            Group data dictionary
        """
        if not visits:
            return {}
            
        # Get data from the first visit as representative
        first_visit = visits[0]
        
        # Extract metadata from visit data
        branch_id = first_visit.get('branchId', '')
        camera = first_visit.get('camera', '')
        entry_time = first_visit.get('entryTime', first_visit.get('entry_time', ''))
        
        # Get event data from entryEventIds if available
        event_data = first_visit.get('entryEventIds', [])
        event_type = ''
        file_name = ''
        
        if event_data and len(event_data) > 0:
            event_info = event_data[0]
            event_type = event_info.get('event', '')
            file_name = event_info.get('fileName', '')
            # Also get camera from entryEventIds if not available at top level
            if not camera:
                camera = event_info.get('camera', '')
        
        # Extract age and gender from customer data if available
        customer_data = first_visit.get('customer', {})
        age = customer_data.get('age')
        gender = customer_data.get('gender')
        
        # If not available in customer data, calculate from visits
        if age is None:
            age = self._calculate_average_age(visits)
        if gender is None:
            gender = self._get_most_common_gender(visits)
        
        group_data = {
            "group_id": first_visit.get('customerId', first_visit.get('customer_id', '')),
            "person_id": person_id,
            "person_name": person_name,
            "timestamp": entry_time,
            "group_score": round(group_score, 3),
            "camera": camera,
            "event": event_type,
            "branchId": branch_id,
            "fileName": file_name,
            "age": age,
            "gender": gender,
            "visit_count": len(visits),
            "visits": [
                {
                    "visit_id": visit.get('visit_id', visit.get('id')),
                    "customer_id": visit.get('customerId', visit.get('customer_id')),
                    "image_url": visit.get('image_url', visit.get('image')),
                    "entry_time": visit.get('entryTime', visit.get('entry_time')),
                    "similarity": visit.get('similarity', 0.0)
                } for visit in visits
            ]
        }
        
        return group_data
    
    def _calculate_average_age(self, visits: List[Dict[str, Any]]) -> Optional[int]:
        """Calculate average age from visits if available"""
        ages = []
        for visit in visits:
            # Look for age in visit data or entryEventIds
            if 'age' in visit:
                try:
                    ages.append(int(visit['age']))
                except (ValueError, TypeError):
                    pass
            
            # Check entryEventIds for age data
            event_data = visit.get('entryEventIds', [])
            for event in event_data:
                if 'age' in event:
                    try:
                        ages.append(int(event['age']))
                    except (ValueError, TypeError):
                        pass
        
        if ages:
            return round(sum(ages) / len(ages))
        return None
    
    def _get_most_common_gender(self, visits: List[Dict[str, Any]]) -> Optional[str]:
        """Get most common gender from visits if available"""
        genders = []
        for visit in visits:
            # Look for gender in visit data or entryEventIds
            if 'gender' in visit:
                gender = visit['gender']
                if gender and gender.lower() in ['male', 'female', 'm', 'f']:
                    genders.append(gender.lower())
            
            # Check entryEventIds for gender data
            event_data = visit.get('entryEventIds', [])
            for event in event_data:
                if 'gender' in event:
                    gender = event['gender']
                    if gender and gender.lower() in ['male', 'female', 'm', 'f']:
                        genders.append(gender.lower())
        
        if genders:
            # Return most common gender
            from collections import Counter
            gender_counts = Counter(genders)
            return gender_counts.most_common(1)[0][0]
        return None
    
    def save_clustering_results(self, 
                              groups: List[Dict[str, Any]], 
                              total_processed: int,
                              results: Dict[str, Any]) -> bool:
        """
        Save clustering results to JSON file
        
        Args:
            groups: List of person groups
            total_processed: Total number of images processed
            results: Processing results dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            job_id = self.create_job_id()
            filename = f"clustering_results_{timestamp}_{job_id[:8]}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            # Format groups for JSON storage
            json_groups = self.format_groups_for_json(groups)
            
            # Create the complete results payload
            payload = {
                "job_id": job_id,
                "status": "finished",
                "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                "total_processed": total_processed,
                "total_groups": len(json_groups),
                "results": results,
                "message": f"Processing completed. Created {len(json_groups)} groups from {total_processed} images",
                "groups": json_groups
            }
            
            print(f"JSONStorageManager: Saving clustering results to: {filepath}")
            print(f"JSONStorageManager: Output directory: {self.output_dir}")
            print(f"JSONStorageManager: Current working directory: {os.getcwd()}")
            self.logger.info(f"JSONStorageManager: Saving clustering results to: {filepath}")
            self.logger.info(f"JSONStorageManager: Output directory: {self.output_dir}")
            self.logger.info(f"JSONStorageManager: Current working directory: {os.getcwd()}")
            
            # Save to JSON file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Clustering results saved successfully to: {filepath}")
            return True
                
        except Exception as e:
            self.logger.error(f"Unexpected error saving clustering results: {str(e)}")
            return False


# Global JSON storage manager instance
json_storage_manager = JSONStorageManager()

def save_clustering_results(groups: List[Dict[str, Any]], 
                          total_processed: int,
                          results: Dict[str, Any]) -> bool:
    """Save clustering results using global manager"""
    return json_storage_manager.save_clustering_results(groups, total_processed, results)
