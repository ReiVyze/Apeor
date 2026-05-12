from pymongo import MongoClient
from datetime import datetime
import os

MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
DB_NAME = 'sentiment_analysis_db'
COLLECTION_NAME = 'feedback_comments'

class MongoDBHandler:
    def __init__(self):
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
    
    def save_feedback(self, feedback_data):
        try:
            if 'timestamp' in feedback_data and isinstance(feedback_data['timestamp'], str):
                feedback_data['timestamp'] = datetime.strptime(feedback_data['timestamp'], "%Y-%m-%d %H:%M:%S")
            result = self.collection.insert_one(feedback_data)
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error saving feedback: {e}")
            return None
    
    def save_batch_feedback(self, feedback_list):
        if not feedback_list:
            return 0
        try:
            for item in feedback_list:
                if 'timestamp' in item and isinstance(item['timestamp'], str):
                    try:
                        item['timestamp'] = datetime.strptime(item['timestamp'], "%Y-%m-%d %H:%M:%S")
                    except:
                        pass
            result = self.collection.insert_many(feedback_list)
            return len(result.inserted_ids)
        except Exception as e:
            print(f"Error saving batch feedback: {e}")
            return 0
    
    def get_all_feedback(self, limit=1000):
        try:
            cursor = self.collection.find().sort('timestamp', -1).limit(limit)
            results = []
            for doc in cursor:
                doc['_id'] = str(doc['_id'])
                if isinstance(doc.get('timestamp'), datetime):
                    doc['timestamp'] = doc['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                results.append(doc)
            return results
        except Exception as e:
            print(f"Error retrieving feedback: {e}")
            return []
    
    def clear_all_feedback(self):
        try:
            self.collection.delete_many({})
            return True
        except Exception as e:
            print(f"Error clearing feedback: {e}")
            return False

db_handler = MongoDBHandler()
