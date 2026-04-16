import pymongo
from datetime import datetime
import os

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "sentiment_analysis_db"
COLLECTION_NAME = "feedback_comments"

class MongoDBHandler:
    def __init__(self):
        try:
            self.client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            # The ismaster command is cheap and does not require auth.
            self.client.admin.command('ismaster')
            self.db = self.client[DB_NAME]
            self.collection = self.db[COLLECTION_NAME]
            print("Connected to MongoDB successfully.")
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            self.client = None
            self.db = None
            self.collection = None

    def save_feedback(self, feedback_data):
        """
        Saves a single feedback entry to MongoDB.
        feedback_data: dict containing keys like 'text', 'label', 'score', 'theme', 'region', etc.
        """
        if self.collection is not None:
            # Ensure timestamp is present
            if 'timestamp' not in feedback_data:
                feedback_data['timestamp'] = datetime.now()
            elif isinstance(feedback_data['timestamp'], str):
                try:
                    feedback_data['timestamp'] = datetime.strptime(feedback_data['timestamp'], "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    feedback_data['timestamp'] = datetime.now()
            
            result = self.collection.insert_one(feedback_data)
            return str(result.inserted_id)
        return None

    def save_batch_feedback(self, feedback_list):
        """
        Saves multiple feedback entries to MongoDB.
        """
        if self.collection is not None and feedback_list:
            for item in feedback_list:
                if 'timestamp' not in item:
                    item['timestamp'] = datetime.now()
                elif isinstance(item['timestamp'], str):
                    try:
                        item['timestamp'] = datetime.strptime(item['timestamp'], "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        item['timestamp'] = datetime.now()
            
            result = self.collection.insert_many(feedback_list)
            return len(result.inserted_ids)
        return 0

    def get_all_feedback(self, limit=1000):
        """
        Retrieves feedback from MongoDB.
        """
        if self.collection is not None:
            cursor = self.collection.find().sort("timestamp", pymongo.DESCENDING).limit(limit)
            return list(cursor)
        return []

    def clear_all_feedback(self):
        """
        Clears the feedback collection.
        """
        if self.collection is not None:
            self.collection.delete_many({})
            return True
        return False

# Singleton instance
db_handler = MongoDBHandler()
