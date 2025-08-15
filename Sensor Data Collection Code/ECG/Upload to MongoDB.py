# James Fothergill v8255920

import serial
import json
import time
from datetime import datetime
from pymongo import MongoClient

# MongoDB setup
connection_string = os.getenv("MONGODB_URI")
client = MongoClient(connection_string)
db = client['ecg_monitoring']
collection = db['ecg_filtered_readings']


def collect_ecg_data():
    try:
        # Test MongoDB connection
        client.admin.command('ping')
        print("Connected to MongoDB successfully!")

        # Connect to Arduino
        ser = serial.Serial('COM3', 9600, timeout=2)
        print("Connected to Arduino on COM3")
        time.sleep(2)  # Wait for Arduino to reset
        
        # Clear any initial data
        ser.reset_input_buffer()
        print("Starting data collection...")
        
        raw_data = []
        filtered_data = []
        sample_count = 0
        
        while sample_count < 50:  # Collect 50 samples
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    if line == "!":
                        print("Leads off detected!")
                        continue
                    if line.startswith("{"):
                        data = json.loads(line)
                        raw_data.append(data['raw_value'])
                        filtered_data.append(data['filtered_value'])
                        sample_count += 1
                        print(f"Collecting samples: {sample_count}/50")
                except json.JSONDecodeError as e:
                    print(f"JSON Error: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing data: {e}")
                    continue

        # Create final document for MongoDB
        document = {
            "device_id": "Arduino_R4",
            "timestamp": datetime.utcnow(),
            "sampling_rate": 50,  # 1000ms/20ms = 50Hz
            "filter_params": {
                "window_size": 3,
                "alpha": 0.5,
                "notch_alpha": 0.08
            },
            "raw_data": raw_data,
            "filtered_data": filtered_data
        }
        
        # Upload to MongoDB
        result = collection.insert_one(document)
        print("\nData Collection and Upload Complete!")
        print(f"Number of samples: {len(raw_data)}")
        print(f"Successfully uploaded to MongoDB with ID: {result.inserted_id}")

    except serial.SerialException as e:
        print(f"\nError: Could not open port COM3: {e}")
        print("Make sure Arduino IDE is completely closed!")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        if 'ser' in locals():
            ser.close()
            print("\nSerial port closed")
        if 'client' in locals():
            client.close()
            print("MongoDB connection closed")

if __name__ == "__main__":
    collect_ecg_data()
