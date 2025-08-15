# James Fothergill v8255920


# Import required libraries
import board
import busio as io
import adafruit_mlx90614      # Temperature sensor library
import max30102               # Heart rate and SpO2 sensor library
import time
import numpy as np
from scipy.signal import find_peaks
from datetime import datetime
from pymongo import MongoClient

# Setup MongoDB connection for data storage
connection_string = os.getenv("MONGODB_URI")
if not connection_string:
    raise ValueError("MongoDB URI not found. Check your .env file.")

client = MongoClient(connection_string)
db = client['ecg_monitoring']
collection = db['readings']


# Initialize I2C communication for the MLX90614 temperature sensor
i2c = io.I2C(board.SCL, board.SDA, frequency=100000)
mlx = adafruit_mlx90614.MLX90614(i2c)

def get_temperature_reading():

    # Interactive function to capture and validate temperature readings
    # Returns dictionary with ambient and target temperatures

    while True:
        ready = input("\nReady to take temperature reading? (y/n): ").lower()
        if ready != 'y':
            print("Okay, waiting until you're ready")
            continue
            
        # Get temperature readings with 2 decimal places
        ambientTemp = "{:.2f}".format(mlx.ambient_temperature)
        targetTemp = "{:.2f}".format(mlx.object_temperature)
        
        print("\nAmbient Temperature:", ambientTemp, "°C")
        print("Target Temperature:", targetTemp,"°C")
        
        # Validate readings with user
        while True:
            validation = input("Is this temperature reading correct? (y/n): ").lower()
            if validation == 'y':
                return {
                    "ambient_temp": float(ambientTemp),
                    "target_temp": float(targetTemp),
                    "timestamp": datetime.now(),
                    "validated": True
                }
            elif validation == 'n':
                print("Okay, let's try taking the temperature again")
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")

def detect_finger(red_value, ir_value):

    # Checks if a finger is present on the sensor
    # Uses minimum threshold values for both RED and IR readings
    
    MINIMUM_THRESHOLD = 10000
    return (red_value > MINIMUM_THRESHOLD) and (ir_value > MINIMUM_THRESHOLD)

def calculate_heart_rate(ir_data, sample_rate=10):

    # Processes IR sensor data to calculate heart rate
    # Uses peak detection to identify heart beats
    # Returns heart rate in BPM or None if calculation fails

    if len(ir_data) < 100:  # Need minimum data points
        return None
        
    # Process the IR signal for better peak detection
    ir_normalized = ir_data - np.mean(ir_data)
    ir_normalized = ir_normalized / np.max(np.abs(ir_normalized))
    
    # Find peaks (heart beats) with minimum spacing
    peaks, _ = find_peaks(ir_normalized, distance=int(sample_rate/2))
    
    if len(peaks) < 2:  # Need at least 2 peaks to calculate rate
        return None
        
    # Calculate heart rate from peak intervals
    peak_intervals = np.diff(peaks)
    mean_interval = np.mean(peak_intervals)
    bpm = 60 * sample_rate / mean_interval
    
    # Validate heart rate is within physiological limits
    if bpm < 40 or bpm > 180:
        return None
        
    return bpm

def calculate_hr_spo2(red_data, ir_data):

    # Calculates both SpO2 and heart rate from sensor data
    # Uses a ratio of RED/IR signals for SpO2 calculation

    if len(red_data) < 100:  # Check for sufficient data
        return None, None
        
    # Verify finger presence
    if not detect_finger(np.mean(red_data), np.mean(ir_data)):
        return None, None
    
    # Calculate SpO2 using R ratio method
    red_normalized = red_data / np.mean(red_data)
    ir_normalized = ir_data / np.mean(ir_data)
    r_ratio = np.std(red_normalized) / np.std(ir_normalized)
    spo2 = 110 - 25 * r_ratio
    
    # Get heart rate from IR signal
    heart_rate = calculate_heart_rate(ir_data)
    
    # Validate SpO2 is within physiological range
    if spo2 > 100 or spo2 < 80:
        spo2 = None
        
    return spo2, heart_rate

def upload_reading(reading):
       
    # Upload validated readings to MongoDB
    # Returns True if successful, False if upload fails

    try:
        result = collection.insert_one(reading)
        print("Reading uploaded successfully to MongoDB")
        return True
    except Exception as e:
        print(f"Error uploading to MongoDB: {e}")
        return False

# Main Program Execution
print("Initializing sensor")
sensor = max30102.MAX30102()

# Setup data collection parameters
WINDOW_SIZE = 100  # Number of samples to collect before processing
red_buffer = []    # Buffer for RED sensor data
ir_buffer = []     # Buffer for IR sensor data

print("Starting new monitoring session")

# Initial temperature measurement
while True:
    temp_data = get_temperature_reading()
    if temp_data is not None:
        break
    print("Let's try again when you're ready.")

print("\nTemperature reading successful!")
ready = input("Ready to measure SpO2 and heart rate? (y/n): ").lower()
if ready != 'y':
    print("Okay, exiting program. Run again when you're ready.")
    client.close()
    exit()

# Main monitoring loop
print("\nPlace your finger on the sensor for SpO2 and heart rate")
try:
    while True:
        # Get new sensor readings
        red, ir = sensor.read_sequential()
        
        if detect_finger(red, ir):
            # Add new readings to buffers
            red_buffer.append(red)
            ir_buffer.append(ir)
            
            # Maintain fixed buffer size
            red_buffer = red_buffer[-WINDOW_SIZE:]
            ir_buffer = ir_buffer[-WINDOW_SIZE:]
            
            # Process data when the buffer is full
            if len(red_buffer) == WINDOW_SIZE:
                spo2, heart_rate = calculate_hr_spo2(np.array(red_buffer), np.array(ir_buffer))
                if spo2 is not None and heart_rate is not None:
                    print(f"SpO2: {spo2:.1f}% | Heart Rate: {heart_rate:.0f} BPM")
                    validation = input("Are these readings correct? (y/n): ").lower()
                    if validation == 'y':
                        # Create a complete reading record
                        reading = {
                            "timestamp": datetime.now(),
                            "spo2": float(f"{spo2:.1f}"),
                            "heart_rate": int(heart_rate),
                            "temperature": temp_data,
                            "validated": True
                        }
                        if upload_reading(reading):
                            print("Reading saved and uploaded.")
                            retry = input("Take another reading? (y/n): ").lower()
                            if retry != 'y':
                                raise KeyboardInterrupt
                        else:
                            print("Failed to upload reading. Continuing monitoring")
                    
                    red_buffer = []  # Reset buffers after processing
                    ir_buffer = []
            
        else:
            red_buffer = []  # Clear buffers when no finger detected
            ir_buffer = []
            print("No finger detected! Please place your finger on the sensor.")
        
        time.sleep(0.1)  # Brief pause between readings

except KeyboardInterrupt:
    print("\nMonitoring session ended.")
    client.close()
