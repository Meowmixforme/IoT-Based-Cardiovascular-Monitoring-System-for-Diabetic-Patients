# James Fothergill v8255920


# You'll need the package from https://github.com/doug-burrell/max30102 and edit it for Python 3





import max30102
import time
import numpy as np
from scipy.signal import find_peaks

def detect_finger(red_value, ir_value):

    # Returns True if a finger is detected based on the IR and Red values

    # Typical values when no finger is present are around 2000-3000
    # When a finger is present, values are typically much higher (>10000)
    MINIMUM_THRESHOLD = 10000
    return (red_value > MINIMUM_THRESHOLD) and (ir_value > MINIMUM_THRESHOLD)

def calculate_heart_rate(ir_data, sample_rate=10):
    """
    Calculate heart rate from IR data
    """
    if len(ir_data) < 100:
        return None
        
    # Detrend and normalize the signal
    ir_normalized = ir_data - np.mean(ir_data)
    ir_normalized = ir_normalized / np.max(np.abs(ir_normalized))
    
    # Find peaks with a minimum distance of 0.5 seconds (assuming 10Hz sample rate)
    peaks, _ = find_peaks(ir_normalized, distance=int(sample_rate/2))
    
    if len(peaks) < 2:
        return None
        
    # Calculate average time between peaks
    peak_intervals = np.diff(peaks)
    mean_interval = np.mean(peak_intervals)
    
    # Convert to BPM
    bpm = 60 * sample_rate / mean_interval
    
    # Basic validation
    if bpm < 40 or bpm > 180:
        return None
        
    return bpm

def calculate_hr_spo2(red_data, ir_data):

   # Calculate SpO2 and heart rate from red and IR data

    if len(red_data) < 100:
        return None, None
        
    # Check if finger is present
    if not detect_finger(np.mean(red_data), np.mean(ir_data)):
        return None, None
    
    # Calculate SpO2
    red_normalized = red_data / np.mean(red_data)
    ir_normalized = ir_data / np.mean(ir_data)
    
    r_ratio = np.std(red_normalized) / np.std(ir_normalized)
    spo2 = 110 - 25 * r_ratio
    
    # Calculate heart rate
    heart_rate = calculate_heart_rate(ir_data)
    
    # Validate SpO2 value
    if spo2 > 100 or spo2 < 80:
        spo2 = None
        
    return spo2, heart_rate

# Initialize sensor
print("Initializing sensor")
sensor = max30102.MAX30102()

# Create arrays to store data
WINDOW_SIZE = 100
red_buffer = []
ir_buffer = []

print("Place your finger on the sensor")
try:
    while True:
        red, ir = sensor.read_sequential()
        
        if detect_finger(red, ir):
            # Add new data to buffers
            red_buffer.append(red)
            ir_buffer.append(ir)
            
            # Keep only the last WINDOW_SIZE samples
            red_buffer = red_buffer[-WINDOW_SIZE:]
            ir_buffer = ir_buffer[-WINDOW_SIZE:]
            
            if len(red_buffer) == WINDOW_SIZE:
                spo2, heart_rate = calculate_hr_spo2(np.array(red_buffer), np.array(ir_buffer))
                output = []
                if spo2 is not None:
                    output.append(f"SpO2: {spo2:.1f}%")
                if heart_rate is not None:
                    output.append(f"Heart Rate: {heart_rate:.0f} BPM")
                if output:
                    print(" | ".join(output))
                else:
                    print("Calculating")
            
            print(f"Raw - Red: {red}, IR: {ir}")
        else:
            red_buffer = []
            ir_buffer = []
            print("No finger detected! Please place your finger on the sensor")
            
        time.sleep(0.1)  # Sample at 10Hz

except KeyboardInterrupt:
    print("\nExiting")
