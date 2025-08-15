// James Fothergill v8255920

// Filter Configuration Constants
const int WINDOW_SIZE = 3;        // Number of samples to average - optimised for clear ECG peaks
const float ALPHA = 0.5;          // Filter strength: higher value = more responsive to changes
const float NOTCH_ALPHA = 0.08;   // Reduces electrical interference from power lines
const float BASELINE_FACTOR = 0.0; // Baseline drift correction - currently disabled

// Variables to store previous measurements
float prevFiltered = 0;           // Remembers last filtered ECG value
float prevNotch = 0;              // Previous input to interference filter
float notchPrev = 0;              // Previous output from interference filter
float movingAverage[WINDOW_SIZE]; // Stores recent ECG values for smoothing
int windowIndex = 0;              // Tracks position in moving average storage


 // Prepares the filtering system
 // Clears any old values to ensure a clean start

void setupFilters() {
  for(int i = 0; i < WINDOW_SIZE; i++) {
    movingAverage[i] = 0;  // Reset all stored values to zero
  }
}


 // Processes raw ECG signal to remove noise
 // Uses three filtering stages for best results
 // 
 // @param rawValue Unprocessed ECG reading from sensor
 // @return Cleaned ECG signal with clear heart beats

float filterECG(float rawValue) {
  // Stage 1: Smooth out sudden spikes
  // Takes average of recent readings for stability
  movingAverage[windowIndex] = rawValue;
  windowIndex = (windowIndex + 1) % WINDOW_SIZE;  // Move to next storage position
  
  float sum = 0;
  for(int i = 0; i < WINDOW_SIZE; i++) {
    sum += movingAverage[i];
  }
  float averaged = sum / WINDOW_SIZE;  // Calculate average

  // Stage 2: Main noise reduction
  // Balances between current and previous readings
  float lowPassed = ALPHA * rawValue + (1 - ALPHA) * prevFiltered;
  prevFiltered = lowPassed;  // Store for next calculation

  // Stage 3: Remove electrical interference
  // Specifically targets unwanted electrical noise
  float notched = NOTCH_ALPHA * notchPrev + 
                 NOTCH_ALPHA * (lowPassed - prevNotch);
  prevNotch = lowPassed;
  notchPrev = notched;
  
  // Return cleaned signal optimised for heart beat detection
  return lowPassed;  // Using main filtered signal for best results
}


 // Initial setup of ECG monitoring system
 // Configures all necessary components
void setup() {
  Serial.begin(9600);            // Start communication with computer
  pinMode(10, INPUT);            // Setup electrode detection (positive)
  pinMode(11, INPUT);            // Setup electrode detection (negative)
  setupFilters();                // Prepare signal cleaning system
}


 // Main monitoring loop
 // Continuously reads and processes ECG signal
 // Runs 50 times per second for accurate monitoring

void loop() {
  // Check if electrodes are properly connected
  if((digitalRead(10) == 1)||(digitalRead(11) == 1)){
    Serial.println('!');         // Alert: electrodes not properly connected
  }
  else{
    // Read and process ECG signal
    int rawValue = analogRead(A0);  // Get reading from ECG sensor
    
    // Clean up the signal
    float filteredValue = filterECG(rawValue);
    
    // Send both original and cleaned signals to computer
    Serial.print(rawValue);
    Serial.print(",");
    Serial.println(filteredValue);
  }
  
  // Wait for next reading (50 samples per second)
  delay(20);
}
