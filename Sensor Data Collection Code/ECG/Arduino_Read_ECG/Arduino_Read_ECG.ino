// James Fothergill v8255920

// Filter Configuration Constants
const int WINDOW_SIZE = 3;        // Keep small window
const float ALPHA = 0.5;          // More direct signal following
const float NOTCH_ALPHA = 0.08;   // Very light filtering
const float BASELINE_FACTOR = 0.0; // Remove baseline correction

// Variables for storing previous values
float prevFiltered = 0;
float prevNotch = 0;
float notchPrev = 0;
float movingAverage[WINDOW_SIZE];
int windowIndex = 0;

void setupFilters() {
  for(int i = 0; i < WINDOW_SIZE; i++) {
    movingAverage[i] = 0;
  }
}

float filterECG(float rawValue) {
  movingAverage[windowIndex] = rawValue;
  windowIndex = (windowIndex + 1) % WINDOW_SIZE;
  
  float sum = 0;
  for(int i = 0; i < WINDOW_SIZE; i++) {
    sum += movingAverage[i];
  }
  float averaged = sum / WINDOW_SIZE;

  float lowPassed = ALPHA * rawValue + (1 - ALPHA) * prevFiltered;
  prevFiltered = lowPassed;

  float notched = NOTCH_ALPHA * notchPrev + 
                 NOTCH_ALPHA * (lowPassed - prevNotch);
  prevNotch = lowPassed;
  notchPrev = notched;
  
  return lowPassed;
}

void setup() {
  Serial.begin(9600);
  pinMode(10, INPUT); // LO +
  pinMode(11, INPUT); // LO -
  setupFilters();
  Serial.println("Arduino ECG Collection Started");
}

void loop() {
  if((digitalRead(10) == 1)||(digitalRead(11) == 1)){
    Serial.println("!"); // Leads off detection
    delay(20);
    return;
  }

  // Read and process values
  int rawValue = analogRead(A0);
  float filteredValue = filterECG(rawValue);

  // Create JSON string for each reading
  Serial.print("{\"raw_value\":");
  Serial.print(rawValue);
  Serial.print(",\"filtered_value\":");
  Serial.print(filteredValue);
  Serial.println("}");
  
  delay(20); // 50Hz sampling rate
}