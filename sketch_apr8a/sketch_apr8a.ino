#include <Wire.h>
#include <hd44780.h>
#include <hd44780ioClass/hd44780_I2Cexp.h>
#include <Adafruit_ADXL345_U.h>
#include <Adafruit_Sensor.h>

// 🔹 LCD + Sensor
hd44780_I2Cexp lcd;
Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);

// 🔹 Variables
float x, y, z;
String receivedData = "";

int buzzerPin = 8;

void setup() {
  Serial.begin(9600);

  // 🔹 Initialize LCD (AUTO detect)
  int status = lcd.begin(16, 2);
  if (status) {
    Serial.print("LCD failed, status=");
    Serial.println(status);
    while (1);
  }

  lcd.setCursor(0,0);
  lcd.print("ML Posture Sys");
  delay(2000);
  lcd.clear();

  // 🔹 Initialize ADXL345
  if (!accel.begin()) {
    lcd.setCursor(0,0);
    lcd.print("ADXL Error!");
    while (1);
  }

  pinMode(buzzerPin, OUTPUT);
}

void loop() {

  // 🔹 Read sensor
  sensors_event_t event;
  accel.getEvent(&event);

  x = event.acceleration.x;
  y = event.acceleration.y;
  z = event.acceleration.z;

  // 🔹 Send data to Python
  Serial.print(x);
  Serial.print(",");
  Serial.print(y);
  Serial.print(",");
  Serial.println(z);

  delay(200);   // sync with Python

  // 🔹 Receive ML result
  if (Serial.available() > 0) {

    receivedData = Serial.readStringUntil('\n');
    receivedData.trim();

    // Debug (optional)
    Serial.print("Received: ");
    Serial.println(receivedData);

    lcd.clear();
    lcd.setCursor(0,0);
    lcd.print("Posture:");

    if (receivedData == "BAD") {
      lcd.setCursor(0,1);
      lcd.print("Fix Posture!");
      digitalWrite(buzzerPin, HIGH);
    } 
    else if (receivedData == "GOOD") {
      lcd.setCursor(0,1);
      lcd.print("Good Posture");
      digitalWrite(buzzerPin, LOW);
    }
  }
}