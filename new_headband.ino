#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <Adafruit_BNO08x.h>

const char* ssid     = "YOUR_WIFI";
const char* password = "YOUR_PASSWORD";
const char* targetIP = "172.20.10.3"; 
const int targetPort = 5005;

#define PULSE_PIN 1
#define BNO08X_RESET -1

Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;
WiFiUDP udp;

void setup() {
  Serial.begin(115200);

  Wire.begin(8, 9);

  if (!bno08x.begin_I2C(0x4B, &Wire) && !bno08x.begin_I2C(0x4A, &Wire)) {
    Serial.println("BNO085 nem található!");
  } else {
    Serial.println("BNO085 kész.");
    bno08x.enableReport(SH2_ROTATION_VECTOR, 20000); 

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
  Serial.println("WiFi Csatlakozva! IP: " + WiFi.localIP().toString());
}

void loop() {

  int pulseRaw = analogRead(PULSE_PIN);

  float qX = 0, qY = 0, qZ = 0, qW = 1.0;
  if (bno08x.getSensorEvent(&sensorValue)) {
    if (sensorValue.sensorId == SH2_ROTATION_VECTOR) {
      qX = sensorValue.un.rotationVector.i;
      qY = sensorValue.un.rotationVector.j;
      qZ = sensorValue.un.rotationVector.k;
      qW = sensorValue.un.rotationVector.real;
    }
  }

  String payload = "{\"pulse\":" + String(pulseRaw) + 
                   ",\"qx\":" + String(qX, 4) + 
                   ",\"qy\":" + String(qY, 4) + 
                   ",\"qz\":" + String(qZ, 4) + 
                   ",\"qw\":" + String(qW, 4) + "}";

  udp.beginPacket(targetIP, targetPort);
  udp.print(payload);
  udp.endPacket();

  delay(50);
