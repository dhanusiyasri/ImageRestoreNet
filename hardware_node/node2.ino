/*
  ============================================================
              MINEGUARD AI - NODE 2
              REMOTE SENSOR NODE
  ============================================================

  Sensors:
    - MPU6500 / MPU6050
    - SW-420 vibration sensor
    - FSR / POT deformation proxy

  Communication:
    - ESP-NOW
    - Channel 1
    - Sends data to Node 1

  LEDs:
    Green = NORMAL
    Blue  = WARNING
    Red   = CRITICAL

  ESP32 Arduino Core 3.x compatible
  ============================================================
*/

#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <esp_mac.h>

// ============================================================
// PIN DEFINITIONS
// ============================================================

#define MPU_SDA         21
#define MPU_SCL         22

#define VIBRATION_PIN   27
#define FSR_PIN         34

#define GREEN_LED       25
#define BLUE_LED        26
#define RED_LED         33
#define BUZZER_PIN      32

// ============================================================
// MPU
// ============================================================

#define MPU_ADDR        0x68

// ============================================================
// ESP-NOW
// ============================================================

#define ESPNOW_CHANNEL  1

// YOUR NODE 1 MAC
uint8_t NODE1_MAC[] = {
  0x88, 0x57, 0x21, 0x79, 0xC6, 0x50
};

// ============================================================
// PACKET IDENTIFICATION
// ============================================================

#define PACKET_MAGIC    0x4D474E32UL

// ============================================================
// NODE 2 PACKET
// IMPORTANT:
// NODE 1 MUST USE EXACTLY THE SAME STRUCTURE
// ============================================================

typedef struct __attribute__((packed)) {

  uint32_t magic;

  uint32_t nodeId;
  uint32_t packetId;
  uint32_t timestamp;

  float accelX;
  float accelY;
  float accelZ;
  float accelMagnitude;

  float gyroX;
  float gyroY;
  float gyroZ;
  float gyroMagnitude;

  float roll;
  float pitch;

  uint16_t fsrRaw;
  float fsrVoltage;
  float fsrResistance;

  bool vibrationState;
  uint32_t vibrationEvents;

  uint8_t alertLevel;

} Node2Packet;

// ============================================================
// ALERT LEVEL
// ============================================================

#define NORMAL_LEVEL     0
#define WARNING_LEVEL    1
#define CRITICAL_LEVEL   2

// ============================================================
// VARIABLES
// ============================================================

uint32_t packetId = 0;
uint32_t vibrationEvents = 0;

bool previousVibration = false;

float fsrBaseline = 0;

unsigned long lastSendTime = 0;

const unsigned long SEND_INTERVAL = 1000;

// ============================================================
// MPU VARIABLES
// ============================================================

float accelX;
float accelY;
float accelZ;

float gyroX;
float gyroY;
float gyroZ;

float accelMagnitude;
float gyroMagnitude;

float roll;
float pitch;

// ============================================================
// MPU FUNCTIONS
// ============================================================

void writeMPU(byte reg, byte data)
{
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(data);
  Wire.endTransmission();
}

void initializeMPU()
{
  Wire.begin(MPU_SDA, MPU_SCL);

  delay(100);

  // Wake MPU
  writeMPU(0x6B, 0x00);

  // Accelerometer ±2g
  writeMPU(0x1C, 0x00);

  // Gyroscope ±250°/s
  writeMPU(0x1B, 0x00);

  delay(100);

  Serial.println("MPU initialized");
}

void readMPU()
{
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission(false);

  Wire.requestFrom(MPU_ADDR, 14);

  if (Wire.available() < 14)
    return;

  int16_t rawAx = (Wire.read() << 8) | Wire.read();
  int16_t rawAy = (Wire.read() << 8) | Wire.read();
  int16_t rawAz = (Wire.read() << 8) | Wire.read();

  // Temperature
  Wire.read();
  Wire.read();

  int16_t rawGx = (Wire.read() << 8) | Wire.read();
  int16_t rawGy = (Wire.read() << 8) | Wire.read();
  int16_t rawGz = (Wire.read() << 8) | Wire.read();

  // ±2g => 16384 LSB/g
  accelX = rawAx / 16384.0;
  accelY = rawAy / 16384.0;
  accelZ = rawAz / 16384.0;

  // ±250°/s => 131 LSB/(°/s)
  gyroX = rawGx / 131.0;
  gyroY = rawGy / 131.0;
  gyroZ = rawGz / 131.0;

  accelMagnitude =
    sqrt(
      accelX * accelX +
      accelY * accelY +
      accelZ * accelZ
    );

  gyroMagnitude =
    sqrt(
      gyroX * gyroX +
      gyroY * gyroY +
      gyroZ * gyroZ
    );

  roll =
    atan2(accelY, accelZ) * 180.0 / PI;

  pitch =
    atan2(
      -accelX,
      sqrt(
        accelY * accelY +
        accelZ * accelZ
      )
    ) * 180.0 / PI;
}

// ============================================================
// FSR
// ============================================================

uint16_t fsrRaw;
float fsrVoltage;
float fsrResistance;

void readFSR()
{
  fsrRaw = analogRead(FSR_PIN);

  fsrVoltage =
    (fsrRaw / 4095.0) * 3.3;

  // 10k reference resistor
  if (fsrVoltage > 0.01 &&
      fsrVoltage < 3.29)
  {
    fsrResistance =
      10000.0 *
      (3.3 - fsrVoltage) /
      fsrVoltage;
  }
  else
  {
    fsrResistance = 0;
  }
}

// ============================================================
// VIBRATION
// ============================================================

bool vibrationState;

void readVibration()
{
  vibrationState =
    digitalRead(VIBRATION_PIN);

  // Count only rising edges
  if (vibrationState && !previousVibration)
  {
    vibrationEvents++;
  }

  previousVibration =
    vibrationState;
}

// ============================================================
// ALERT CALCULATION
// ============================================================

uint8_t calculateAlertLevel()
{
  int warningScore = 0;
  int criticalScore = 0;

  // -----------------------------
  // TILT
  // -----------------------------

  float maxTilt =
    max(abs(roll), abs(pitch));

  if (maxTilt >= 5.0)
  {
    criticalScore++;
  }
  else if (maxTilt >= 2.0)
  {
    warningScore++;
  }

  // -----------------------------
  // ACCELERATION DEVIATION
  // -----------------------------

  float accelDeviation =
    abs(accelMagnitude - 1.0);

  if (accelDeviation >= 0.30)
  {
    criticalScore++;
  }
  else if (accelDeviation >= 0.15)
  {
    warningScore++;
  }

  // -----------------------------
  // VIBRATION
  // -----------------------------

  if (vibrationState)
  {
    warningScore++;
  }

  // -----------------------------
  // FSR CHANGE
  // -----------------------------

  if (fsrBaseline > 10)
  {
    float fsrChange =
      abs(
        ((float)fsrRaw - fsrBaseline) /
        fsrBaseline
      ) * 100.0;

    if (fsrChange >= 40.0)
    {
      criticalScore++;
    }
    else if (fsrChange >= 20.0)
    {
      warningScore++;
    }
  }

  // -----------------------------
  // FINAL ALERT
  // -----------------------------

  if (criticalScore >= 1)
    return CRITICAL_LEVEL;

  if (warningScore >= 1)
    return WARNING_LEVEL;

  return NORMAL_LEVEL;
}

// ============================================================
// LED CONTROL
// ============================================================

void setLEDs(uint8_t level)
{
  digitalWrite(GREEN_LED, LOW);
  digitalWrite(BLUE_LED, LOW);
  digitalWrite(RED_LED, LOW);

  if (level == NORMAL_LEVEL)
  {
    digitalWrite(GREEN_LED, HIGH);
  }
  else if (level == WARNING_LEVEL)
  {
    digitalWrite(BLUE_LED, HIGH);
  }
  else
  {
    digitalWrite(RED_LED, HIGH);
  }
}

// ============================================================
// BUZZER
// ============================================================

void updateBuzzer(uint8_t level)
{
  if (level == CRITICAL_LEVEL)
  {
    tone(BUZZER_PIN, 2000, 150);
  }
  else
  {
    noTone(BUZZER_PIN);
  }
}

// ============================================================
// ESP-NOW SEND CALLBACK
// ESP32 CORE 3.x
// ============================================================

void onDataSent(
  const wifi_tx_info_t *info,
  esp_now_send_status_t status
)
{
  Serial.print("ESP-NOW: ");

  if (status == ESP_NOW_SEND_SUCCESS)
  {
    Serial.println("SUCCESS");
  }
  else
  {
    Serial.println("FAILED");
  }
}

// ============================================================
// INITIALIZE ESP-NOW
// ============================================================

bool initializeESPNow()
{
  WiFi.mode(WIFI_STA);

  delay(200);

  // Force channel 1
  esp_err_t channelResult =
    esp_wifi_set_channel(
      ESPNOW_CHANNEL,
      WIFI_SECOND_CHAN_NONE
    );

  if (channelResult != ESP_OK)
  {
    Serial.println("Failed to set WiFi channel");
    return false;
  }

  Serial.print("Node 2 MAC: ");
  Serial.println(WiFi.macAddress());

  if (esp_now_init() != ESP_OK)
  {
    Serial.println("ESP-NOW initialization failed");
    return false;
  }

  esp_now_register_send_cb(onDataSent);

  // Add Node 1
  esp_now_peer_info_t peerInfo = {};

  memcpy(
    peerInfo.peer_addr,
    NODE1_MAC,
    6
  );

  peerInfo.channel = ESPNOW_CHANNEL;
  peerInfo.encrypt = false;

  if (esp_now_is_peer_exist(NODE1_MAC))
  {
    esp_now_del_peer(NODE1_MAC);
  }

  if (esp_now_add_peer(&peerInfo) != ESP_OK)
  {
    Serial.println("Failed to add Node 1");
    return false;
  }

  Serial.println("Node 1 peer added");
  Serial.println("ESP-NOW ready");

  return true;
}

// ============================================================
// SEND SENSOR PACKET
// ============================================================

void sendSensorData(uint8_t alertLevel)
{
  Node2Packet packet;

  memset(&packet, 0, sizeof(packet));

  packet.magic = PACKET_MAGIC;

  packet.nodeId = 2;
  packet.packetId = packetId++;
  packet.timestamp = millis();

  packet.accelX = accelX;
  packet.accelY = accelY;
  packet.accelZ = accelZ;
  packet.accelMagnitude = accelMagnitude;

  packet.gyroX = gyroX;
  packet.gyroY = gyroY;
  packet.gyroZ = gyroZ;
  packet.gyroMagnitude = gyroMagnitude;

  packet.roll = roll;
  packet.pitch = pitch;

  packet.fsrRaw = fsrRaw;
  packet.fsrVoltage = fsrVoltage;
  packet.fsrResistance = fsrResistance;

  packet.vibrationState = vibrationState;
  packet.vibrationEvents = vibrationEvents;

  packet.alertLevel = alertLevel;

  esp_err_t result =
    esp_now_send(
      NODE1_MAC,
      (uint8_t *)&packet,
      sizeof(packet)
    );

  Serial.println();
  Serial.println("----- NODE 2 DATA -----");

  Serial.print("Packet: ");
  Serial.println(packet.packetId);

  Serial.print("Accel: ");
  Serial.print(accelMagnitude, 3);
  Serial.println(" g");

  Serial.print("Roll: ");
  Serial.print(roll, 2);
  Serial.print("  Pitch: ");
  Serial.println(pitch, 2);

  Serial.print("FSR: ");
  Serial.println(fsrRaw);

  Serial.print("Vibration: ");
  Serial.println(vibrationState ? "YES" : "NO");

  Serial.print("Alert: ");

  if (alertLevel == NORMAL_LEVEL)
    Serial.println("NORMAL");
  else if (alertLevel == WARNING_LEVEL)
    Serial.println("WARNING");
  else
    Serial.println("CRITICAL");

  Serial.print("Send: ");

  if (result == ESP_OK)
    Serial.println("OK");
  else
    Serial.println("ERROR");
}

// ============================================================
// SETUP
// ============================================================

void setup()
{
  Serial.begin(115200);

  delay(1500);

  Serial.println();
  Serial.println("================================");
  Serial.println("       MINEGUARD AI NODE 2");
  Serial.println("================================");

  // Pins
  pinMode(
    VIBRATION_PIN,
    INPUT
  );

  pinMode(
    GREEN_LED,
    OUTPUT
  );

  pinMode(
    BLUE_LED,
    OUTPUT
  );

  pinMode(
    RED_LED,
    OUTPUT
  );

  pinMode(
    BUZZER_PIN,
    OUTPUT
  );

  // ADC
  analogReadResolution(12);

  // Startup LED test
  digitalWrite(GREEN_LED, HIGH);
  delay(300);
  digitalWrite(GREEN_LED, LOW);

  digitalWrite(BLUE_LED, HIGH);
  delay(300);
  digitalWrite(BLUE_LED, LOW);

  digitalWrite(RED_LED, HIGH);
  delay(300);
  digitalWrite(RED_LED, LOW);

  // MPU
  initializeMPU();

  // Initial FSR baseline
  delay(500);

  long fsrSum = 0;

  for (int i = 0; i < 20; i++)
  {
    fsrSum += analogRead(FSR_PIN);
    delay(20);
  }

  fsrBaseline =
    fsrSum / 20.0;

  Serial.print("FSR baseline: ");
  Serial.println(fsrBaseline);

  // ESP-NOW
  if (!initializeESPNow())
  {
    Serial.println("ESP-NOW FAILED");
  }

  Serial.println("================================");
  Serial.println("NODE 2 STARTED");
  Serial.println("================================");
}

// ============================================================
// LOOP
// ============================================================

void loop()
{
  readMPU();
  readFSR();
  readVibration();

  uint8_t alertLevel =
    calculateAlertLevel();

  setLEDs(alertLevel);

  updateBuzzer(alertLevel);

  if (millis() - lastSendTime >= SEND_INTERVAL)
  {
    lastSendTime = millis();

    sendSensorData(alertLevel);
  }

  delay(50);
}