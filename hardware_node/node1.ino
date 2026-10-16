/*
  ============================================================
        MINEGUARD AI - NODE 1 GATEWAY
        ESP32 Mine Subsidence Monitoring Prototype
  ============================================================

  NODE 1:
    - Reads its own MPU6050
    - Reads its own FSR/POT
    - Reads its own vibration sensor
    - Receives Node 2 data through ESP-NOW
    - Combines Node 1 + Node 2 alert levels
    - OLED display
    - SD card logging
    - Green / Blue / Red LEDs
    - Buzzer

  ESP-NOW:
    Channel = 1

  IMPORTANT:
    Node 2 packet structure MUST be identical.
*/


#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>


// ============================================================
// ESP-NOW SETTINGS
// ============================================================

#define ESPNOW_CHANNEL 1

#define PACKET_MAGIC 0x4D474E32UL

#define NORMAL_LEVEL    0
#define WARNING_LEVEL   1
#define CRITICAL_LEVEL  2


// ============================================================
// NODE 1 SENSOR PINS
// ============================================================

#define MPU_SDA 21
#define MPU_SCL 22

#define VIBRATION_PIN 27
#define FSR_PIN       34

#define BUZZER_PIN    25

#define RED_LED_PIN   32
#define BLUE_LED_PIN  33
#define GREEN_LED_PIN 26


// ============================================================
// OLED
// ============================================================

#define OLED_SDA 16
#define OLED_SCL 17

#define OLED_WIDTH 128
#define OLED_HEIGHT 64

#define OLED_ADDRESS 0x3C


// ============================================================
// SD CARD
// ============================================================

#define SD_CS   5
#define SD_SCK  18
#define SD_MISO 19
#define SD_MOSI 23


// ============================================================
// OBJECTS
// ============================================================

Adafruit_MPU6050 mpu;

Adafruit_SSD1306 display(
  OLED_WIDTH,
  OLED_HEIGHT,
  &Wire1,
  -1
);


// ============================================================
// NODE 2 PACKET
// MUST MATCH NODE 2 EXACTLY
// ============================================================

typedef struct __attribute__((packed))
{
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
// NODE 1 LOCAL SENSOR DATA
// ============================================================

float localAccelX = 0;
float localAccelY = 0;
float localAccelZ = 0;

float localAccelMagnitude = 0;

float localGyroX = 0;
float localGyroY = 0;
float localGyroZ = 0;

float localGyroMagnitude = 0;

float localRoll = 0;
float localPitch = 0;

uint16_t localFsrRaw = 0;

float localFsrVoltage = 0;

float localFsrResistance = 0;

bool localVibrationState = false;

bool previousVibrationState = false;

uint32_t localVibrationEvents = 0;

float initialFsr = 0;


// ============================================================
// NODE 2 DATA
// ============================================================

Node2Packet node2Data;

volatile bool node2DataAvailable = false;

portMUX_TYPE node2Mux =
  portMUX_INITIALIZER_UNLOCKED;


// ============================================================
// NODE 2 STATUS
// ============================================================

uint32_t lastNode2PacketTime = 0;

uint32_t lastProcessedPacket = 0;

bool node2Connected = false;

#define NODE2_TIMEOUT 5000


// ============================================================
// TIMERS
// ============================================================

unsigned long lastSensorRead = 0;

unsigned long lastOLEDUpdate = 0;

unsigned long lastSDLog = 0;

unsigned long lastStatusPrint = 0;


// ============================================================
// INTERVALS
// ============================================================

#define SENSOR_INTERVAL 100

#define OLED_INTERVAL 500

#define SD_INTERVAL 1000

#define STATUS_INTERVAL 1000


// ============================================================
// ESP-NOW RECEIVE CALLBACK
// ============================================================

void onDataReceive(
  const esp_now_recv_info_t *info,
  const uint8_t *incomingData,
  int len
)
{
  if (len != sizeof(Node2Packet))
  {
    return;
  }

  Node2Packet received;

  memcpy(
    &received,
    incomingData,
    sizeof(Node2Packet)
  );

  if (received.magic != PACKET_MAGIC)
  {
    return;
  }

  portENTER_CRITICAL_ISR(&node2Mux);

  memcpy(
    &node2Data,
    &received,
    sizeof(Node2Packet)
  );

  node2DataAvailable = true;

  portEXIT_CRITICAL_ISR(&node2Mux);
}


// ============================================================
// PRINT NODE 1 MAC
// ============================================================

void printHardwareMAC()
{
  Serial.print("Node 1 MAC: ");

  Serial.println(
    WiFi.macAddress()
  );
}


// ============================================================
// INITIALIZE ESP-NOW
// ============================================================

bool initializeESPNow()
{
  Serial.println();
  Serial.println(
    "Initializing ESP-NOW..."
  );

  WiFi.mode(WIFI_STA);

  delay(100);

  esp_err_t result =
    esp_wifi_set_channel(
      ESPNOW_CHANNEL,
      WIFI_SECOND_CHAN_NONE
    );

  if (result != ESP_OK)
  {
    Serial.println(
      "ERROR: Failed to set WiFi channel"
    );

    return false;
  }

  if (esp_now_init() != ESP_OK)
  {
    Serial.println(
      "ERROR: ESP-NOW initialization failed"
    );

    return false;
  }

  esp_now_register_recv_cb(
    onDataReceive
  );

  Serial.println(
    "ESP-NOW initialized"
  );

  Serial.print(
    "Channel: "
  );

  Serial.println(
    ESPNOW_CHANNEL
  );

  return true;
}


// ============================================================
// INITIALIZE MPU6050
// ============================================================

bool initializeMPU()
{
  Wire.begin(
    MPU_SDA,
    MPU_SCL
  );

  if (!mpu.begin())
  {
    Serial.println(
      "ERROR: MPU6050 not detected"
    );

    return false;
  }

  Serial.println(
    "MPU6050 initialized"
  );

  mpu.setAccelerometerRange(
    MPU6050_RANGE_8_G
  );

  mpu.setGyroRange(
    MPU6050_RANGE_500_DEG
  );

  mpu.setFilterBandwidth(
    MPU6050_BAND_21_HZ
  );

  return true;
}


// ============================================================
// INITIALIZE OLED
// ============================================================

bool initializeOLED()
{
  Wire1.begin(
    OLED_SDA,
    OLED_SCL
  );

  if (!display.begin(
        SSD1306_SWITCHCAPVCC,
        OLED_ADDRESS
      ))
  {
    Serial.println(
      "OLED initialization failed"
    );

    return false;
  }

  display.clearDisplay();

  display.setTextColor(
    SSD1306_WHITE
  );

  display.setTextSize(1);

  display.setCursor(
    0,
    0
  );

  display.println(
    "MINEGUARD AI"
  );

  display.println();

  display.println(
    "NODE 1 GATEWAY"
  );

  display.println();

  display.println(
    "Initializing..."
  );

  display.display();

  return true;
}


// ============================================================
// INITIALIZE SD CARD
// ============================================================

bool initializeSD()
{
  SPI.begin(
    SD_SCK,
    SD_MISO,
    SD_MOSI,
    SD_CS
  );

  if (!SD.begin(SD_CS))
  {
    Serial.println(
      "SD card initialization failed"
    );

    return false;
  }

  Serial.println(
    "SD card initialized"
  );

  if (!SD.exists(
        "/mineguard.csv"
      ))
  {
    File file =
      SD.open(
        "/mineguard.csv",
        FILE_WRITE
      );

    if (file)
    {
      file.println(
        "time,"
        "node1_accelX,"
        "node1_accelY,"
        "node1_accelZ,"
        "node1_accelMagnitude,"
        "node1_gyroX,"
        "node1_gyroY,"
        "node1_gyroZ,"
        "node1_gyroMagnitude,"
        "node1_roll,"
        "node1_pitch,"
        "node1_fsrRaw,"
        "node1_fsrVoltage,"
        "node1_fsrResistance,"
        "node1_vibration,"
        "node1_vibrationEvents,"
        "node1_alert,"
        "node2_packet,"
        "node2_accelX,"
        "node2_accelY,"
        "node2_accelZ,"
        "node2_accelMagnitude,"
        "node2_gyroX,"
        "node2_gyroY,"
        "node2_gyroZ,"
        "node2_gyroMagnitude,"
        "node2_roll,"
        "node2_pitch,"
        "node2_fsrRaw,"
        "node2_fsrVoltage,"
        "node2_fsrResistance,"
        "node2_vibration,"
        "node2_vibrationEvents,"
        "node2_alert,"
        "combined_alert"
      );

      file.close();
    }
  }

  return true;
}


// ============================================================
// READ NODE 1 MPU6050
// ============================================================

void readLocalMPU()
{
  sensors_event_t accel;
  sensors_event_t gyro;
  sensors_event_t temp;

  mpu.getEvent(
    &accel,
    &gyro,
    &temp
  );

  // Acceleration in g

  localAccelX =
    accel.acceleration.x / 9.80665;

  localAccelY =
    accel.acceleration.y / 9.80665;

  localAccelZ =
    accel.acceleration.z / 9.80665;

  localAccelMagnitude =
    sqrt(
      localAccelX * localAccelX +
      localAccelY * localAccelY +
      localAccelZ * localAccelZ
    );

  // Gyroscope

  localGyroX =
    gyro.gyro.x;

  localGyroY =
    gyro.gyro.y;

  localGyroZ =
    gyro.gyro.z;

  localGyroMagnitude =
    sqrt(
      localGyroX * localGyroX +
      localGyroY * localGyroY +
      localGyroZ * localGyroZ
    );

  // Roll

  localRoll =
    atan2(
      localAccelY,
      localAccelZ
    )
    * 180.0 / PI;

  // Pitch

  localPitch =
    atan2(
      -localAccelX,
      sqrt(
        localAccelY * localAccelY +
        localAccelZ * localAccelZ
      )
    )
    * 180.0 / PI;
}


// ============================================================
// READ NODE 1 FSR
// ============================================================

void readLocalFSR()
{
  localFsrRaw =
    analogRead(
      FSR_PIN
    );

  localFsrVoltage =
    (
      localFsrRaw /
      4095.0
    )
    * 3.3;

  if (
    localFsrVoltage > 0.01 &&
    localFsrVoltage < 3.29
  )
  {
    localFsrResistance =
      10000.0 *
      (
        3.3 /
        localFsrVoltage -
        1.0
      );
  }
  else
  {
    localFsrResistance =
      0;
  }

  if (initialFsr == 0)
  {
    initialFsr =
      localFsrRaw;
  }
}


// ============================================================
// READ NODE 1 VIBRATION
// ============================================================

void readLocalVibration()
{
  localVibrationState =
    digitalRead(
      VIBRATION_PIN
    );

  if (
    localVibrationState &&
    !previousVibrationState
  )
  {
    localVibrationEvents++;
  }

  previousVibrationState =
    localVibrationState;
}


// ============================================================
// CALCULATE NODE 1 ALERT
// ============================================================

uint8_t calculateLocalAlert()
{
  uint8_t level =
    NORMAL_LEVEL;

  float maxTilt =
    max(
      abs(localRoll),
      abs(localPitch)
    );

  if (maxTilt >= 5.0)
  {
    level =
      CRITICAL_LEVEL;
  }
  else if (maxTilt >= 2.0)
  {
    level =
      max(
        level,
        (uint8_t)WARNING_LEVEL
      );
  }

  float accelDeviation =
    abs(
      localAccelMagnitude - 1.0
    );

  if (accelDeviation >= 0.30)
  {
    level =
      CRITICAL_LEVEL;
  }
  else if (accelDeviation >= 0.15)
  {
    level =
      max(
        level,
        (uint8_t)WARNING_LEVEL
      );
  }

  if (localVibrationState)
  {
    level =
      max(
        level,
        (uint8_t)WARNING_LEVEL
      );
  }

  if (initialFsr > 0)
  {
    float fsrChange =
      abs(
        (
          (float)localFsrRaw -
          initialFsr
        )
        /
        initialFsr
      );

    if (fsrChange >= 0.40)
    {
      level =
        CRITICAL_LEVEL;
    }
    else if (fsrChange >= 0.20)
    {
      level =
        max(
          level,
          (uint8_t)WARNING_LEVEL
        );
    }
  }

  return level;
}


// ============================================================
// GET NODE 2 DATA SAFELY
// ============================================================

bool getNode2Data(
  Node2Packet &data
)
{
  bool available;

  portENTER_CRITICAL(
    &node2Mux
  );

  available =
    node2DataAvailable;

  if (available)
  {
    memcpy(
      &data,
      &node2Data,
      sizeof(Node2Packet)
    );
  }

  portEXIT_CRITICAL(
    &node2Mux
  );

  return available;
}


// ============================================================
// PROCESS NODE 2 DATA
// ============================================================

void processNode2Data()
{
  Node2Packet data;

  if (!getNode2Data(data))
  {
    return;
  }

  if (
    data.packetId ==
    lastProcessedPacket
  )
  {
    return;
  }

  lastProcessedPacket =
    data.packetId;

  lastNode2PacketTime =
    millis();

  node2Connected =
    true;


  Serial.println();
  Serial.println(
    "================================"
  );

  Serial.println(
    "       NODE 2 DATA RECEIVED"
  );

  Serial.println(
    "================================"
  );

  Serial.print(
    "Packet ID       : "
  );

  Serial.println(
    data.packetId
  );

  Serial.print(
    "Timestamp       : "
  );

  Serial.println(
    data.timestamp
  );

  Serial.print(
    "Accel X         : "
  );

  Serial.println(
    data.accelX,
    3
  );

  Serial.print(
    "Accel Y         : "
  );

  Serial.println(
    data.accelY,
    3
  );

  Serial.print(
    "Accel Z         : "
  );

  Serial.println(
    data.accelZ,
    3
  );

  Serial.print(
    "Accel Magnitude : "
  );

  Serial.print(
    data.accelMagnitude,
    3
  );

  Serial.println(
    " g"
  );

  Serial.print(
    "Gyro X          : "
  );

  Serial.println(
    data.gyroX,
    2
  );

  Serial.print(
    "Gyro Y          : "
  );

  Serial.println(
    data.gyroY,
    2
  );

  Serial.print(
    "Gyro Z          : "
  );

  Serial.println(
    data.gyroZ,
    2
  );

  Serial.print(
    "Gyro Magnitude  : "
  );

  Serial.println(
    data.gyroMagnitude,
    2
  );

  Serial.print(
    "Roll            : "
  );

  Serial.println(
    data.roll,
    2
  );

  Serial.print(
    "Pitch           : "
  );

  Serial.println(
    data.pitch,
    2
  );

  Serial.print(
    "FSR Raw         : "
  );

  Serial.println(
    data.fsrRaw
  );

  Serial.print(
    "FSR Voltage     : "
  );

  Serial.println(
    data.fsrVoltage,
    3
  );

  Serial.print(
    "FSR Resistance  : "
  );

  Serial.println(
    data.fsrResistance,
    2
  );

  Serial.print(
    "Vibration       : "
  );

  Serial.println(
    data.vibrationState
      ? "YES"
      : "NO"
  );

  Serial.print(
    "Vibration Events: "
  );

  Serial.println(
    data.vibrationEvents
  );

  Serial.print(
    "Alert Level     : "
  );

  if (
    data.alertLevel ==
    NORMAL_LEVEL
  )
  {
    Serial.println(
      "NORMAL"
    );
  }
  else if (
    data.alertLevel ==
    WARNING_LEVEL
  )
  {
    Serial.println(
      "WARNING"
    );
  }
  else
  {
    Serial.println(
      "CRITICAL"
    );
  }

  Serial.println(
    "================================"
  );
}


// ============================================================
// NODE 2 CONNECTION CHECK
// ============================================================

void checkNode2Connection()
{
  if (
    node2Connected &&
    millis() -
    lastNode2PacketTime >
    NODE2_TIMEOUT
  )
  {
    node2Connected =
      false;

    Serial.println();

    Serial.println(
      "WARNING: NODE 2 CONNECTION LOST"
    );
  }
}


// ============================================================
// COMBINED ALERT
// ============================================================

uint8_t calculateCombinedAlert()
{
  uint8_t localLevel =
    calculateLocalAlert();

  uint8_t node2Level =
    NORMAL_LEVEL;

  Node2Packet data;

  if (
    getNode2Data(data)
  )
  {
    node2Level =
      data.alertLevel;
  }

  uint8_t combined =
    max(
      localLevel,
      node2Level
    );

  if (!node2Connected)
  {
    combined =
      max(
        combined,
        (uint8_t)WARNING_LEVEL
      );
  }

  return combined;
}


// ============================================================
// LED CONTROL
// ============================================================

void setLEDs(
  uint8_t level
)
{
  digitalWrite(
    GREEN_LED_PIN,
    LOW
  );

  digitalWrite(
    BLUE_LED_PIN,
    LOW
  );

  digitalWrite(
    RED_LED_PIN,
    LOW
  );

  if (
    level ==
    NORMAL_LEVEL
  )
  {
    digitalWrite(
      GREEN_LED_PIN,
      HIGH
    );
  }
  else if (
    level ==
    WARNING_LEVEL
  )
  {
    digitalWrite(
      BLUE_LED_PIN,
      HIGH
    );
  }
  else
  {
    digitalWrite(
      RED_LED_PIN,
      HIGH
    );
  }
}


// ============================================================
// BUZZER
// ============================================================

void updateBuzzer(
  uint8_t level
)
{
  static unsigned long lastBeep = 0;

  static bool buzzerState = false;

  if (
    level ==
    NORMAL_LEVEL
  )
  {
    digitalWrite(
      BUZZER_PIN,
      LOW
    );

    buzzerState =
      false;

    return;
  }

  if (
    level ==
    WARNING_LEVEL
  )
  {
    if (
      millis() -
      lastBeep >
      1000
    )
    {
      lastBeep =
        millis();

      buzzerState =
        !buzzerState;

      digitalWrite(
        BUZZER_PIN,
        buzzerState
      );
    }
  }

  else
  {
    if (
      millis() -
      lastBeep >
      250
    )
    {
      lastBeep =
        millis();

      buzzerState =
        !buzzerState;

      digitalWrite(
        BUZZER_PIN,
        buzzerState
      );
    }
  }
}


// ============================================================
// OLED
// ============================================================

void updateOLED(
  uint8_t level
)
{
  display.clearDisplay();

  display.setTextSize(1);

  display.setCursor(
    0,
    0
  );

  display.println(
    "MINEGUARD AI"
  );

  display.setCursor(
    0,
    12
  );

  display.print(
    "STATUS: "
  );

  if (
    level ==
    NORMAL_LEVEL
  )
  {
    display.println(
      "NORMAL"
    );
  }
  else if (
    level ==
    WARNING_LEVEL
  )
  {
    display.println(
      "WARNING"
    );
  }
  else
  {
    display.println(
      "CRITICAL"
    );
  }

  display.setCursor(
    0,
    25
  );

  display.print(
    "NODE 2: "
  );

  if (node2Connected)
  {
    display.println(
      "CONNECTED"
    );
  }
  else
  {
    display.println(
      "OFFLINE"
    );
  }

  display.setCursor(
    0,
    38
  );

  display.print(
    "N1 Tilt: "
  );

  display.print(
    max(
      abs(localRoll),
      abs(localPitch)
    ),
    1
  );

  display.println(
    " deg"
  );

  display.setCursor(
    0,
    50
  );

  display.print(
    "N2 Packet: "
  );

  display.println(
    lastProcessedPacket
  );

  display.display();
}


// ============================================================
// SD LOGGING
// ============================================================

void logToSD(
  uint8_t combinedLevel
)
{
  Node2Packet data;

  bool hasNode2 =
    getNode2Data(data);

  File file =
    SD.open(
      "/mineguard.csv",
      FILE_APPEND
    );

  if (!file)
  {
    return;
  }

  // Time

  file.print(
    millis()
  );

  file.print(",");


  // ==========================================================
  // NODE 1 - ALL ATTRIBUTES
  // ==========================================================

  file.print(
    localAccelX,
    3
  );

  file.print(",");

  file.print(
    localAccelY,
    3
  );

  file.print(",");

  file.print(
    localAccelZ,
    3
  );

  file.print(",");

  file.print(
    localAccelMagnitude,
    3
  );

  file.print(",");

  file.print(
    localGyroX,
    3
  );

  file.print(",");

  file.print(
    localGyroY,
    3
  );

  file.print(",");

  file.print(
    localGyroZ,
    3
  );

  file.print(",");

  file.print(
    localGyroMagnitude,
    3
  );

  file.print(",");

  file.print(
    localRoll,
    2
  );

  file.print(",");

  file.print(
    localPitch,
    2
  );

  file.print(",");

  file.print(
    localFsrRaw
  );

  file.print(",");

  file.print(
    localFsrVoltage,
    3
  );

  file.print(",");

  file.print(
    localFsrResistance,
    2
  );

  file.print(",");

  file.print(
    localVibrationState
  );

  file.print(",");

  file.print(
    localVibrationEvents
  );

  file.print(",");

  file.print(
    calculateLocalAlert()
  );

  file.print(",");


  // ==========================================================
  // NODE 2 - ALL ATTRIBUTES
  // ==========================================================

  if (hasNode2)
  {
    file.print(
      data.packetId
    );

    file.print(",");

    file.print(
      data.accelX,
      3
    );

    file.print(",");

    file.print(
      data.accelY,
      3
    );

    file.print(",");

    file.print(
      data.accelZ,
      3
    );

    file.print(",");

    file.print(
      data.accelMagnitude,
      3
    );

    file.print(",");

    file.print(
      data.gyroX,
      3
    );

    file.print(",");

    file.print(
      data.gyroY,
      3
    );

    file.print(",");

    file.print(
      data.gyroZ,
      3
    );

    file.print(",");

    file.print(
      data.gyroMagnitude,
      3
    );

    file.print(",");

    file.print(
      data.roll,
      2
    );

    file.print(",");

    file.print(
      data.pitch,
      2
    );

    file.print(",");

    file.print(
      data.fsrRaw
    );

    file.print(",");

    file.print(
      data.fsrVoltage,
      3
    );

    file.print(",");

    file.print(
      data.fsrResistance,
      2
    );

    file.print(",");

    file.print(
      data.vibrationState
    );

    file.print(",");

    file.print(
      data.vibrationEvents
    );

    file.print(",");

    file.print(
      data.alertLevel
    );

    file.print(",");
  }
  else
  {
    file.print(
      "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
    );
  }


  // Combined alert

  file.println(
    combinedLevel
  );

  file.close();
}


// ============================================================
// STATUS PRINT
// ============================================================

void printStatus()
{
  Serial.println();

  Serial.println(
    "================================================"
  );

  Serial.println(
    "              NODE 1 STATUS"
  );

  Serial.println(
    "================================================"
  );


  // ==========================================================
  // NODE 1 COMPLETE SENSOR DATA
  // ==========================================================

  Serial.println(
    "NODE 1 SENSOR DATA"
  );

  Serial.println(
    "------------------------------------------------"
  );

  Serial.print(
    "Accel X         : "
  );

  Serial.println(
    localAccelX,
    3
  );

  Serial.print(
    "Accel Y         : "
  );

  Serial.println(
    localAccelY,
    3
  );

  Serial.print(
    "Accel Z         : "
  );

  Serial.println(
    localAccelZ,
    3
  );

  Serial.print(
    "Accel Magnitude : "
  );

  Serial.print(
    localAccelMagnitude,
    3
  );

  Serial.println(
    " g"
  );

  Serial.print(
    "Gyro X          : "
  );

  Serial.println(
    localGyroX,
    3
  );

  Serial.print(
    "Gyro Y          : "
  );

  Serial.println(
    localGyroY,
    3
  );

  Serial.print(
    "Gyro Z          : "
  );

  Serial.println(
    localGyroZ,
    3
  );

  Serial.print(
    "Gyro Magnitude  : "
  );

  Serial.println(
    localGyroMagnitude,
    3
  );

  Serial.print(
    "Roll            : "
  );

  Serial.println(
    localRoll,
    2
  );

  Serial.print(
    "Pitch           : "
  );

  Serial.println(
    localPitch,
    2
  );

  Serial.print(
    "FSR Raw         : "
  );

  Serial.println(
    localFsrRaw
  );

  Serial.print(
    "FSR Voltage     : "
  );

  Serial.println(
    localFsrVoltage,
    3
  );

  Serial.print(
    "FSR Resistance  : "
  );

  Serial.println(
    localFsrResistance,
    2
  );

  Serial.print(
    "Vibration       : "
  );

  Serial.println(
    localVibrationState
      ? "YES"
      : "NO"
  );

  Serial.print(
    "Vibration Events: "
  );

  Serial.println(
    localVibrationEvents
  );


  uint8_t localLevel =
    calculateLocalAlert();


  Serial.print(
    "Alert Level     : "
  );

  if (
    localLevel ==
    NORMAL_LEVEL
  )
  {
    Serial.println(
      "NORMAL"
    );
  }
  else if (
    localLevel ==
    WARNING_LEVEL
  )
  {
    Serial.println(
      "WARNING"
    );
  }
  else
  {
    Serial.println(
      "CRITICAL"
    );
  }


  // ==========================================================
  // NODE 2 STATUS
  // ==========================================================

  Serial.println();

  Serial.println(
    "NODE 2 STATUS"
  );

  Serial.println(
    "------------------------------------------------"
  );

  Serial.print(
    "Connection      : "
  );

  Serial.println(
    node2Connected
      ? "CONNECTED"
      : "DISCONNECTED"
  );

  Serial.print(
    "Last Packet     : "
  );

  Serial.println(
    lastProcessedPacket
  );


  // ==========================================================
  // COMBINED ALERT
  // ==========================================================

  uint8_t level =
    calculateCombinedAlert();


  Serial.println();

  Serial.print(
    "Combined Alert  : "
  );

  if (
    level ==
    NORMAL_LEVEL
  )
  {
    Serial.println(
      "NORMAL"
    );
  }
  else if (
    level ==
    WARNING_LEVEL
  )
  {
    Serial.println(
      "WARNING"
    );
  }
  else
  {
    Serial.println(
      "CRITICAL"
    );
  }


  Serial.println(
    "================================================"
  );
}


// ============================================================
// SETUP
// ============================================================

void setup()
{
  Serial.begin(
    115200
  );

  delay(1500);

  Serial.println();
  Serial.println();

  Serial.println(
    "========================================"
  );

  Serial.println(
    "       MINEGUARD AI - NODE 1"
  );

  Serial.println(
    "       ESP-NOW GATEWAY"
  );

  Serial.println(
    "========================================"
  );


  // Pins

  pinMode(
    VIBRATION_PIN,
    INPUT
  );

  pinMode(
    BUZZER_PIN,
    OUTPUT
  );

  pinMode(
    RED_LED_PIN,
    OUTPUT
  );

  pinMode(
    BLUE_LED_PIN,
    OUTPUT
  );

  pinMode(
    GREEN_LED_PIN,
    OUTPUT
  );


  digitalWrite(
    BUZZER_PIN,
    LOW
  );

  digitalWrite(
    RED_LED_PIN,
    LOW
  );

  digitalWrite(
    BLUE_LED_PIN,
    LOW
  );

  digitalWrite(
    GREEN_LED_PIN,
    LOW
  );


  analogReadResolution(
    12
  );


  // WiFi

  WiFi.mode(
    WIFI_STA
  );

  delay(100);


  // MAC

  printHardwareMAC();


  // MPU

  if (!initializeMPU())
  {
    Serial.println(
      "MPU initialization failed."
    );
  }


  // OLED

  initializeOLED();


  // SD

  initializeSD();


  // ESP-NOW

  if (!initializeESPNow())
  {
    Serial.println(
      "ESP-NOW FAILED."
    );

    while (true)
    {
      digitalWrite(
        RED_LED_PIN,
        !digitalRead(
          RED_LED_PIN
        )
      );

      delay(500);
    }
  }


  // Initial FSR

  readLocalFSR();

  initialFsr =
    localFsrRaw;


  Serial.println();

  Serial.println(
    "========================================"
  );

  Serial.println(
    "NODE 1 READY"
  );

  Serial.println(
    "Waiting for Node 2..."
  );

  Serial.println(
    "========================================"
  );

  Serial.println();
}


// ============================================================
// LOOP
// ============================================================

void loop()
{
  // ==========================================================
  // NODE 1 SENSOR READING
  // ==========================================================

  if (
    millis() -
    lastSensorRead >=
    SENSOR_INTERVAL
  )
  {
    lastSensorRead =
      millis();

    readLocalMPU();

    readLocalFSR();

    readLocalVibration();
  }


  // ==========================================================
  // PROCESS NODE 2
  // ==========================================================

  processNode2Data();


  // ==========================================================
  // CHECK NODE 2
  // ==========================================================

  checkNode2Connection();


  // ==========================================================
  // COMBINED ALERT
  // ==========================================================

  uint8_t combinedLevel =
    calculateCombinedAlert();


  // ==========================================================
  // LED
  // ==========================================================

  setLEDs(
    combinedLevel
  );


  // ==========================================================
  // BUZZER
  // ==========================================================

  updateBuzzer(
    combinedLevel
  );


  // ==========================================================
  // OLED
  // ==========================================================

  if (
    millis() -
    lastOLEDUpdate >=
    OLED_INTERVAL
  )
  {
    lastOLEDUpdate =
      millis();

    updateOLED(
      combinedLevel
    );
  }


  // ==========================================================
  // SD
  // ==========================================================

  if (
    millis() -
    lastSDLog >=
    SD_INTERVAL
  )
  {
    lastSDLog =
      millis();

    logToSD(
      combinedLevel
    );
  }


  // ==========================================================
  // STATUS
  // ==========================================================

  if (
    millis() -
    lastStatusPrint >=
    STATUS_INTERVAL
  )
  {
    lastStatusPrint =
      millis();

    printStatus();
  }


  delay(10);
}
