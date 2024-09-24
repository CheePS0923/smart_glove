//const int S0 = A0;    // select the input pin for the potentiometer
const int S1 = A1; 
//const int S2 = A2; 
const int S3 = A3; 
//const int S4 = A4;
const int S5 = A5;

int LED = 13;
int button = 7;      // select the pin for the LED

//float sensorValue0 = 0;  // variable to store the value coming from the sensor
float sensorValue1 = 0;  // variable to store the value coming from the sensor
//float sensorValue2 = 0;  // variable to store the value coming from the sensor
float sensorValue3 = 0;  // variable to store the value coming from the sensor
//float sensorValue4 = 0;  // variable to store the value coming from the sensor
float sensorValue5 = 0;  // variable to store the value coming from the sensor

//float ratio1 = 0;
//float ratio2 = 0;
//float ratio3 = 0;


int buttonState = LOW;
int prevState = LOW;
unsigned long lastDebounceTime = 0;
unsigned long debounceDelay = 60;
int incomingByte = 0;

void setup() {
  Serial.begin(9600);
}

void loop() {
  buttonState = digitalRead(button);
  
  if ((millis() - lastDebounceTime) > debounceDelay){
    if(buttonState == HIGH && prevState == LOW){
      lastDebounceTime = millis();
      prevState = HIGH;
      Serial.println("1"); //send start sentinel to PC
    }
    else if(buttonState == LOW){
      prevState = LOW;
      lastDebounceTime = millis();
    }
  }
  


  if(Serial.available()>0){
    incomingByte = Serial.read();
    Serial.println(incomingByte);
    Serial.println("Hi");

    while(Serial.available() == 0){
        if(incomingByte == 49){ //ASCII for 1
          //sensorValue0 = analogRead(S0);
          sensorValue1 = analogRead(S1);
          //sensorValue2 = analogRead(S2);
          sensorValue3 = analogRead(S3);
          //sensorValue4 = analogRead(S4);
          sensorValue5 = analogRead(S5);

          sensorValue1 = map(sensorValue1, 0, 100, 0,1023);
          sensorValue3 = map(sensorValue3, 0, 100, 0,1023);
          sensorValue5 = map(sensorValue5, 0, 100, 0,1023);

          Serial.print("R3= ");
          Serial.print(sensorValue1);
          Serial.print("\tR3= ");
          Serial.print(sensorValue3);
          Serial.print("\tR3= ");
          Serial.println(sensorValue5);


        }
        else if(incomingByte == 83){ //ASCII for 'S' : Stop sentinel
          incomingByte = 0;
          break;
        }
      }
  }



}
