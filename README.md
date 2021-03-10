# Replication Package for "Security and Machine Learning Adoption in IoT: A Preliminary Study of IoT Developer Discussions"

## StackOverflow IoT dataset:
The dataset is collected from IoT related 53K SO posts. List of all tags used in the IoT data collection: The Stack Overflow September 2019 data dump is used. The dump can be downlaoded from archive.org

The final list of 78 IoT tags that we used to collect our IoT posts from Stack Overflow as as follows.

- arduino: arduino, arduino-c++, arduino-due, arduino-esp8266, arduino-every, arduino-ide, arduino-mkr1000, arduino-ultra-sonic, arduino-uno, arduino-uno-wifi, arduino-yun, platformio
- iot: audiotoolbox, audiotrack, aws-iot, aws-iot-analytics, azure-iot-central, azure-iot-edge, azure-iot-hub, azure-iot-hub-device-management, azure-iot-sdk, azure iot-suite, bosch-iot-suite, eclipse-iot, google-cloud-iot, hypriot, iot-context-mapping, iot-devkit, iot-driver-behavior, iot-for-automotive, iot-workbench, iotivity, microsoft iot-central, nb-iot, rhiot, riot, riot-games-api, riot.js, riotjs, watson-iot, windows-10-iot-core, windows-10-iot-enterprise, windows-iot-core-10, windowsiot, wso2iot, xamarin.iot
- raspberry-pi: adafruit, android-things, attiny, avrdude, esp32, esp8266, firmata, gpio, hm-10, home-automation, intel-galileo, johnny-five, lora, motordriver, mpu6050, nodemc, omxplayer, raspberry-pi, raspberry-pi-zero, raspberry-pi2, raspberry-pi3, raspberry-pi4, raspbian, serial-communication, servo, sim900, teensy, wiringpi, xbee

## The code of RoBERTa and the Benchmark 
Can be found under the code folder

## The labeled dataset for security and ML
Can be found in https://www.dropbox.com/s/q0p6ewhs2163uuy/IoT_Data_ML_Final.csv?dl=0 This is a CSV file. It is 96MB. It has all the sentences that we picked from our 53K IoT posts, each labeled as security and ML as discussed in the paper
