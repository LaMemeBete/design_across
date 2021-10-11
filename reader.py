import serial
import time

# make sure the 'COM#' is set according the Windows Device Manager
ser = serial.Serial('/dev/cu.usbserial-01937934', 9800, timeout=1)
#time.sleep(2)

while True:
    line = ser.readline()   # read a byte
    #time.sleep(0.1)
    if line:
        string = line.decode()  # convert the byte string to a unicode string
        print(string)

#ser.close()