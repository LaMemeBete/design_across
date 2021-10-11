import socket
import time
import random
UDP_IP = "127.0.0.1"
UDP_PORT = 6400
MESSAGE = "Hello, World!"
print("UDP target IP: %s" % UDP_IP)
print("UDP target port: %s" % UDP_PORT)
print("message: %s" % MESSAGE)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Internet socket.SOCK_DGRAM) # UDP
while True:
    time.sleep(1)
    sock.sendto(str.encode(MESSAGE + str(random.random())), (UDP_IP, UDP_PORT))