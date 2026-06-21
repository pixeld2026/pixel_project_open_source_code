from machine import I2C, Pin
import network
import socket
import time
import ujson
import struct
import math

SSID       = "YOUR_WIFI"
PASSWORD   = "YOUR_PASSWORD"
PI_IP      = "172.20.10.3"   
PORT       = 5005
SEND_HZ    = 20        

wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(SSID, PASSWORD)
print("WiFi csatlakozás...")
timeout = 20
while not wlan.isconnected() and timeout > 0:
    time.sleep(0.5)
    timeout -= 1
if wlan.isconnected():
    print("Csatlakozva:", wlan.ifconfig())
else:
    print("WiFi hiba! Újraindítás...")
    import machine
    machine.reset()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

i2c_imu = I2C(0, sda=Pin(0), scl=Pin(1), freq=400000)
i2c_hr  = I2C(1, sda=Pin(2), scl=Pin(3), freq=400000) 

MPU_ADDR = 0x68
MAX_ADDR = 0x57

i2c_imu.writeto_mem(MPU_ADDR, 0x6B, bytes([0x00]))  
i2c_imu.writeto_mem(MPU_ADDR, 0x1B, bytes([0x00]))  
i2c_imu.writeto_mem(MPU_ADDR, 0x1C, bytes([0x00]))  
print("MPU6050 kész")

max_ok = False
try:
    i2c_hr.writeto_mem(MAX_ADDR, 0x09, bytes([0x40])) 
    time.sleep(0.2)
    i2c_hr.writeto_mem(MAX_ADDR, 0x04, bytes([0x00]))  
    i2c_hr.writeto_mem(MAX_ADDR, 0x05, bytes([0x00]))  
    i2c_hr.writeto_mem(MAX_ADDR, 0x06, bytes([0x00]))  
    i2c_hr.writeto_mem(MAX_ADDR, 0x09, bytes([0x03]))  
    i2c_hr.writeto_mem(MAX_ADDR, 0x0A, bytes([0x27]))  
    i2c_hr.writeto_mem(MAX_ADDR, 0x0C, bytes([0x24]))  
    i2c_hr.writeto_mem(MAX_ADDR, 0x0D, bytes([0x24]))  
    max_ok = True
    print("MAX30102 kész")
except Exception as e:
    print("MAX30102 hiba:", e)

def mpu_read():
    d = i2c_imu.readfrom_mem(MPU_ADDR, 0x3B, 14)
    def s16(h, l):
        v = (h << 8) | l
        return v - 65536 if v > 32767 else v
    ax = s16(d[0],  d[1])  / 16384.0
    ay = s16(d[2],  d[3])  / 16384.0
    az = s16(d[4],  d[5])  / 16384.0
    gx = s16(d[8],  d[9])  / 131.0
    gy = s16(d[10], d[11]) / 131.0
    gz = s16(d[12], d[13]) / 131.0
    return ax, ay, az, gx, gy, gz

def max_read():
    if not max_ok:
        return None, None
    try:
        wr = i2c_hr.readfrom_mem(MAX_ADDR, 0x04, 1)[0] & 0x1F
        rd = i2c_hr.readfrom_mem(MAX_ADDR, 0x06, 1)[0] & 0x1F
        if ((wr - rd) & 0x1F) == 0:
            return None, None
        d   = i2c_hr.readfrom_mem(MAX_ADDR, 0x07, 6)
        red = ((d[0] & 0x03) << 16) | (d[1] << 8) | d[2]
        ir  = ((d[3] & 0x03) << 16) | (d[4] << 8) | d[5]
        return red, ir
    except:
        return None, None

interval   = 1.0 / SEND_HZ
last_send  = time.ticks_ms()
err_count  = 0

print(f"Adatküldés indult → {PI_IP}:{PORT} @ {SEND_HZ}Hz")

while True:
    now = time.ticks_ms()
    if time.ticks_diff(now, last_send) >= int(interval * 1000):
        last_send = now

        ax, ay, az, gx, gy, gz = mpu_read()
        red, ir = max_read()

        packet = {
            "ax": round(ax, 4),
            "ay": round(ay, 4),
            "az": round(az, 4),
            "gx": round(gx, 2),
            "gy": round(gy, 2),
            "gz": round(gz, 2),
        }
        if red is not None:
            packet["red"] = red
            packet["ir"]  = ir

        try:
            sock.sendto(ujson.dumps(packet).encode(), (PI_IP, PORT))
            err_count = 0
        except Exception as e:
            err_count += 1
            print("UDP hiba:", e)
            if err_count > 20:
                print("Túl sok hiba, újraindítás...")
                import machine
                machine.reset()

    time.sleep_ms(1)
