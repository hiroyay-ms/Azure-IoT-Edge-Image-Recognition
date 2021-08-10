import smbus2 as smbus
import time

class LCD:
    def __init__(self):
        self.I2C_ADDR = 0x27
        self.LCD_WIDTH = 16

        self.LCD_CHR = 1
        self.LCD_CMD = 0

        self.LCD_LINE_1 = 0x80
        self.LCD_LINE_2 = 0xC0

        self.LCD_BACKLIGHT = 0x08

        self.ENABLE = 0b00000100

        self.E_PULSE = 0.0005
        self.E_DELAY = 0.0005

        self.bus = smbus.SMBus(1)


    def clear(self):
        self.lcd_byte(0x33, self.LCD_CMD)
        self.lcd_byte(0x32, self.LCD_CMD)
        self.lcd_byte(0x06, self.LCD_CMD)
        self.lcd_byte(0x0C, self.LCD_CMD)
        self.lcd_byte(0x28, self.LCD_CMD)
        self.lcd_byte(0x01, self.LCD_CMD)

        time.sleep(self.E_DELAY)


    def lcd_byte(self, bits, mode):
        bits_high = mode | (bits & 0xF0) | self.LCD_BACKLIGHT
        bits_low = mode | ((bits<<4) & 0xF0) | self.LCD_BACKLIGHT

        self.bus.write_byte(self.I2C_ADDR, bits_high)
        self.lcd_toggle_enable(bits_high)

        self.bus.write_byte(self.I2C_ADDR, bits_low)
        self.lcd_toggle_enable(bits_low)


    def lcd_toggle_enable(self, bits):
        time.sleep(self.E_DELAY)
        self.bus.write_byte(self.I2C_ADDR, (bits | self.ENABLE))

        time.sleep(self.E_PULSE)
        self.bus.write_byte(self.I2C_ADDR, (bits | ~self.ENABLE))

        time.sleep(self.E_DELAY)


    def lcd_string(self, message, line):
        message = message.ljust(self.LCD_WIDTH, " ")

        self.lcd_byte(line, self.LCD_CMD)

        for i in range(self.LCD_WIDTH):
            self.lcd_byte(ord(message[i]), self.LCD_CHR)
    
    def write(self, line1, line2):
        self.lcd_string(line1, self.LCD_LINE_1)
        self.lcd_string(line2, self.LCD_LINE_2)


    def stop(self):
        self.lcd_byte(0x01, self.LCD_CMD)


    def __exit__(self, exception_type, exceotion_value, traceback):
        self.lcd_byte(0x01, self.LCD_CMD)