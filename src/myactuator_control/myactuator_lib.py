import os
import time
import can
import numpy as np
import rospy
from threading import Lock
from can_master.msg import can_in, can_out
import serial
import select
from pymodbus.utilities import computeCRC
from typing import Tuple

# Trying to move this to the individual files
# ########################## 
# # CAN = True				## If CAN is used for drivetrain, set to True
# CAN = False				## If UART is used for drivetrain, set to False
# ########################## 

# List containing valid myActuator types for parameters
myActuator_List = [
    "X6_S2_V2_1",
    "X6_S2_V2_2",
    "X10_S2_V3"
]

control_messages = [
    0x9C,   # Status Message 2
    0xA1,   # Torque Control
    0xA2,   # Speed Control
    0xA3,   # Position Tracking Control
    0xA4,   # Absolute Tracking Control
    0xA8,   # Incremental Position Control
]

# Parameter limits for specific myActuator motor types
X6_S2_V2_1_PARAMS = {
    "CURR_KP_MAX" : 0.5,
    "CURR_KI_MAX" : 0.1,
    "SPEED_KP_MAX" : 0.01,
    "SPEED_KI_MAX" : 0.001,
    "POS_KP_MAX" : 0.5,
    "POS_KI_MAX" : 0.005,
    "VOLT_MAX" : 55,      # 0-100 V
    "VOLT_MIN" : 20,      # 0-100 V
    "STALL_LIMIT" : 0,    # 0-(2^32 - 1) s
    "MAX_SPEED" : 600    # Max speed of motor is 600 RPM
}

X6_S2_V2_2_PARAMS = {
    "CURR_KP_MAX" : 0.5,
    "CURR_KI_MAX" : 0.05,
    "SPEED_KP_MAX" : 0.01,
    "SPEED_KI_MAX" : 0.0005,
    "POS_KP_MAX" : 0.1,
    "POS_KI_MAX" : 0.005,
    "VOLT_MAX" : 55,      # 0-100 V
    "VOLT_MIN" : 20,      # 0-100 V
    "STALL_LIMIT" : 0,    # 0-(2^32 - 1) s
    "MAX_SPEED" : 600    # Max speed of motor is 600 RPM
}

X10_S2_PARAMS = {
    "CURR_KP_MAX" : 0.5,
    "CURR_KI_MAX" : 0.1,
    "SPEED_KP_MAX" : 0.2,
    "SPEED_KI_MAX" : 0.005,
    "POS_KP_MAX" : 0.5,
    "POS_KI_MAX" : 0.05,
    "VOLT_MAX" : 55,      # 0-100 V
    "VOLT_MIN" : 20,      # 0-100 V
    "STALL_LIMIT" : 0,    # 0-(2^32 - 1) s
    "MAX_SPEED" : 600    # Max speed of motor is 600 RPM
}

# Potential error states when reading motor_status_1
ERROR_STATES = {
    0x0002 : "MOTOR_STALL",
    0x0004 : "LOW_PRESSURE",
    0x0008 : "OVER_VOLTAGE",
    0x0010 : "OVER_CURRENT",
    0x0040 : "POWER_OVERRUN",
    0x0100 : "SPEEDING",
    0x1000 : "MOTOR_OVER_TEMPERATURE",
    0x2000 : "ENCODER_CALIBRATION_ERR"
}

# Delays between sending and recieving a message (msec)
CAN_DELAY = 0.01
CAN_TIMEOUT = 0.2

# Conversion factor for Current Amps
amp_conversion = 0.01    # Use to convert int16 to floating point Amp and vice versa (0.01A/LSB)
int16_min = -32767
int16_max = 32767

def params(motor_type):
    """
    Determines which params used based on motor type chosen
    
    Parameters
    ----------
    motor_type : string
        The model of myActuator motor being used 
    -------
    Returns
    -------
    params : dictionary
        all parameters specific to that model
    """
    assert motor_type in myActuator_List, "Invalid motor type."
    if motor_type == "X6_S2_V2_1":
        params = X6_S2_V2_1_PARAMS
    elif motor_type == "X6_S2_V2_2":
        params = X6_S2_V2_2_PARAMS
    elif motor_type == "X10_S2_V3":
        params = X10_S2_PARAMS
    return params


def uint8_to_uint16(byte_high, byte_low):
    """
    Combines two uint8 bytes to create single uint16 
    
    Parameters
    ---------
    - byte_high : uint8
    - byte_low : uint8
    -------
    Returns
    -------
    - uint16 equivalent of the combined bytes
    """
    ubyte_16 = ((byte_high << 8) | byte_low)
    ubyte_16 = np.uint16(ubyte_16)     # Ensure data type is uint16
    return ubyte_16

def uint16_to_uint8(ubyte_16):
    """
    Seperates single uint16 into two uint8 bytes
    Can also seperate int16 into two bytes
    
    Parameters
    ---------
    - byte_16 : uint16
    -------
    Returns
    -------
    - bytes : list (uint8)
        - bytes[0] is the low byte
        - bytes[1] is the high byte
    """
    ubyte_high = ubyte_16 >> 8
    ubyte_low = ubyte_16 & 0x00FF
    bytes = np.array([ubyte_low, ubyte_high], dtype=np.uint8)   # Ensure returned list is in uint8
    return bytes

def int32_to_uint8(byte_32):
    """
    Seperates int32 into four uint8 bytes
    
    Parameters
    ---------
    - byte_32 : int32
    -------
    Returns
    -------
    - bytes : list (uint8)
        - bytes[0] is the lowest byte   (byte 1)
        - bytes[1] is second lowest     (byte 2)
        - bytes[2] is second highest    (byte 3)
        - bytes[3] is the highest byte  (byte 4)
    """
    byte_4 = (byte_32 >> 24)
    byte_3 = (byte_32 >> 16) & 0xFF
    byte_2 =  (byte_32 >> 8) & 0xFF
    byte_1 = byte_32 & 0xFF
    bytes = np.array([byte_1, byte_2, byte_3, byte_4], dtype=np.uint8)   # Ensure returned list is in uint8
    return bytes    

def uint8_to_int32(byte1, byte2, byte3, byte4):
    """
    Combines four uint8 to single int32 value
    
    Parameters
    ---------
    - byte1 : Lowest Byte       (uint8)
    - byte2 : Second Lowest     (uint8)
    - byte3 : Second Highest    (uint8)
    - byte4 : Highest Byte      (uint8)
    -------
    Returns
    -------
    - byte_32 : (int32)
    """    
    byte2 = byte2 << 8
    byte3 = byte3 << 16
    byte4 = byte4 << 24
    byte_32 = np.int32(byte4 | byte3 | byte2 | byte1)
    return byte_32


class MyActuatorMotor:
    """
    MyActuatorMotor Class
        Used to create motor object that includes various control functions
        allowing for communication over CAN connection
    ---------------------
    Supported Motor Types
    ---------------------
    - "X6_S2_V2",
    - "X10_S2_V3"
    ----------
    Functions
    ----------
    - read_pid()   
    - display_pid()
    - pid_params_to_value(curr_KP, curr_KI, speed_KP, speed_KI, pos_KP, pos_KI): 
    - write_pid(save_type, curr_KP, curr_KI, speed_KP, speed_KI, pos_KP, pos_KI)
    - read_motor_status_1()
    - read_motor_status_2()
    - display_motor_status_1()
    - display_motor_status_2()
    - read_abs_position()
    - read_position()
    - read_home_position()
    - read_zero_offset()
    - display_abs_position()
    - display_position()
    - display_home_position()
    - display_zero_offset()
    - torque_control(current)
    - speed_control(speed)
    - position_tracking_control(angle)
    - absolute_position_control(angle, max_speed)
    - incremental_position_control(self, angle, max_speed)
    - write_acceleration(control_type, acceleration/deceleration)
    - set_timeout(timeout)  # Don't use
    - set_baud(select)
    - write_canID(id)   # Only use when one motor connected
    - read_canID()      # Only use when one motor connected
    - stop()
    - reset()


    """
    def __init__(self, motor_type, id, use_can, port=None): 
        self.id = id
        self.params = params(motor_type)
        self.motor_type = motor_type
        self.last_tick_lock = Lock()
        self.last_tick = time.time()
        self.current_response = None    
        self.ack = 0
        self.read = False
        # Used for CAN comms
        self.can_pub = rospy.Publisher("can_out", can_out, queue_size=20)
        self.can_sub = rospy.Subscriber("can_in", can_in, self.response_callback)
        # Used for UART comms
        serial_name = port
        baud_rate = 115200
        # timeout = 10 # Not used??
        self.CAN = use_can
        if (self.CAN == False):
        # We are using UART and need to connect to the UART Devices
            connected = False # Have we connected to the uart device yet?
            while not connected and not rospy.is_shutdown():
                try: # Try to connect to the Uart device over and over until it works
                    print("Trying to Connect to UART Device")
                    self.conn = serial.Serial(serial_name, baud_rate, timeout=0)
                    connected = True
                except serial.SerialException as e:
                    if "Port is already open." in str(e):
                        print("The port is already open.")
                        connected = True
                        continue
                    print("Unable to connect to UART Device, Error: ", e)
                    time.sleep(0.1)
            print("connected to UART Device")

        # The response dict saves the most recent responses to the different queries
        self.response_dict = dict()


    def response_callback(self, msg:can_in):    
        # If the message matches the motor's id, save the data for that message
        if (self.id + 0x100) == msg.arb_id:
            self.response_dict[msg.data[0]] = (msg.data, time.time())

    def __response(self, cmd) -> Tuple:
        # This is a lookup for responses
        # time.sleep(0.01) # Wait a little to give time to possibly recieve new messages
        # Look into the response_dict for the response matching this command type
        return self.response_dict.get(cmd, (None, None))

    def _send_cmd(self, command, read:bool=False):
        """
        Sends commconnand over CAN/UART bus to a specified motor.
        
        Parameters
        ----------
        - command : list (uint_8) (8 bytes)
            - The hexidecimal command to be sent to the motor.
        -------
        Returns
        -------
        - response_msg : list (uint_8) (8 bytes)
            - List containing the 8 bytes of the response data.
            - One or two bytes of this data may be used to represent a parameter
            that was requested in the initial command.
        """
        self.read = read
        if self.CAN:
            # print("Sending CAN")
            return self._send_cmd_CAN(command) # send_cmd_CAN doesnt have any return 
        else:
            # print("Sending UART")
            self._send_cmd_UART(command)
            return (None, None) # TODO: This is wrong but somethings need to be fixed for handling UART responses
        

    def _send_cmd_CAN(self, command):
        """
        Send command if using CAN communication
        """
        out_msg = can_out()
        out_msg.arb_id = self.id
        out_msg.sub_id = int(command[0])
        out_msg.data = bytes(command)
        # print("Library Send:", hex(out_msg.arb_id)," ", out_msg.data)
        self.time_sent = time.time()
        self.can_pub.publish(out_msg)
        if self.read:
            return self.__response(command[0])
        return (None, None)


    def _send_cmd_UART(self, command):
        """
        Send command if using UART communication
        """
        header = [0x3e, self.id-0x140, 0x08]
        command = list(command)
        crc = computeCRC(bytes(header + command))          
        values = bytearray(header + command + [crc>>8, crc&0xFF])
        try:
            self.conn.write(values)
        except serial.SerialException as e:
            print("Unable to write UART Command. Error: ", e)
            self.conn.close()
            is_open = False
            while not is_open and not rospy.is_shutdown():
                print(is_open)
                try:
                    print("Attempting to reopen port")
                    self.conn.open()
                    is_open = True
                except serial.SerialException as e:
                    if "Port is already open." in str(e):
                        print("The port is already open.")
                        is_open = True
                        continue
                    print("Unable to open port. Error: ", e)
                    time.sleep(0.1)
        # Only read if requested
        if self.read:   
            read,_,_ = select.select([self.conn], [], [], 0.02)
            try:
                read_data = self.conn.read(13) 
                if len(read_data) == 0:
                    return (None, None)
                reply = read_data[3:]
                return reply
            except Exception as e:
                print("UART failed to read data: ", e)
        return (None, None)


    def __convert_control_return(self, data):
        """
        Converts data from the CAN response message into correct format
        for temp, torque current, output speed, and output angle parameters.
        
        These parameters returned in torque and speed control,
        as well as read_motor_status_2
        
        Parameters
        ----------
        - data : list (uint8) (8 bytes)
            - Raw data from the CAN response message

        ----------
        Returns
        ----------
        - status : List containing motor status parameters, 4 parameters returned
            - status[0] : motor temperature    (int8)
                - Value in Celcius straight from data
            - status[1] : torque current       (int16) 
                - Torque current in Amps
            - status[2] : output shaft speed   (int16)
                - Shaft speed in degrees per second (dps)
            - status[3] : output shaft angle   (int16)
                - encoder position in degrees, range +-32767°
        -------
        """        
        temperature = np.int8(data[1])
        # Convert uint8 bytes to uint16, then to int16
        # After correct data type, do conversion rates to get correct units
        goal_torque = uint8_to_uint16(data[3], data[2])
        goal_torque = np.int16(goal_torque) * amp_conversion
        motor_speed = uint8_to_uint16(data[5],data[4])
        motor_speed = np.int16(motor_speed)    
        motor_angle = uint8_to_uint16(data[7],data[6])
        motor_angle = np.int16(motor_angle)

        status = [temperature, goal_torque, motor_speed, motor_angle]
        return status


    def __display_control_return(self, status):
        """
        Displays motor parameters returned from torque and speed control functions
        to the console.
        
        Parameters
        ----------
        status : list
            List of parameters for temp, torque current, output speed, and output angle
        -------
        """
        print("Temperature (C):", status[0])
        print("Torque Current (A):", status[1])
        print("Output Speed (dps):", status[2])
        print("Output Angle (degree):", status[3], "\n")

    def read_pid(self):
        """
        Reads the parameters of current, speed, and position loop KP and KI. 
        
        Returns
        -------
        data : list (float)
            List containing converted uint8 to float data for the PID Parameters
            data[2] : Current Loop KP
            data[3] : Current Loop KI
            data[4] : Speed Loop KP
            data[5] : Speed Loop KI
            data[6] : Position Loop KP
            data[7] : Position Loop KI
        """
        command = [0x30, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        data = self._send_cmd(command, True)
        # Convert data to floats using max value parameters
        if data is None:
            return None
        data = np.array(data, dtype=np.float)
        data[2] = data[2] * (self.params["CURR_KP_MAX"] / 256)      # Curr_KP 
        data[3] = data[3] * (self.params["CURR_KI_MAX"] / 256)      # Curr_KI 
        data[4] = data[4] * (self.params["SPEED_KP_MAX"] / 256)     # Speed_KP 
        data[5] = data[5] * (self.params["SPEED_KI_MAX"] / 256)     # Speed_KI
        data[6] = data[6] * (self.params["POS_KP_MAX"] / 256)       # Pos_KP 
        data[7] = data[7] * (self.params["POS_KI_MAX"] / 256)       # Pos_KI
        return data

    def display_pid(self):
        """
        Displays the results of read_pid() to console

        """             
        pid_params = self.read_pid()
        print(pid_params)


    def pid_params_to_value(self, curr_KP, curr_KI, speed_KP, speed_KI, pos_KP, pos_KI):
        # Calculate actual values set for the KI and KP parameters
        actual_values = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        actual_values[0] = curr_KP * (self.params["CURR_KP_MAX"] / 256)      # Curr_KP 
        actual_values[1] = curr_KI * (self.params["CURR_KI_MAX"] / 256)      # Curr_KI 
        actual_values[2] = speed_KP * (self.params["SPEED_KP_MAX"] / 256)    # Speed_KP 
        actual_values[3] = speed_KI * (self.params["SPEED_KI_MAX"] / 256)    # Speed_KI
        actual_values[4] = pos_KP * (self.params["POS_KP_MAX"] / 256)        # Pos_KP 
        actual_values[5] = pos_KI * (self.params["POS_KI_MAX"] / 256)        # Pos_KI
        return actual_values


    def write_pid(self, save_type, curr_KP, curr_KI, speed_KP, speed_KI, pos_KP, pos_KI):
        """
        Writes the parameters of current, speed, and position loop KP and KI to RAM or ROM.
        For RAM: Parameters are not saved when motor powers off and must be set again. 
        For ROM: Parameters are saved after power off
        ----------
        Parameters
        ----------
        - save_type : String
            - String to denote whether writing to "RAM" or "ROM"
        - curr_KP : float
            - Current loop KP
        - curr_KI : float
            - Current loop KI
        - speed_KP : float
            - Speed loop KP
        - speed_KI : float
            - Speed loop KI 
        - pos_KP : float
            - Position loop KP
        - pos_KI : float
            - Position loop KI 
        """
        # Determine which command header byte to set depending on save type
        if save_type == "RAM":
            header = 0x31
        elif save_type == "ROM":
            header = 0x32
        # Create command and convert float values for parameters to equivalent uint8 values
        # Using NumPy to convert to uint8 or ubyte
        command = [header, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        # command[2] = curr_KP    # Curr_KP 
        # command[3] = curr_KI    # Curr_KI 
        # command[4] = speed_KP   # Speed_KP 
        # command[5] = speed_KI   # Speed_KI
        # command[6] = pos_KP     # Pos_KP 
        # command[7] = pos_KI     # Pos_KI
        command[2] = curr_KP / (self.params["CURR_KP_MAX"] / 256)      # Curr_KP 
        command[3] = curr_KI / (self.params["CURR_KI_MAX"] / 256)      # Curr_KI 
        command[4] = speed_KP / (self.params["SPEED_KP_MAX"] / 256)    # Speed_KP 
        command[5] = speed_KI / (self.params["SPEED_KI_MAX"] / 256)    # Speed_KI
        command[6] = pos_KP / (self.params["POS_KP_MAX"] / 256)        # Pos_KP 
        command[7] = pos_KI / (self.params["POS_KI_MAX"] / 256)        # Pos_KI
        # Make sure command list is uint8 (I think this is how the function works, may be mistaken)
        command = np.array(command, dtype=np.uint8)
        # Send command to motor, no need for return since same data sent is same data 
        self._send_cmd(command)

    def read_motor_status_1(self):
        """
        Reads the current motor temperature, voltage, and error status flags
        
        -------
        Returns
        -------
        status : List containing motor status parameters, 4 parameters returned
            - status[0] : motor temperature    (int8)
                - Value in Celcius straight from data
            - status[1] : brake release command (uint8) 
                - 1 for brake release command, 0 for brake lock command
            - status[2] : voltage       (float)
                - Value in Volts
            - status[3] : error flag status    (string)
                - String containing the error status description from ERROR_STATES, or None if no error

        """
        command = [0x9A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        reply = self._send_cmd(command, True)

        if reply is None:
            return None
        else:
            temperature = np.int8(reply[1])    # Temperature, converted to int8
            brake_rel = reply[3]     # Brake Release Command
            # Convert high and low bytes of uint8 data to uint16
            # Multiply reply[4] by 0.1 for conversion factor to volts
            voltage = uint8_to_uint16(reply[5],reply[4]) * 0.1    # Voltage
            err_status = uint8_to_uint16(reply[7], reply[6])      # Error Status Code
            
            # Check if response corresponds to any error code, and set status description accordingly
            err_desc = ERROR_STATES.get(err_status, None)
            if err_desc == None:
                err_status = None
            else:
                err_status = err_desc
            status = [temperature, brake_rel, voltage, err_desc]
            return status

    
    def read_motor_status_2(self):
        """
        Reads the current motor temperature, torque, speed, and encoder position 
        of the motor
        
        -------
        Returns
        -------
        status : List containing motor status parameters, 4 parameters returned
            - status[0] : motor temperature    (int8)
                - Value in Celcius straight from data
            - status[1] : torque current       (int16) 
                - Torque current in Amps
            - status[2] : output shaft speed   (int16)
                - Shaft speed in degrees per second (dps)
            - status[3] : output shaft angle   (int16)
                - encoder position in degrees, range +-32767°

        """
        command = [0x9C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        reply = self._send_cmd(command, True)
        # If no reply, no need to do post-processing
        if reply is None:
            return None
        status = self.__convert_control_return(reply)
        return status


    def display_motor_status_1(self):
        """
        Displays the results of read_motor_status_1 to the console.

        """
        status = self.read_motor_status_1()
        if status is None:
            return
        else:
            print("Motor Status 1:")
            print("Temperature (C°):", status[0])
            if status[1] == 0:
                status[1] = "Enabled"
            else:
                status[1] = "Disabled"
            print("Brake Release:", status[1])
            print("Voltage (V):", status[2])
            print("Error Flags:", status[3], "\n")    
    

    def display_motor_status_2(self):
        """
        Displays the results of read_motor_status_2() to the console.

        """
        status = self.read_motor_status_2()
        # If no data, no need to display
        if status is None:
            return None
        print("Motor Status 2:")
        self.__display_control_return(status)
        return status

    def read_abs_position(self):
        """
        Reads the current multi-turn absolute angle value of the motor
        
        -------
        Returns
        -------
        - position : output shaft angle   (int32)
            - encoder position in pulses
            - The value after subtracting encoders zero offset (initial pos.) from original position

        """        
        command = [0x92, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        reply, t = self._send_cmd(command, True)
        if reply is None or reply[0] != 0x92 or len(reply) < 8:
            return None
        # Convert reply data to usable position data
        position = uint8_to_int32(reply[4], reply[5], reply[6], reply[7]) * 0.01
        return position

    def read_position(self):
        """
        Reads the multi-turn position of the encoder, which represents the rotational angle
        of the motor output shaft.
        
        -------
        Returns
        -------
        - position : output shaft angle   (int32)
            - encoder position in pulses
            - The value after subtracting encoders zero offset (initial pos.) from original position

        """        
        command = [0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        reply, t = self._send_cmd(command, True)
        if reply is None:
            return None
        # Convert reply data to usable position data
        position = uint8_to_int32(reply[4], reply[5], reply[6], reply[7])
        return position


    def read_home_position(self):
        """
        Reads the multi-turn home position of the encoder, which represents the encoder value
        without the zero offset.
        
        -------
        Returns
        -------
        - position : output shaft angle   (int32)
            - encoder position in pulses

        """        
        command = [0x61, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        reply, t = self._send_cmd(command, True)
        if reply is None:
            return None
        # Convert reply data to usable position data
        position = uint8_to_int32(reply[4], reply[5], reply[6], reply[7])
        return position


    def read_zero_offset(self):
        """
        Reads the multi-turn zero offset value (initial position) of the encoder.
        
        -------
        Returns
        -------
        - position : output shaft angle   (int32)
            - encoder position in pulses
        """        
        command = [0x62, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        reply, t = self._send_cmd(command, True)
        if reply is None:
            return None
        # Convert reply data to usable position data
        position = uint8_to_int32(reply[4], reply[5], reply[6], reply[7])
        return position


    def display_abs_position(self):
        """
        Displays the results of read_abs_position() to the console.

        """             
        position = self.read_abs_position()
        print("Read Absolute Position:", position)
        return position


    def display_position(self):
        """
        Displays the results of read_position() to the console.

        """             
        position = self.read_position()
        print("Read Position:", position)
        return position


    def display_home_position(self):
        """
        Displays the results of read_home_position() to the console.

        """  
        position = self.read_home_position()
        print("Home Position:", position)
        return position


    def display_zero_offset(self):
        """
        Displays the results of read_zero_offset() to the console.

        """                    
        position = self.read_zero_offset()
        print("Zero Offset:", position)
        return position    


    def write_encoder_as_zero(self, position):
        """
        Write the zero offset (initial position) of the encoder.
        New zero offset should be used as reference when setting target position

        -------
        Parameters
        -------
        - position : output shaft angle   (int32)
            - encoder position in pulses
            - The value after subtracting encoders zero offset (initial pos.) from original position

        """        
        command = [0x63, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        position_bytes = int32_to_uint8(position)
        command[4] = position_bytes[0]
        command[5] = position_bytes[1]
        command[6] = position_bytes[2]
        command[7] = position_bytes[3]
        reply, t = self._send_cmd(command, True)
        if reply is None:
            return None
        # Convert reply data to usable position data
        position = uint8_to_int32(reply[4], reply[5], reply[6], reply[7])
        return position        


    def torque_control(self, torque_current:float):
        """
        Sets the torque current control output of the motor
        
        Parameters
        ----------
        - torque_current : float
            - Current set should be in Amps

        -------
        Returns
        -------
        - status : list
            - List contains motor temp, torque current value, motor output speed, and motor output angle
            - status[0] : motor_temp (int8),        (1° Celcius / LSB)
                - Motor temp in Celcius
            - status[1] : torque_current (int16),   (0.01A / LSB)
                - Torque current in Amps
            - status[2] : output_speed (int16),     (1dps / LSB)
                - Output shaft speed in degrees per second (dps)
            - status[3] : output_angle (int16),     (1° / LSB)
                - Output shaft angle in degrees (relative to the zero position), range +-32767°
                - Testing saw angle range as +-3640°
        """
        if (torque_current < -4) | (torque_current > 4):
            print("CANNOT SET TORQUE CONTROL VALUE. KEEP IN SAFE RANGE (-4 - +4) AMPS")
            return
        command = [0xA1, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        torque_current = np.int16(torque_current / amp_conversion)
        torque_bytes = uint16_to_uint8(torque_current)
        command[4] = torque_bytes[0]    # Low Byte to data[4]
        command[5] = torque_bytes[1]    # High Byte to data[5]
        print("Command:", command)
        reply, t = self._send_cmd(command, True)   
        print("Reply:", reply)

        # If no reply, no need to do post-processing
        if reply is None:
            return None
        else:
            status = self.__convert_control_return(reply)
            print("Torque Control:")
            self.__display_control_return(status)
            return status

    def speed_control(self, speed, read=False):
        """
        Sets the speed of the motor output shaft, in degrees per second.
        
        Max speed is 600 RPM
        Nominal speed around 300 or 350 DPS 
        (May change for different Torque current, and KI,KP values)
        
        ----------
        Parameters
        ----------
        speed: int
            Speed that the motor shaft will output

        -------
        Returns
        -------
        - status : list
            - List contains motor temp, torque current value, motor output speed, and motor output angle
            - status[0] : motor_temp (int8),        (1° Celcius / LSB)
                - Motor temp in Celcius
            - status[1] : torque_current (int16),   (0.01A / LSB)
                - Torque current in Amps
            - status[2] : output_speed (int16),     (1dps / LSB)
                - Output shaft speed in degrees per second (dps)
            - status[3] : output_angle (int16),     (1° / LSB)
                - Output shaft angle in degrees, range +-32767°
                - Testing saw angle range as +-3640°
        """
		# update last motor_cmd received time
        # No need since not checking last_tick time
        # with self.last_tick_lock:
        #     self.last_tick = time.time()

        # If speed is 0, send stop command instead
        # if speed <= 1 & speed >= -1:
        #     self.stop()

        # max_speed = self.params.get("MAX_SPEED", 600)   # Get MAX_SPEED from params, or if not there set to default 600
        # max_speed_str = str(max_speed)
        # if (speed > max_speed) | (speed < -max_speed):
        #     print("CANNOT SET SPEED CONTROL VALUE. MUST BE IN RANGE (-600",max_speed_str," - +",max_speed_str,")")
        command = [0xA2, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        # Do speed conversion and convert to int32        
        speed = np.int32(speed / .01)
        # Convert int32 to 4 uint8 bytes to send in command
        speed_bytes = int32_to_uint8(speed)
        command[4] = speed_bytes[0]
        command[5] = speed_bytes[1]
        command[6] = speed_bytes[2]
        command[7] = speed_bytes[3]
        reply, t = self._send_cmd(command, read)
        # reply = self.__response()
        if read:
            if reply is None:
                return None
            else:
                # Put in list and display the return status to console
                status = self.__convert_control_return(reply)
                #print("Speed Control:")
                #self.__display_control_return(status)
                current = status[1]
                # print(current)
                return current
        return None


    def position_tracking_control(self, angle):
        """
        Controls the position of the motor to the desired angle given by passing through PI controller.
        Rotational direction is determined by the difference between target and current
        position. Used for direct position tracking.
        
        Ex.
        Giving an angle of 1000 will move the motor to the exact encoder position at 1000°,
        Moving in the direction that will reach the position fastest. 
        
        ----------
        Parameters
        ----------
        - angle: int
            - destination angle in degrees that the motor will turn to.

        -------
        Returns
        -------
        - status : list
            - List contains motor temp, torque current value, motor output speed, and motor output angle
            - status[0] : motor_temp (int8),        (1° Celcius / LSB)
                - Motor temp in Celcius
            - status[1] : torque_current (int16),   (0.01A / LSB)
                - Torque current in Amps
            - status[2] : output_speed (int16),     (1dps / LSB)
                - Output shaft speed in degrees per second (dps)
            - status[3] : output_angle (int16),     (1° / LSB)
                - Output shaft angle in degrees, range +-32767°
                - Testing saw angle range as +-3640°
        """
        angle = np.int32(angle / 0.01)
        if (angle > 65535) | (angle < -65535):
            print("CANNOT SET POSITION TRACKING CONTROL VALUE. MUST BE IN RANGE (-65535 - +65535)")
            return
        
        command = [0xA3, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        angle_bytes = int32_to_uint8(angle)
        command[4] = angle_bytes[0]
        command[5] = angle_bytes[1]
        command[6] = angle_bytes[2]
        command[7] = angle_bytes[3]
        reply, t = self._send_cmd(command)
        if reply is None:
            return None
        else:
            # Put in list and display the return status to console
            status = self.__convert_control_return(reply)
            print("Position Tracking Control:")
            self.__display_control_return(status)
            return status

    def absolute_position_control(self, angle:int, max_speed=500, read:bool=False):
        """
        Controls the position of the motor to the desired angle given.
        Rotational direction is determined by the difference between target and current
        position.
        
        Ex. 
        Giving an angle of 1000 will move the motor to the exact encoder position at 1000°,
        Moving in the direction that will reach the position fastest. 
        
        ----------
        Parameters
        ----------
        - angle : int (-65535 - +65535)
            - destination angle in degrees that the motor will turn to.
        - max_speed : int (0 - 500)
            - max speed that can move to goal position
            - Default is set to 500 dps

        -------
        Returns
        -------
        - status : list
            - List contains motor temp, torque current value, motor output speed, and motor output angle
            - status[0] : motor_temp (int8),        (1° Celcius / LSB)
                - Motor temp in Celcius
            - status[1] : torque_current (int16),   (0.01A / LSB)
                - Torque current in Amps
            - status[2] : output_speed (int16),     (1dps / LSB)
                - Output shaft speed in degrees per second (dps)
            - status[3] : output_angle (int16),     (1° / LSB)
                - Output shaft angle in degrees, range +-32767°
                - Testing saw angle range as +-3640°
        """
        angle = np.int32(angle / 0.01)
        if (angle > 65535) | (angle < -65535):
            print("CANNOT SET POSITION TRACKING CONTROL VALUE. MUST BE IN RANGE (-65535 - +65535)")
            return
        
        command = [0xA4, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        # Put respective bytes fro max speed into command
        speed_bytes = uint16_to_uint8(max_speed)
        command[2] = speed_bytes[0]
        command[3] = speed_bytes[1]
        
        # Put respective bytes for angle into command
        angle_bytes = int32_to_uint8(angle)
        command[4] = angle_bytes[0]
        command[5] = angle_bytes[1]
        command[6] = angle_bytes[2]
        command[7] = angle_bytes[3]
        reply, t = self._send_cmd(command, read)
        if reply is None:
            return None
            
        # Put in list and display the return status to console
        status = self.__convert_control_return(reply)
        print("Absolute Position Control:")
        self.__display_control_return(status)
        return status        

    
    def incremental_position_control(self, angle:int, max_speed=500, read:bool=False):
        """
        Controls the incremental position (multi-turn angle) of the motor,
        by running the input position increment with the current position as the
        starting point. 
        
        Ex. 
        Giving an angle of 1000 will move the motor clockwise 1000° from current position.
        
        ----------
        Parameters
        ----------
        - angle : int (-65535 - +65535)
            - destination angle in degrees that the motor will turn to.
        - max_speed : int (0 - 500)
            - max speed that can move to goal position
            - Default is set to 500 dps

        -------
        Returns
        -------
        - status : list
            - List contains motor temp, torque current value, motor output speed, and motor output angle
            - status[0] : motor_temp (int8),        (1° Celcius / LSB)
                - Motor temp in Celcius
            - status[1] : torque_current (int16),   (0.01A / LSB)
                - Torque current in Amps
            - status[2] : output_speed (int16),     (1dps / LSB)
                - Output shaft speed in degrees per second (dps)
            - status[3] : output_angle (int16),     (1° / LSB)
                - Output shaft angle in degrees, range +-32767°
                - Testing saw angle range as +-3640°
        """
        angle = np.int32(angle / 0.01)

        command = [0xA8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        # Put respective bytes fro max speed into command
        speed_bytes = uint16_to_uint8(max_speed)
        command[2] = speed_bytes[0]
        command[3] = speed_bytes[1]
        
        # Put respective bytes for angle into command
        angle_bytes = int32_to_uint8(angle)
        command[4] = angle_bytes[0]
        command[5] = angle_bytes[1]
        command[6] = angle_bytes[2]
        command[7] = angle_bytes[3]
        reply, t = self._send_cmd(command, read)
        if reply is None:
            return None
            
        # Put in list and display the return status to console
        status = self.__convert_control_return(reply)
        print("Incremental Position Control:")
        self.__display_control_return(status)
        return status   
        

    def write_acceleration(self, control_type, acceleration:int):
        """
        Writes the acceleration into RAM and ROM.
        This will save after powering off motor.

        Debugger says max acceleration should be 10000,
        but is able to be set in range of 100-60000...
        Keep acceleration lower for testing movement functions.
        
        ----------
        Parameters
        ----------
        - control_type : uint8
            - 0x00 for position acceleration
            - 0x01 for position deceleration
            - 0x02 for speed acceleration
            - 0x03 for speed deceleration
        - acceleration : int
            - Acceleration in dps/s , Range:(100 - 60000)
        """
        # Dont write to motor if not in acceptable range
        if (acceleration < 100) | (acceleration > 60000):
            print("CANNOT WRITE ACCELERATION TO MOTOR. MUST BE IN RANGE (100 - 60000) DPS/S")
            return
        command = [0x43, control_type, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        accel_bytes = int32_to_uint8(acceleration)
        command[4] = accel_bytes[0]
        command[5] = accel_bytes[1]
        command[6] = accel_bytes[2]
        command[7] = accel_bytes[3]
        self._send_cmd(command)



    def set_timeout(self, timeout):
        """
        Sets the protection timeout to turn off motors when connection lost.
        Ex: 0.5 = 0.5 second timeout
        ----------
        Params
        ----------
        - timeout : uint32
            - Value in seconds, will be converted to ms

        """
        if timeout > 5:
            print("TIMEOUT SHOULD BE SET NO HIGHER THAN 5 SECONDS FOR INCREASED SAFETY.")
            return
        if timeout < 0:
            print("TIMEOUT CANNOT BE SET TO A NEGATIVE VALUE")
            return
        timeout = timeout*1000  
        command = [0xB3, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        timeout_bytes = int32_to_uint8(timeout)
        command[4] = timeout_bytes[0]
        command[5] = timeout_bytes[1]
        command[6] = timeout_bytes[2]
        command[7] = timeout_bytes[3]
        self._send_cmd(command)


    def set_baud(self, select:int):
        """
        Sets the CAN baudrate for the motor.
        
        ----------
        Params
        ----------
        - rate : uint8
            - Value either 0 or 1
            - 0 for 500,000 Kbps or 500 KHz
            - 1 for 1Mbps or 1 MHz

        """      
        if (select == 0) | (select == 1):
            command = [0xB4, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, select]
            self._send_cmd(command)
            try:
                os.system("sudo /sbin/ip link set can0 down")
                time.sleep(0.1)
            except Exception as e:
                print(e)

            try:
                if select==0:
                    os.system("sudo /sbin/ip link set can0 up type can bitrate 500000")
                else:
                    os.system("sudo /sbin/ip link set can0 up type can bitrate 1000000")
                time.sleep(0.1)
            except Exception as e:
                print(e)
        else:
            print("Motor Baud Rate value must be 0 or 1. \n0 = 500 KHz, 1 = 1MHz")



    #################### NEEDS MORE TESTING
    def write_canID(self, id):
        """
        Writes CAN ID to the motor
        
        ----------
        Parameters
        ----------
        - id : uint8
            Value from 1 - 32
        """
        # Dont write to motor if not in acceptable range
        if (id < 1) | (id > 32):
            print("CANNOT WRITE CAN ID TO MOTOR. MUST BE IN RANGE (1 - 32)")
            return
        
        id = np.uint8(id)
        command = [0x79, 0x00, 0, 0x00, 0x00, 0x00, 0x00, id]
        reply, t = self._send_cmd(command)
        if reply is None:
            return
        reply_id = uint8_to_uint16(reply[7], reply[6])
        reply_id = reply_id - 0x100
        # Only set the motors id param if a return message is recieved
        if reply_id == (id+0x140):   
            self.id = id + 0x140
                
    #################### NEEDS MORE TESTING
    def read_canID(self):
        """
        Reads CAN ID of the motor
        
        ----------
        Returns
        ----------
        - id : uint8
            Value from 1 - 32
        """      
        command = [0x79, 0x00, 1, 0x00, 0x00, 0x00, 0x00, 0x00]
        motor300 = MyActuatorMotor(self.motor_type, 0x300)
        reply, t = motor300._send_cmd(command, True)
        if reply is None:
            return None
        id = uint8_to_uint16(reply[7], reply[6])
        id = id - 0x100
        print("ID Converted:", hex(id))
        time.sleep(CAN_DELAY)



    def encoder_to_angle(self, encoder_value:int):
        """
        Converts encoder position value to an angle in degrees
        
        ----------
        Parameters
        ----------
        - encoder_value (int32)

        -------
        Returns
        -------
        - angle (int32)
            
        """
        angle = np.int32((encoder_value*360) / (65536*9))

        return angle

    def brake_release(self):
        """
        Releases holding brake, allowing motor to be in a moveable state

        """
        command = [0x77, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]        
        self._send_cmd(command)


    def brake_lock(self):
        """
        Enables holding brake, locking the motor at its current position
        """
        command = [0x78, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]        
        self._send_cmd(command)


    def stop(self):
        """
        Stops the motor
        """
        command = [0x81, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        self._send_cmd(command)


    def reset(self):
        """
        Resets the motor
        """
        command = [0x76, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
        self._send_cmd(command)   
        # Sleep for a time to allow  for reset  
        time.sleep(3)   
        print("Motor Reset")
    

class MyActuatorMotor_CAN(MyActuatorMotor):
    """
    MyActuatorMotor_CAN Class
        Used to create motor object that includes various control functions
        allowing for communication over CAN connection
    ---------------------
    Supported Motor Types
    ---------------------
    - "X6_S2_V2",
    - "X10_S2_V3"
    ----------
    Functions
    ----------
    - read_pid()   
    - display_pid()
    - pid_params_to_value(curr_KP, curr_KI, speed_KP, speed_KI, pos_KP, pos_KI): 
    - write_pid(save_type, curr_KP, curr_KI, speed_KP, speed_KI, pos_KP, pos_KI)
    - read_motor_status_1()
    - read_motor_status_2()
    - display_motor_status_1()
    - display_motor_status_2()
    - read_abs_position()
    - read_position()
    - read_home_position()
    - read_zero_offset()
    - display_abs_position()
    - display_position()
    - display_home_position()
    - display_zero_offset()
    - torque_control(current)
    - speed_control(speed)
    - position_tracking_control(angle)
    - absolute_position_control(angle, max_speed)
    - incremental_position_control(self, angle, max_speed)
    - write_acceleration(control_type, acceleration/deceleration)
    - set_timeout(timeout)  # Don't use
    - set_baud(select)
    - write_canID(id)   # Only use when one motor connected
    - read_canID()      # Only use when one motor connected
    - stop()
    - reset()


    """
    def __init__(self, motor_type, id):
        # Placed this info in super()
        # self.can_pub = rospy.Publisher("can_out", can_in, queue_size=1)
        # self.can_sub = rospy.Subscriber("can_in", can_in, self.response_callback)
        super().__init__(motor_type, id)


    def _send_cmd(self, command):
        """
        Sends command over CAN bus to a specified motor.
        
        Parameters
        ----------
        - command : list (uint_8) (8 bytes)
            - The hexidecimal command to be sent to the motor.
        -------
        Returns
        -------
        - response_msg : list (uint_8) (8 bytes)
            - List containing the 8 bytes of the response data.
            - One or two bytes of this data may be used to represent a parameter
            that was requested in the initial command.
        """
        return super()._send_cmd_CAN(command)


class MyActuatorMotor_UART(MyActuatorMotor):
    """
    MyActuatorMotor_UART Class
        Used to create motor object that includes various control functions
        allowing for communication over Serial connection
    ---------------------
    Supported Motor Types
    ---------------------
    - "X6_S2_V2",
    - "X10_S2_V3"
    ----------
    Functions
    ----------
    - read_pid()   
    - display_pid()
    - pid_params_to_value(curr_KP, curr_KI, speed_KP, speed_KI, pos_KP, pos_KI): 
    - write_pid(save_type, curr_KP, curr_KI, speed_KP, speed_KI, pos_KP, pos_KI)
    - read_motor_status_1()
    - read_motor_status_2()
    - display_motor_status_1()
    - display_motor_status_2()
    - read_abs_position()
    - read_position()
    - read_home_position()
    - read_zero_offset()
    - display_abs_position()
    - display_position()
    - display_home_position()
    - display_zero_offset()
    - torque_control(current)
    - speed_control(speed)
    - position_tracking_control(angle)
    - absolute_position_control(angle, max_speed)
    - incremental_position_control(self, angle, max_speed)
    - write_acceleration(control_type, acceleration/deceleration)
    - set_timeout(timeout)  # Don't use
    - set_baud(select)
    - write_canID(id)   # Only use when one motor connected
    - read_canID()      # Only use when one motor connected
    - stop()
    - reset()
    """
    def __init__(self, motor_type, id, port):
        # Placed this info in super()
        # serial_name = port
        # baud_rate = 115200
        # # timeout = 10
        # self.conn = serial.Serial(serial_name, baud_rate, timeout=0)
        super().__init__(motor_type, id, port)        

    def _send_cmd(self, command):
        """
        Sends command over CAN bus to a specified motor.
        
        Parameters
        ----------
        - command : list (uint_8) (8 bytes)
            - The hexidecimal command to be sent to the motor.
        -------
        Returns
        -------
        - response_msg : list (uint_8) (8 bytes)
            - List containing the 8 bytes of the response data.
            - One or two bytes of this data may be used to represent a parameter
            that was requested in the initial command.
        """
        return super()._send_cmd_UART(command)

    


    
            
    

