import cv2
import numpy as np
import vgamepad as vg 

class Controller():
	def __init__(self):
		self.states = {
			'accel': 0,
			'angle': 0,
			'brake': 0,
			'nitro': 0,
		}
		self.prev_states = {
			'accel': 0,
			'angle': 0,
			'brake': 0,
			'nitro': 0
		}
		self.gamepad = vg.VX360Gamepad()

	# Get rotation angle based on 2 anchor coordinates
	# return angle and anchor axis center
	def getAngle(self,a1,a2): # 2 blue anchors' centers
		a1 = np.array(a1)
		a2 = np.array(a2)
		mid_point = (a1+a2)/2
		mid2a2 = a2 - mid_point
		horz_line = (1,0)

		cos = np.inner(horz_line,mid2a2) / (np.linalg.norm(horz_line)*np.linalg.norm(mid2a2))
		rad = np.arccos(cos)
		deg = np.rad2deg(rad)

		if a2[1] > mid_point[1]:
			return deg, mid_point.astype(np.uint16)
		return -deg, mid_point.astype(np.uint16)

	

	# get states
	def updateStates(self,detections):
		
		self.states['nitro'] = 0
		self.states['brake'] = 0
		anchors = []
		for label, conf, bbox in detections:
			x,y = bbox[:2]
			if label == 'anchor':
				anchors.append((x,y))
			elif label == 'nitro':
				self.states['nitro'] = 1
			elif label == 'brake':
				self.states['brake'] = 1

		# steering, accelerating
		center = None
		if len(anchors) == 2:
			self.states['accel'] = 1
			if anchors[0][0] > anchors[1][0]:
				anchors[0],anchors[1] = anchors[1],anchors[0]
			self.states['angle'], center = self.getAngle(anchors[0],anchors[1])
			self.updateSteer()
		else:
			self.states['accel'] = 0

		# acellerating
		if self.states['accel'] != self.prev_states['accel']:
			self.updateAccel()
			self.prev_states['accel'] = self.states['accel'] 
		# brake/back
		if self.states['brake'] != self.prev_states['brake']:
			self.updateBrake()
			self.prev_states['brake'] = self.states['brake']
		# nitro
		if self.states['nitro'] != self.prev_states['nitro']:
			self.updateNitro()
			self.prev_states['nitro'] = self.states['nitro'] 

		return {
			'angle': self.states['angle'],
			'brake': self.states['brake'],
			'nitro': self.states['nitro'],
			'center': center
		}


	def updateAccel(self):
		if self.states['accel']:
			self.gamepad.right_trigger(value=150)
		else:
			self.gamepad.right_trigger(value=0)
		self.gamepad.update()

	def updateSteer(self):
		if -5 < self.states['angle'] < 5:
			self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
		else:
			strength = self.states['angle']/80
			self.gamepad.left_joystick_float(x_value_float=strength, y_value_float=0.0)
		self.gamepad.update()


	def updateBrake(self):
		if self.states['brake']:
			self.gamepad.left_trigger(value=255)
		else:
			self.gamepad.left_trigger(value=0)
		self.gamepad.update()

	def updateNitro(self):
		if self.states['nitro']:
			self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
		else:
			self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
		self.gamepad.update()


	def terminate(self):
		self.gamepad.right_trigger(value=0)
		self.gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
		self.gamepad.left_trigger(value=0)
		self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
		print("Controller terminated")