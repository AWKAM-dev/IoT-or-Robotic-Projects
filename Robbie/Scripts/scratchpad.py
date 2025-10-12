import math

AB = 3
AE = int(input("Enter distance: ").strip())
DE = 3.5
BC = 16
CD = 10.5

#Triangle ABE
BE = math.sqrt(AB**2 + AE**2)
print(f"BE is {BE}")
BEA = math.degrees(math.acos(AE/BE))
print(f"Angle BEA is {BEA}")
ABE = 90 - BEA

#Triangle BDE
BED = ABE
BD = math.sqrt(BE**2 + DE**2 - 2*BE*DE*(math.cos(math.radians(BED))))
print(f"BD is {BD}")

try:
	EBD = math.degrees(math.acos((DE**2 - BD**2 - BE**2)/(-2*BD*BE)))
	print(f"EBD is {EBD}")
except Exception as e:
	print(f"[ERROR]: {e} for trianglr BDE")
	EBD = 0

#Triangle FBD
FBD = 90 - ABE - EBD

#Traingle BCD
try:
	BCD = math.degrees(math.acos((BD**2 - BC**2 - CD**2)/(-2*BC*CD)))
	print(f"BCD is {BCD}")
	CBD = math.degrees(math.acos((CD**2-BC**2-BD**2)/(-2*BC*BD)))
	print(f"CBD is {CBD}")
except Exception as e:
	print(f"[ERROR]: {e} for triangle BCD")
	BCD = 0
	CBD = 0

#Servos
botServo = 90 - CBD - FBD
print(f"Bottom servo is at: {botServo}")
topServo = 180 - BCD
print(f"Top servo is at: {topServo}")
