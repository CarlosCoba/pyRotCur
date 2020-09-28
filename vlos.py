 
import numpy as np

def Rings(x,y,pa,inc,x0,y0):
	X = -(x-x0)*np.sin(pa)+(y-y0)*np.cos(pa)
	Y = ((x-x0)*np.cos(pa)+(y-y0)*np.sin(pa))/np.cos(inc)
	R= np.sqrt(X**2+Y**2)
	return R

def Vlos(x,y,Vrot,Vr2,pa,inc,x0,y0,Vsys):
	pa,inc=(pa)*np.pi/180,inc*np.pi/180
	X = -(x-x0)*np.sin(pa)+(y-y0)*np.cos(pa)
	Y = ((x-x0)*np.cos(pa)+(y-y0)*np.sin(pa))/np.cos(inc)
	R= np.sqrt(X**2+Y**2)
	cos_tetha = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))/R
	sin_tetha = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))/(np.cos(inc)*R)
	vlos = Vsys+np.sin(inc)*(Vrot*cos_tetha + Vr2*sin_tetha)
	return vlos
