import numpy as np
import numpy as np
import matplotlib.pylab as plt
import sys
import lmfit
from lmfit import Model
from lmfit import Parameters, fit_report, minimize
from matplotlib.gridspec import GridSpec
import sys
sys.path.append("/usr/local/bin/cmaps")
import cmap_califa
import cmap_vfield
califa=cmap_vfield.CALIFA()
pixel_scale = 0.2

def pa_positive(Pos_Ang):
	pa_i=Pos_Ang
	if  pa_i > 180:
		pa_i = pa_i -180
	if -180 < pa_i < 0:
		pa_i = abs(pa_i)
	if pa_i < -180:
		pa_i = pa_i +180
	return pa_i


def inc_positive(inclination):
	inc_i = inclination
	if inc_i > 90:
		inc_i = 180-inc_i
	if inc_i < 0:
		inc_i = abs(inc_i)
	return inc_i

 
def Rings(xy_mesh,pa,inc,x0,y0):
	(x,y) = xy_mesh
	X = -(x-x0)*np.sin(pa)+(y-y0)*np.cos(pa)
	Y = ((x-x0)*np.cos(pa)+(y-y0)*np.sin(pa))/np.cos(inc)
	R= np.sqrt(X**2+Y**2)
	#R[R<5] = -1#np.nan
	#plt.imshow(R,origin = "l",cmap = califa)
	#plt.show()

	return R#+1

def Rings0(xy_mesh,pa,inc,x0,y0):

	(x,y) = xy_mesh
	X = -(x-x0)*np.sin(pa)+(y-y0)*np.cos(pa)
	Y = ((x-x0)*np.cos(pa)+(y-y0)*np.sin(pa))/np.cos(inc)
	R= np.sqrt(X**2+Y**2)
	return R



def Rings_r_1(R,vel,ring,delta=0):
	[ny,nx]= vel.shape
	M=np.ones((ny,nx))

	if ring == 0:
		mask = (R>=ring) & (R <= ring+delta)
		#mask = R==0
	else:
		#ring = ring +1
		ring = ring +delta
		mask = (R>=ring) & (R <= ring+delta)		
	
	s = M*mask
	return s



def Rings_r_2(R,vel,ring,delta=0):
	[ny,nx]= vel.shape
	M=np.ones((ny,nx))

	if ring == 0:
		mask = (R>=ring) & (R <= ring+delta)
		#mask = R==0
	else:
		#ring = ring +1
		ring = ring*delta +delta
		mask = (R>=ring) & (R <= ring+delta)		
	
	s = M*mask
	return s


def Rings_r_3(R,vel,ring,delta=0):
	[ny,nx]= vel.shape
	M=np.ones((ny,nx))

	if ring == 0:
		mask = (R>=ring) & (R <= ring+delta)
	else:
		mask = (R>=ring) & (R <= ring+delta)		
	
	s = M*mask
	return s


def Rings_r_4(R,vel,ring,delta=0):
	[ny,nx]= vel.shape
	M=np.ones((ny,nx))

	if ring == 0:
		mask = (R>=ring) & (R <= ring+delta)
	else:
		mask = (R>=ring-0.5*delta) & (R <= ring+0.5*delta)		
	
	s = M*mask
	#plt.imshow(s)
	#plt.show()
	return s

def Vlos_BISYM(xy_mesh,Vrot,Vr2,pa,inc,x0,y0,Vsys):
	(x,y) = xy_mesh
	pa,inc=(pa)*np.pi/180,inc*np.pi/180
	R  = Rings(xy_mesh,pa,inc,x0,y0)
	cos_tetha = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))/R
	sin_tetha = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))/(np.cos(inc)*R)
	#vlos =  Vrot*cos_tetha*np.sin(inc)+Vsys
	#m = 2
	#vlos = Vsys+np.sin(inc)*((Vrot*cos_tetha)-Vt2*np.cos(m*theta_b*np.pi/180)*cos_tetha - Vr2*np.sin(m*theta_b*np.pi/180)*sin_tetha)
	vlos = Vsys+np.sin(inc)*(Vrot*cos_tetha + Vr2*sin_tetha)
	return np.ravel(vlos)


def Vlos_ROT(xy_mesh,Vrot,pa,inc,x0,y0,Vsys):
	(x,y) = xy_mesh
	pa,inc=(pa)*np.pi/180,inc*np.pi/180
	R  = Rings(xy_mesh,pa,inc,x0,y0)
	cos_tetha = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))/R
	sin_tetha = (- (x-x0)*np.cos(pa) - (y-y0)*np.sin(pa))/(np.cos(inc)*R)
	vlos =  Vrot*cos_tetha*np.sin(inc)+Vsys
	return np.ravel(vlos)



def Weight(xy_mesh,Vrot,pa,inc,x0,y0,Vsys):
	(x,y) = xy_mesh
	pa,inc=(pa)*np.pi/180,inc*np.pi/180
	R  = Rings0(xy_mesh,pa,inc,x0,y0)
	cos_tetha = (- (x-x0)*np.sin(pa) + (y-y0)*np.cos(pa))/R
	abs_cos = abs(cos_tetha) 
	return np.ravel(abs_cos)

def xi_sq(xy_mesh,weight,guess,data=None):
	vrot0,vr20,pa0,inc0,x0,y0,vsys0 = guess
	model = Vlos_BISYM(xy_mesh,vrot0,vr20,pa0,inc0,x0,y0,vsys0)
	#weight = 1
	objective = (model-data)*weight
	return np.nansum(objective**2)


def polynomial(x,a0,a1,a2,a3,a4,a5):
	x = np.asarray(x)
	y = a0 + a1*x +a2*x**2 +a3*x**3 + a4*x**4 + a5*x**5
	return y

def linear(x,a0,a1):
	x = np.asarray(x)
	y = a0 + a1*x 
	return y

def fit_polynomial(x,dato):
	x = np.asarray(x)
	dato = np.asarray(dato)


	def residual_line(pars,x,data=None):
		parvals = pars.valuesdict()
		a0 = parvals['a0']
		a1 = parvals['a1']
		a2 = parvals['a2']
		a3 = parvals['a3']
		a4 = parvals['a4']
		a5 = parvals['a5']
		model = polynomial(x,a0,a1,a2,a3,a4,a5)
		objective = model - data
		return objective**2


	fit_param = Parameters()
	fit_param.add('a0', value=0)
	fit_param.add('a1', value=0)
	fit_param.add('a2', value=0)
	fit_param.add('a3', value=0)
	fit_param.add('a4', value=0)
	fit_param.add('a5', value=0)

	out = minimize(residual_line, fit_param, args=(x,), kws={'data': dato},method='Powell', nan_policy = "omit")#, method = "emcee")
	best = out.params
	a0, a1, a2, a3, a4, a5 = best["a0"].value,best["a1"].value, best["a2"].value, best["a3"].value, best["a4"].value, best["a5"].value
	return a0, a1, a2, a3, a4, a5


def fit_linear(x,dato):
	x = np.asarray(x)
	dato = np.asarray(dato)


	def residual_line(pars,x,data=None):
		parvals = pars.valuesdict()
		a0 = parvals['a0']
		a1 = parvals['a1']
		model = linear(x,a0,a1)
		objective = model - data
		return objective**2


	fit_param = Parameters()
	fit_param.add('a0', value=1e3)
	fit_param.add('a1', value=0)

	out = minimize(residual_line, fit_param, args=(x,), kws={'data': dato},method='Powell')#, method = "emcee")
	best = out.params
	a0, a1 = best["a0"].value,best["a1"].value
	return a0, a1

def fit(param,vel_val,e_vel,xy_mesh,guess,vary,mode,sigma, model = False):
	vrot0,vr20,pa0,inc0,X0,Y0,vsys0 = guess
	#[ny,nx] = 
	#MODEL = np.zeros((ny,nx))

	n_sigma = len(sigma)
	if n_sigma != 0:
		e_vrot,e_vr2,e_pa,e_inc,e_x0,e_y0,e_vsys = sigma


	if mode == "bisymetric":

		XC,YC,V0,Pos_Ang,inclination,theta=[],[],[],[],[],[]
		rings = []


		def residual(pars,xy_mesh,data=None, weight = e_vel):
			# unpack parameters: extract .value attribute for each parameter
			parvals = pars.valuesdict()
			pa = parvals['pa']
			inc = parvals['inc']
			Vsys = parvals['Vsys']
			Vrot = parvals['Vrot']
			Vr2 = parvals['Vr2']
			x0,y0 = guess[4],guess[5]

			model = Vlos_BISYM(xy_mesh,Vrot,Vr2,pa,inc,x0,y0,Vsys)
			#model = Vlos_ROT(xy_mesh,Vrot,Vr2,pa,inc,x0,y0,Vsys)
			#weight = Weight(xy_mesh,Vrot,pa,inc,x0,y0,Vsys)
			#objective = (model-data)/weight

			objective = (model-data)*weight
			return objective**1
			#return objective**2


		fit_params = Parameters()
		if n_sigma == 0:

				fit_params.add('pa', value=pa0, vary = vary[2], min = 0, max = 360)
				fit_params.add('inc', value=inc0, vary = vary[3], min = 0, max = 90)
				fit_params.add('Vsys', value=vsys0, vary = vary[6])#, min = vsys0 -50, max = vsys0 + 50 )
				fit_params.add('Vr2', value=vr20, vary = vary[1])
				fit_params.add('Vrot', value=vrot0, vary = vary[0], min = 0)
		else:
				fit_params.add('pa', value=pa0, vary = vary[2], min = pa0-e_pa, max = pa0+e_pa)
				fit_params.add('inc', value=inc0, vary = vary[3], min =inc0-e_inc, max = inc0+e_inc)
				fit_params.add('Vsys', value=vsys0, vary = vary[6], min = vsys0-e_vsys, max = vsys0+e_vsys)
				fit_params.add('Vr2', value=vr20, vary = vary[1])
				fit_params.add('Vrot', value=vrot0, vary = vary[0])#, min = 0, max = 500)
				#fit_params.add('Vrot', value=vrot0, vary = vary[0], min = vrot0-e_vrot, max = vrot0+e_vrot)#, min = 0, max = 500)


		out = minimize(residual, fit_params, args=(xy_mesh,), kws={'data': np.ravel(vel_val)},method='leastsq')#,method='Powell')#, method = "emcee")
		best = out.params
		pa, inc, Vsys, Vr2, Vrot = best["pa"].value,best["inc"].value, best["Vsys"].value, best["Vr2"].value, best["Vrot"].value
		x0,y0  = guess[4],guess[5]

		#def Vlos_BISYM(xy_mesh,Vrot,Vr2,pa,inc,x0,y0,Vsys):
		#MODEL = MODEL + Vlos_BISYM(xy_mesh,Vrot,Vr2,pa,inc,x0,y0,Vsys)

		best_params = [Vrot,Vr2,pa,inc,X0,Y0,Vsys]
		N = len(vel_val)
		xi = xi_sq(xy_mesh,e_vel,best_params,np.ravel(vel_val))

		#if model == True:
		#	return Vrot , Vr2, Vsys,  pa, inc ,MODEL, xi/(N-3)
		#else:

		return Vrot , Vr2, Vsys,  pa, inc ,xi/(N-2)





	if mode == "rotation":

		XC,YC,V0,Pos_Ang,inclination,theta=[],[],[],[],[],[]
		rings = []


		def residual(pars,xy_mesh,data=None, weight = e_vel):
			# unpack parameters: extract .value attribute for each parameter
			parvals = pars.valuesdict()
			pa = parvals['pa']
			inc = parvals['inc']
			Vsys = parvals['Vsys']
			Vrot = parvals['Vrot']
			x0,y0 = guess[4],guess[5]

			model = Vlos_ROT(xy_mesh,Vrot,pa,inc,x0,y0,Vsys)

			#weight = Weight(xy_mesh,Vrot,pa,inc,x0,y0,Vsys)
			#objective = (model-data)/weight
			objective = (model-data)*weight
			return objective**2


		fit_params = Parameters()
		if n_sigma == 0:
			fit_params.add('pa', value=pa0, vary = vary[2], min = 0, max = 360)
			fit_params.add('inc', value=inc0, vary = vary[3], min = 0, max = 90)
			fit_params.add('Vsys', value=vsys0, vary = vary[6])
			fit_params.add('Vrot', value=vrot0, vary = vary[0], min = 0, max = 500)
		else:
			fit_params.add('pa', value=pa0, vary = vary[2], min = pa0-e_pa, max = pa0+e_pa)
			fit_params.add('inc', value=inc0, vary = vary[3], min = inc0-e_inc, max = inc0+e_inc)
			fit_params.add('Vsys', value=vsys0, vary = vary[6], min = vsys0-e_vsys, max = vsys0+e_vsys)
			fit_params.add('Vrot', value=vrot0, vary = vary[0], min = 0, max = 500)


		out = minimize(residual, fit_params, args=(xy_mesh,), kws={'data': np.ravel(vel_val)},method='Powell')#, method = "emcee")
		best = out.params
		pa, inc, Vsys, Vrot = best["pa"].value,best["inc"].value, best["Vsys"].value, best["Vrot"].value
		x0,y0  = guess[4],guess[5]
		Vr2 = 0
		N = len(vel_val)
		xi = xi_sq(xy_mesh,e_vel,guess,np.ravel(vel_val))

		return Vrot , Vr2, Vsys,  pa, inc ,xi/(N-3)








