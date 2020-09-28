import numpy as np
import numpy as np
import matplotlib.pylab as plt
import sys
import lmfit
import random
from lmfit import Model
from lmfit import Parameters, fit_report, minimize
from matplotlib.gridspec import GridSpec
from astropy.stats import sigma_clip
from scipy.interpolate import interp1d
from vlos import Vlos
from axis import AXIS
import random
import sys
sys.path.append("/usr/local/bin/cmaps")
import cmap_califa
import cmap_vfield
califa=cmap_vfield.CALIFA()

 
 
from pixel_params import pixels
import fit_params
from fit_params import fit
from fit_params import fit_polynomial
from fit_params import fit_linear


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
	PA,inc=(pa)*np.pi/180,inc*np.pi/180
	(x,y) = xy_mesh
	X = -(x-x0)*np.sin(PA)+(y-y0)*np.cos(PA)
	Y = ((x-x0)*np.cos(PA)+(y-y0)*np.sin(PA))/np.cos(inc)
	R= np.sqrt(X**2+Y**2)
	#R[R<5] = -1#np.nan
	#plt.imshow(R,origin = "l",cmap = califa)
	#plt.show()

	return R#+1

def Rings0(xy_mesh,pa,inc,x0,y0):
	PA,inc=(pa)*np.pi/180,inc*np.pi/180
	(x,y) = xy_mesh
	X = -(x-x0)*np.sin(PA)+(y-y0)*np.cos(PA)
	Y = ((x-x0)*np.cos(PA)+(y-y0)*np.sin(PA))/np.cos(inc)
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
	R  = Rings(xy_mesh,pa,inc,x0,y0)
	PA,inc=(pa)*np.pi/180,inc*np.pi/180
	cos_tetha = (- (x-x0)*np.sin(PA) + (y-y0)*np.cos(PA))/R
	sin_tetha = (- (x-x0)*np.cos(PA) - (y-y0)*np.sin(PA))/(np.cos(inc)*R)
	#vlos =  Vrot*cos_tetha*np.sin(inc)+Vsys
	#m = 2
	#vlos = Vsys+np.sin(inc)*((Vrot*cos_tetha)-Vt2*np.cos(m*theta_b*np.pi/180)*cos_tetha - Vr2*np.sin(m*theta_b*np.pi/180)*sin_tetha)
	vlos = Vsys+np.sin(inc)*(Vrot*cos_tetha + Vr2*sin_tetha)
	return np.ravel(vlos)


def Vlos_ROT(xy_mesh,Vrot,pa,inc,x0,y0,Vsys):#,Vt2=0,Vr2=0,theta_b=0):
	(x,y) = xy_mesh
	R  = Rings0(xy_mesh,pa,inc,x0,y0)
	#print "R=",R
	PA,inc=(pa-90*0)*np.pi/180,inc*np.pi/180
	cos_tetha = (- (x-x0)*np.sin(PA) + (y-y0)*np.cos(PA))/R
	sin_tetha = (- (x-x0)*np.cos(PA) - (y-y0)*np.sin(PA))/(np.cos(inc)*R)
	vlos =  Vrot*cos_tetha*np.sin(inc)+Vsys
	return np.ravel(vlos)



def Vrot_MODEL(xy_mesh,vlos,pa,inc,x0,y0,Vsys):#,Vt2=0,Vr2=0,theta_b=0):
	(x,y) = xy_mesh
	R  = Rings(xy_mesh,pa,inc,x0,y0)
	PA,inc=(pa-90*0)*np.pi/180,inc*np.pi/180
	cos_tetha = (- (x-x0)*np.sin(PA) + (y-y0)*np.cos(PA))/R
	sin_tetha = (- (x-x0)*np.cos(PA) - (y-y0)*np.sin(PA))/(np.cos(inc)*R)
	vrot = (vlos - Vsys)/cos_tetha*np.sin(inc)
	return vrot


def Weight(xy_mesh,Vrot,pa,inc,x0,y0,Vsys):
	(x,y) = xy_mesh
	PA,inc=(pa)*np.pi/180,inc*np.pi/180
	R  = Rings0(xy_mesh,pa,inc,x0,y0)
	cos_tetha = (- (x-x0)*np.sin(PA) + (y-y0)*np.cos(PA))/R
	abs_cos = abs(cos_tetha) 
	return np.ravel(abs_cos)




#from emcee_fit import EMCEE

vmaxrot = 400
def RC_sigma(vel,evel,nrings,guess0,vary,sigma = [],mode = "rotation", delta=1,ring = "pixel",rstart = 4,iter = 3, pos = 2, plot = False,model = False ,pixel_scale = 1):

	#rstart = int(rstart/pixel_scale)

	vrot0,vr20,pa0,inc0,X0,Y0,vsys0 = guess0
	guess = [vrot0,vr20,pa0,inc0,X0,Y0,vsys0]
	n_sigma = len(sigma)


	if n_sigma != 0:
		e_vrot,e_vr2,e_pa,e_inc,e_x0,e_y0,e_vsys = sigma


	[ny,nx] = vel.shape

	x = np.arange(0, nx, 1)
	y = np.arange(0, ny, 1)
	mesh = np.meshgrid(x,y,sparse=True)

	VLOS= np.zeros((ny,nx))
	VROT = np.zeros((ny,nx))
	VLOS= np.zeros((ny,nx))
	VROT = np.zeros((ny,nx))
	VT2 = np.zeros((ny,nx))
	VR2 = np.zeros((ny,nx))
	MODEL = np.zeros((ny,nx))


	nrings = nrings*pixel_scale
	#ring_position = np.arange(rstart,nrings,pos)
	ring_position_arc = np.arange(rstart,nrings,1)
	ring_position_pix = np.arange(rstart,nrings,1)/pixel_scale
	ring_position = ring_position_arc


	vsys = []
	r = np.array([])
	vsys_it =  np.array([]) 
	pa_free = np.array([])
	guess0 = [vrot0,vr20,pa0,inc0,X0,Y0,vsys0]
	pa_it_0 = []
	inc_int_0 = []
	frac_pix = 0.3333

	for n_iter in range(5):

		vary = [True, True, True,True,False,False,True]
		#print(ring_position)
		guess_test = guess0
		for i in ring_position:
				#XY_mesh, vel_val, e_vel, f_pixel = pixels(vel,evel,guess,ringpos = i, delta=2+n_iter/2.,ring = "pixel",pixel_scale=pixel_scale)
				XY_mesh, vel_val, e_vel, f_pixel = pixels(vel,evel,guess,ringpos = i, delta=2+n_iter/2.,ring = "ARCSEC",pixel_scale=pixel_scale)
				if f_pixel > frac_pix:
						Vrot , Vr2, Vsys,  pa, inc ,xi = fit("vsys",vel_val,e_vel,XY_mesh,guess,vary,mode,sigma)
						guess00 = [Vrot , 0,  pa, inc,X0,Y0,Vsys]
						guess_test = guess00
						if Vrot > 10 and Vrot < vmaxrot:
							vsys_it = np.append(vsys_it,Vsys)
							r = np.append(r,i)
							pa_free = np.append(pa_free,pa)



	sigma_c_vsys = sigma_clip(vsys_it,sigma=2, maxiters=2)
	vsys_it0 = np.nanmean(sigma_c_vsys)


	sigma_c_pa = sigma_clip(pa_free,sigma=2, maxiters=2)
	pa_free0 = np.nanmean(sigma_c_pa)


	"""
	fig=plt.figure(figsize=(3,2.))
	gs2 = GridSpec(1, 1)
	gs2.update(left=0.15, right=0.99,top=0.99,hspace=0.0,bottom=0.19,wspace=0)
	ax0=plt.subplot(gs2[0,0])

	ax0.scatter(r,pa_free,marker = "o", color = "crimson",s = 9,zorder = 0)
	ax0.scatter(r,sigma_c_pa,marker = "o", color = "dodgerblue",s = 10,zorder = 1)

	#ax0.scatter(r,vsys_it,marker = "o", color = "crimson",s = 9,zorder = 0)
	#ax0.scatter(r,sigma_c_vsys,marker = "o", color = "dodgerblue",s = 10,zorder = 1)


	ax0.set_xlabel("Ring position (arcsec)",fontsize = 10)

	ax0.set_ylabel("P.A. (degree)",fontsize = 10)
	#ax0.set_ylabel("$v_\mathrm{sys}~~(km/s)$",fontsize = 10)

	AXIS(ax0,tickscolor = "k")
	ax0.set_facecolor('#e8ebf2')
	#ax0.set_ylim(int(np.min(vsys_it))-10,int(np.max(vsys_it))+10)
	#ax0.set_ylim(int(np.min(pa_free))-10,int(np.max(pa_free))+10)
	#plt.savefig("/home/carlos/Documents/PhD_Thesis/figures/pa.png",dpi = 300)
	plt.show()
	"""

	vrot_it = [vrot0]
	vrot_it_2 = [vrot0]
	vrot_it_3 = [vrot0]
	guess00 = [vrot_it[0],vr20,pa_free0,inc0,X0,Y0,vsys_it0]
	pa_it_1,inc_it_1,vr_it_1,vr2_it_1,vsys_it_1, r_gal_1 = [],[],[],[],[], []
	pa_it_2,inc_it_2,vr_it_2,vr2_it_2,vsys_it_2, r_gal_2 = np.asarray([]),np.asarray([]),np.asarray([]),np.asarray([]),np.asarray([]),np.asarray([])
	pa_it_3,inc_it_3,vr_it_3,vr2_it_3,vsys_it_3, r_gal_3 = np.asarray([]),np.asarray([]),np.asarray([]),np.asarray([]),np.asarray([]),np.asarray([])
	pa_it_4,inc_it_4,vr_it_4,vr2_it_4,vsys_it_4, r_gal_4 = np.asarray([]),np.asarray([]),np.asarray([]),np.asarray([]),np.asarray([]),np.asarray([])
	pa_it,inc_it,vr_it,vr2_it = [],[],[],[]
	r = np.asarray([])
	for n_iter in range(10):
		vary = [True,True,True,True,False,False,False]
		ring_position = np.arange(rstart,nrings,1+n_iter)
		for i in ring_position:
			#XY_mesh, vel_val, e_vel, f_pixel = pixels(vel,evel,guess0,ringpos = i, delta=2+n_iter/2.,ring = "arcsec",pixel_scale=pixel_scale)
			XY_mesh, vel_val, e_vel, f_pixel = pixels(vel,evel,guess0,ringpos = i, delta=2+n_iter/2.,ring = "ARCSEC",pixel_scale=pixel_scale)
			if f_pixel > frac_pix:
					Vrot , Vr2, Vsys,  pa, inc ,xi = fit("vsys",vel_val,e_vel,XY_mesh,guess00,vary,mode,sigma)
					if Vrot>10 and Vrot<vmaxrot:
					#if 1>0:
						pa_it.append(pa)
						inc_it.append(inc)
						r = np.append(r,i)




		#a=fit_polynomial(r,pa_it)
		#plt.plot(r,fit_params.polynomial(r,a[0],a[1],a[2],a[3],a[4],a[5]),"ro")
		#b=fit_polynomial(r,inc_it)
		#plt.plot(r,fit_params.polynomial(r,b[0],b[1],b[2],b[3],a[4],a[5]),"bo")
		

		#
		#  THE MASKED IN SIGMA CLIP IS TRUE for clipped values: 
		# You have to remove or replace the TRUE values
		#
		#


		pa_it_array = np.asarray(pa_it)
		sigma_c_pa = sigma_clip(pa_it_array,sigma=1, maxiters=1)
		mask_pa = sigma_c_pa.mask 
		pa_it_array[mask_pa] = np.nanmean(sigma_c_pa)

		legfit = np.polynomial.legendre.legfit(r,pa_it_array, 5)
		legval_pa = np.polynomial.legendre.legval(r,legfit)


		inc_it_array = np.asarray(inc_it)
		sigma_c_inc = sigma_clip(inc_it_array,sigma=1, maxiters=1)
		mask_inc = sigma_c_inc.mask 
		inc_it_array[mask_inc] = np.nanmean(sigma_c_inc)

		legfit = np.polynomial.legendre.legfit(r,inc_it_array, 5)
		legval_inc = np.polynomial.legendre.legval(r,legfit)


		"""

		fig=plt.figure(figsize=(3,2.))
		gs2 = GridSpec(1, 1)
		gs2.update(left=0.15, right=0.99,top=0.99,hspace=0.0,bottom=0.19,wspace=0)
		ax0=plt.subplot(gs2[0,0])

		ax0.plot(r,legval_pa,"k-",markersize = 1,alpha = 0.7)
		ax0.scatter(r,pa_it_array,marker = "o", color = "dodgerblue",s = 9,zorder = 2)
		ax0.scatter(r,pa_it,marker = "o", color = "crimson",s = 9,zorder = 0)

		#ax0.plot(r,legval_inc,"k-",markersize = 1,alpha = 0.7)
		#ax0.scatter(r,inc_it_array,marker = "o", color = "dodgerblue",s = 9,zorder = 2)
		#ax0.scatter(r,inc_it,marker = "o", color = "crimson",s = 9,zorder = 0)



		#ax0.plot(r,legval_inc,"r-",markersize = 1,alpha = 0.7)
		#ax0.plot(r,inc_it,"ko",markersize = 2)


		ax0.set_xlabel("Ring position (arcsec)",fontsize = 10)

		#ax0.set_ylabel("P.A. (degree)",fontsize = 10)
		ax0.set_ylabel("inc. (degree)",fontsize = 10)


		#ax0.set_ylim(int(np.min(pa_it))-10,int(np.max(pa_it))+10)
		ax0.set_ylim(int(np.min(inc_it))-10,int(np.max(inc_it))+10)

		AXIS(ax0,tickscolor = "k")
		ax0.set_facecolor('#e8ebf2')
		#plt.savefig("/home/carlos/Documents/PhD_Thesis/figures/inc_legendre.png",dpi = 300)
		plt.show()

		"""


		f_pa = interp1d(r,legval_pa)
		f_inc = interp1d(r,legval_inc)



		#
		# Estimate vsys
		#

		k = 0
		vr_it_0 = 50
		vr2_it_0 = 0
		vsys_it_n= vsys_it0
		for i in ring_position:
			#XY_mesh, vel_val, e_vel, f_pixel = pixels(vel,evel,guess0,ringpos = i, delta=2+n_iter/2.,ring = "arcsec",pixel_scale=pixel_scale)
			XY_mesh, vel_val, e_vel, f_pixel = pixels(vel,evel,guess0,ringpos = i, delta=1+n_iter/2.,ring = "ARCSEC",pixel_scale=pixel_scale)
			if f_pixel > frac_pix:
					vary =  [True,True,False,False,False,False,True]
					if k>=len(legval_pa): 
						k = -1
					guess = [vr_it_0,vr2_it_0,legval_pa[k],legval_pa[k],X0,Y0,vsys_it_n]
					Vrot , Vr2, Vsys,  pa, inc ,xi = fit("vsys",vel_val,e_vel,XY_mesh,guess,vary,mode,sigma)
					k = k+1
					if Vrot > 10 and Vrot<vmaxrot:
						vr2_it_0 = Vr2
						vr_it_0 = Vrot
						vsys_it_n = Vsys

						vrot_it.append(Vrot)
						vr_it_1.append(Vrot)
						vsys_it_1.append(Vsys)
						r_gal_1.append(i)



		sigma_c = sigma_clip(vsys_it_1,sigma=2, maxiters=3)
		mask_vsys = sigma_c.mask 
		vsys_it0 = np.nanmean(sigma_c)


		"""
		fig=plt.figure(figsize=(3,2.))
		gs2 = GridSpec(1, 1)
		gs2.update(left=0.15, right=0.99,top=0.99,hspace=0.0,bottom=0.19,wspace=0)
		ax0=plt.subplot(gs2[0,0])

		ax0.plot(r_gal_1,vsys_it_1,"ko",markersize = 1)
		#ax0.plot(r,pa_it,"ko",markersize = 1)


		#ax0.plot(r,legval_inc,"r-",markersize = 1)
		#ax0.plot(r,inc_it,"ko",markersize = 1)


		ax0.set_xlabel("Ring position (arcsec)",fontsize = 10)
		ax0.set_ylabel("vsys (km/s)",fontsize = 10)
		AXIS(ax0,tickscolor = "k")
		ax0.set_facecolor('#e8ebf2')
		#plt.savefig("/home/carlos/Documents/PhD_Thesis/figures/inc.png",dpi = 300)
		plt.show()
		"""


	vsys_it_1 = np.asarray(vsys_it_1)


	"""
	fig=plt.figure(figsize=(3,2.))
	gs2 = GridSpec(1, 1)
	gs2.update(left=0.15, right=0.99,top=0.99,hspace=0.0,bottom=0.19,wspace=0)
	ax0=plt.subplot(gs2[0,0])


	binwidth = 2
	min_v,max_v=int(np.min(vsys_it_1)),int(np.max(vsys_it_1))+1
	bins=np.arange(min_v,max_v+0.01,binwidth)



	n, bins, patches = ax0.hist(vsys_it_1,edgecolor='k',bins=bins,linewidth=0.1 )
	ax0.set_xlabel("$v_\mathrm{sys}$ (km/s)")
	ax0.set_ylabel("Frecuency")
	AXIS(ax0,tickscolor = "k")
	ax0.set_facecolor('#e8ebf2')
	#plt.savefig("/home/carlos/Documents/PhD_Thesis/figures/vsys_hist.png",dpi = 300)
	plt.show()
	"""

	sigma_c_vsys = sigma_clip(vsys_it_1,sigma=1, maxiters=2)
	vsys_it_ = vsys_it_1[~sigma_c_vsys.mask]
	vsys_final = np.nanmedian(vsys_it_ )
	
	# The final values of vsys
	median_vsys = vsys_final
	sigma_vsys = np.nanstd(vsys_it_1 )


	#
	# Prepare initial conditions for the next iteration
	#

	#PA:
	pa_it = np.asarray(pa_it)
	sigma_c_pa = sigma_clip(pa_it,sigma=2, maxiters=3)
	mask_pa = sigma_c_pa.mask 
	pa_it_next = np.nanmean(pa_it[~mask_pa])


	pa_it[mask_pa] = pa_it_next
	legfit = np.polynomial.legendre.legfit(r,pa_it, 5)
	legval_pa = np.polynomial.legendre.legval(r,legfit)


	#plt.plot(r,pa_it,"ko")
	#plt.plot(r,sigma_c_pa,"ro")
	#plt.plot(r,legval_pa,"b-")
	#plt.show()


	#inc:
	inc_it = np.asarray(inc_it)
	sigma_c = sigma_clip(inc_it,sigma=2, maxiters=3)
	mask_inc = sigma_c.mask 
	inc_it_next = np.nanmean(sigma_c)
	inc_it[mask_inc] = inc_it_next
	legfit = np.polynomial.legendre.legfit(r,inc_it, 5)
	legval_inc = np.polynomial.legendre.legval(r,legfit)



	r_sort, vr_sort = zip(*sorted(zip(r_gal_1, vr_it_1)))
	max_r = int(np.nanmax(r_sort))

	r_sort = np.asarray(r_sort)
	vr_sort = np.asarray(vr_sort)


	vr_mean_ring = np.asarray([])
	r_mean_ring = np.asarray([]) 
	for kk in range(max_r+1):
		mask = r_sort == kk
		s = vr_sort[mask]
		if len(s)>0:
			vr_mean_ring = np.append(vr_mean_ring,np.nanmean(s))
			r_mean_ring = np.append(r_mean_ring,kk)

		else:
			vr_mean_ring = np.append(vr_mean_ring,20)
			r_mean_ring = np.append(r_mean_ring,kk)


	legfit = np.polynomial.legendre.legfit(r_mean_ring,vr_mean_ring, 5)
	legval_vrot = np.polynomial.legendre.legval(r_mean_ring,legfit)


	f = interp1d(r_mean_ring,legval_vrot)
	def vr_interp(w):
		try:
			Vr_interp = f(w)
		except (ValueError):
			Vr_interp = 60

		if np.isfinite(Vr_interp) == True:
			return Vr_interp
		else:
			return 50
		

	#plt.plot(r_mean_ring,legval_vrot,"b-")
	#plt.plot(r_mean_ring,vr_mean_ring,"ko")
	#plt.plot(r,vr_interp(r),"ro")
	#plt.show()

	r_it_fix = np.array([])
	vr_it = 50
	for n_iter in range(10):
		ring_position = np.arange(rstart,nrings,1+n_iter)
		for i in ring_position:
			#XY_mesh, vel_val, e_vel, f_pixel = pixels(vel,evel,guess0,ringpos = i, delta=2+n_iter,ring = "pixel",pixel_scale=pixel_scale)
			XY_mesh, vel_val, e_vel, f_pixel = pixels(vel,evel,guess0,ringpos = i, delta=1+n_iter/2.,ring = "ARCSEC",pixel_scale=pixel_scale)
			if f_pixel > frac_pix:
					guess = [vr_it,vr20,pa_it_next,inc_it_next,X0,Y0,vsys_final]
					vary = [True,True,True,True,False,False,False] 
					Vrot , Vr2, Vsys,  pa, inc ,xi = fit("vsys",vel_val,e_vel,XY_mesh,guess,vary,mode,sigma)
					if Vrot>10 and Vrot<vmaxrot:
						vr_it = Vrot
						pa_it_2  = np.append(pa_it_2,pa)
						inc_it_2  = np.append(inc_it_2,inc)
						r_it_fix = np.append(r_it_fix,i)


	#plt.plot(r_it_fix,pa_it_2,"ko")
	#plt.plot(r_it_fix,inc_it_2,"ko")
	#plt.show()


	#inc:

	sigma_c_inc = sigma_clip(inc_it_2,sigma=2, maxiters=1)
	mask_inc = sigma_c_inc.mask 
	inc_it_mean = np.nanmedian(sigma_c_inc[~mask_inc])


	#PA:

	sigma_c_pa = sigma_clip(pa_it_2,sigma=2, maxiters=1)
	mask_pa = sigma_c_pa.mask 
	pa_it_mean = np.nanmedian(sigma_c_pa[~mask_pa])
	pa_it_2[mask_pa] = pa_it_mean


	# Error in pa & inc

	e_pa = np.nanstd(sigma_c_pa)
	e_inc = np.nanstd(sigma_c_pa)
	#sigma = [0,0,2*e_pa,2*e_inc,1,1,1]

	#plt.plot(r_it_fix,inc_it_2,"ro")
	#plt.show()



	#fix inclination:

	PA_final_vals = np.asarray([])
	r_PA = np.asarray([])
	vr_it = 50
	vr2_it = 10
	guess_it = [vr_it,vr2_it,pa_it_mean,inc_it_mean,X0,Y0,vsys_final]
	for n_iter in range(10):
		ring_position = np.arange(rstart,nrings,1+n_iter)
		for i in ring_position:
			#XY_mesh, vel_val, e_vel, f_pixel = pixels(vel,evel,guess0,ringpos = i, delta=2+n_iter,ring = "pixel",pixel_scale=pixel_scale)
			XY_mesh, vel_val, e_vel, f_pixel = pixels(vel,evel,guess_it,ringpos = i, delta=2+n_iter/2.,ring = "ARCSEC",pixel_scale=pixel_scale)
			if f_pixel > frac_pix and i >0 :
					guess = [vr_it,vr2_it,pa_it_mean,inc_it_mean,X0,Y0,vsys_final]
					vary = [True,True,True,False,False,False,False] 
					Vrot , Vr2, Vsys,  pa, inc ,xi = fit("vsys",vel_val,e_vel,XY_mesh,guess,vary,mode,sigma)
					if Vrot>10 and Vrot<vmaxrot:
						vr_it = Vrot
						vr2_it = Vr2
						PA_final_vals  = np.append(PA_final_vals,pa)
						r_PA = np.append(r_PA,i)



	median_pa = np.nanmedian(PA_final_vals)
	sigma_pa = np.nanstd(PA_final_vals)


	#fix PA:

	inc_final_vals = np.asarray([])
	r_inc = np.asarray([])
	vr_it = 50
	for n_iter in range(10):
		ring_position = np.arange(rstart,nrings,1+n_iter)
		for i in ring_position:
			#XY_mesh, vel_val, e_vel, f_pixel = pixels(vel,evel,guess0,ringpos = i, delta=2+n_iter,ring = "pixel",pixel_scale=pixel_scale)
			XY_mesh, vel_val, e_vel, f_pixel = pixels(vel,evel,guess0,ringpos = i, delta=2+n_iter/2.,ring = "ARCSEC",pixel_scale=pixel_scale)
			if f_pixel > frac_pix:
					guess = [vr_it,vr20,median_pa,inc_it_mean,X0,Y0,vsys_final]
					vary = [True,True,False,True,False,False,False]
					Vrot , Vr2, Vsys,  pa, inc ,xi = fit("vsys",vel_val,e_vel,XY_mesh,guess,vary,mode,sigma)
					if Vrot>10 and Vrot<vmaxrot:
						vr_it = Vrot
						inc_final_vals  = np.append(inc_final_vals,inc)
						r_inc = np.append(r_inc,i)



	median_inc = np.nanmedian(inc_final_vals)
	sigma_inc = np.nanstd(inc_final_vals)



	sigma_f = [500,500,1*sigma_pa,1*sigma_inc,1,1,1]

	



	inc_final = []#np.asarray([])
	pa_final = []#np.asarray([])
	r_final = []#np.asarray([])
	vr_inc_pa = []#np.asarray([])

	vr_it_0 = 50#vr_interp(i)
	vr2_it_0 = 0
	guess0 = [vrot0,vr20,pa0,inc0,X0,Y0,vsys0]
	for n_iter in range(5):
		ring_position = np.arange(rstart,nrings,1+n_iter)
		for i in ring_position:
			#XY_mesh, vel_val, e_vel, f_pixel = pixels(vel,evel,guess0,ringpos = i, delta=2+n_iter,ring = "pixel",pixel_scale=pixel_scale)
			XY_mesh, vel_val, e_vel, f_pixel = pixels(vel,evel,guess0,ringpos = i, delta=2+n_iter/2.,ring = "ARCSEC",pixel_scale=pixel_scale)
			if f_pixel > frac_pix:
					guess = [vr_it_0,vr20,median_pa,inc_it_mean,X0,Y0,vsys_final]
					vary = [True,True,True,True,False,False,False] 
					Vrot , Vr2, Vsys,  pa, inc ,xi = fit("vsys",vel_val,e_vel,XY_mesh,guess,vary,mode,sigma_f)
					#guess0 = guess

					if Vrot>10 and Vrot<vmaxrot:
						vr_it_0 = Vrot
						vr2_it_0 = Vr2

						r_final.append(i)
						pa_final.append(pa)
						inc_final.append(inc)
						vr_inc_pa.append(Vrot)


	r_sort, pa_sort = zip(*sorted(zip(r_final, pa_final)))
	r_sort, inc_sort = zip(*sorted(zip(r_final, inc_final)))
	r_sort, vr_sort = zip(*sorted(zip(r_final, vr_inc_pa))) 
	max_r = int(np.nanmax(r_sort))
	

	inc_sort = np.asarray(inc_sort)
	pa_sort = np.asarray(pa_sort)
	r_final = np.asarray(r_sort)
	vr_sort = np.asarray(vr_sort)
	r_sort = np.asarray(r_sort)


	pa_mean_ring = np.asarray([])
	inc_mean_ring = np.asarray([])
	vr_mean_ring = np.asarray([])
	r_mean_ring = np.asarray([])
	for kk in range(max_r+1):
		mask = r_sort == kk
		pa_i = pa_sort[mask]
		inc_i = inc_sort[mask]
		vr_i = vr_sort[mask]

		if len(pa_i)>0:
			vr_mean_ring = np.append(vr_mean_ring,np.nanmean(vr_i))
			pa_mean_ring = np.append(pa_mean_ring,np.nanmean(pa_i))
			inc_mean_ring = np.append(inc_mean_ring,np.nanmean(inc_i))
			r_mean_ring = np.append(r_mean_ring,kk)

		else:
			vr_mean_ring = np.append(vr_mean_ring,20)
			pa_mean_ring = np.append(pa_mean_ring,median_pa)
			inc_mean_ring = np.append(inc_mean_ring,median_inc)
			r_mean_ring = np.append(r_mean_ring,kk)


	legfit = np.polynomial.legendre.legfit(r_mean_ring,inc_mean_ring, 6)
	legval_inc = np.polynomial.legendre.legval(r_mean_ring,legfit)

	legfit = np.polynomial.legendre.legfit(r_mean_ring,pa_mean_ring, 6)
	legval_pa = np.polynomial.legendre.legval(r_mean_ring,legfit)


	legfit = np.polynomial.legendre.legfit(r_mean_ring,vr_mean_ring, 6)
	legval_vr = np.polynomial.legendre.legval(r_mean_ring,legfit)

	
	"""
	plt.plot(r_mean_ring,pa_mean_ring,"bo")
	plt.plot(r_mean_ring,legval_pa,"k-")
	plt.plot(r_mean_ring,inc_mean_ring,"bo")
	plt.plot(r_mean_ring,legval_inc,"k-")
	plt.show()

	plt.plot(r_mean_ring,vr_mean_ring,"bo")
	plt.plot(r_mean_ring,legval_vr,"k-")
	plt.show()
	"""


	f_pa = interp1d(r_mean_ring,legval_pa)
	f_inc = interp1d(r_mean_ring,legval_inc)
	f_vr = interp1d(r_mean_ring,legval_vr)


	vc_final = np.asarray([])
	vrad_final = np.asarray([])
	ring_position = r_mean_ring
	R = np.asarray([])
	k = 0
	guess_final = [50,20,median_pa,median_inc,X0,Y0,vsys_final]

	vr_temp = 50#f_vr(1)
	vrad_temp = 0
	sigma = []

	nrings = 50
	ring_position = np.arange(0,nrings,1)

	frac_pix = 0.25
	xi_0 = 0
	for i in ring_position:
		#XY_mesh, vel_val, e_vel, f_pixel = pixels(vel,evel,guess_final,ringpos = i, delta= 4,ring = "pixel",pixel_scale=pixel_scale)
		XY_mesh, vel_val, e_vel, f_pixel = pixels(vel,evel,guess_final,ringpos = i, delta= 1 ,ring = "ARCSEC",pixel_scale=pixel_scale)

		if f_pixel > frac_pix and i > 0:


					#guess = [f_vr(i),vrad_temp,median_pa,median_inc,X0,Y0,vsys_final]
					guess = [vr_temp,vrad_temp,median_pa,median_inc,X0,Y0,vsys_final]

					vary = [True,True,False,False,False,False,False] 
					Vrot , Vr2, Vsys,  pa, inc ,xi = fit("vsys",vel_val,e_vel,XY_mesh,guess,vary,mode,sigma)

					if Vrot>10 and Vrot<vmaxrot:

						if i<=10:
						#if 1>0:

							vr_temp = Vrot
							vrad_temp = Vr2
							xi_0 = xi
							vc_final = np.append(vc_final,Vrot)
							vrad_final = np.append(vrad_final,Vr2)
							R = np.append(R,i)
							for mm,nn in zip(XY_mesh[0],XY_mesh[1]): 
								MODEL[nn][mm] = Vlos(mm,nn,Vrot,Vr2,pa,inc,X0,Y0,Vsys) - Vsys
						
						#"""	
						else:
							if abs(xi_0-xi)<5 or xi < xi_0:
								xi_0 = xi
								vc_final = np.append(vc_final,Vrot)
								vrad_final = np.append(vrad_final,Vr2)
								R = np.append(R,i)
								for mm,nn in zip(XY_mesh[0],XY_mesh[1]): 
									MODEL[nn][mm] = Vlos(mm,nn,Vrot,Vr2,pa,inc,X0,Y0,Vsys) - Vsys
						"""

							else:
								sigma_f = [500,500,2*sigma_pa,2*sigma_inc,1,1,1]
								vary = [True,True,True,True,False,False,False] 
								XY_mesh, vel_val, e_vel, f_pixel = pixels(vel,evel,guess_final,ringpos = i, delta= 2 ,ring = "ARCSEC",pixel_scale=pixel_scale)
								guess = [f_vr(i),vr_temp,vrad_temp,f_pa(i),f_inc(i),X0,Y0,vsys_final]
								Vrot , Vr2, Vsys,  pa, inc ,xi = fit("vsys",vel_val,e_vel,XY_mesh,guess,vary,mode,sigma_f)
								print("xi2=",xi)
						"""



	#plt.plot(r_gal_4,vr_it_4,"ko")
	#plt.plot(r_gal_4,vr_it_4, color = "k",linestyle='-', alpha = 0.6)
	#plt.plot(r_gal_4,legval_Vr, color = "r",linestyle='-', alpha = 0.6)
	#plt.plot(r_gal_4,vr2_it_4,"bo")
	#plt.plot(r_gal_4,vr2_it_4, color = "b",linestyle='-', alpha = 0.6)
	#plt.show()

	
	MODEL[MODEL == 0] = np.nan

	return median_pa,median_inc,median_vsys,sigma_pa,sigma_inc,sigma_vsys,R,vc_final,vrad_final,MODEL

import random

def RC_sigma_rnd2(vel,evel,nrings,guess0,vary,sigma = [],mode = "rotation", delta=1,ring = "pixel",rstart = 4,iter = 5, ring_position = [],pixel_scale = 1 ):

	vrot0,vr20,pa0,inc0,X0,Y0,vsys0 = guess0
	guess = [vrot0,vr20,pa0,inc0,X0,Y0,vsys0]
	guess_copy = np.copy(guess)

	n_sigma = len(sigma)

	if n_sigma != 0:
		e_vrot,e_vr2,e_pa,e_inc,e_x0,e_y0,e_vsys = sigma


	[ny,nx] = vel.shape

	x = np.arange(0, nx, 1)
	y = np.arange(0, ny, 1)
	mesh = np.meshgrid(x,y,sparse=True)



	from pixel_params import pixels
	import fit_params
	from fit_params import fit
	from fit_params import fit_polynomial
	from fit_params import fit_linear


	vary = [True,True,True,True,False,False,True]
	#ring_position = np.arange(rstart,nrings,2) 
	#ring_rposition = np.arange(rstart,nrings,4)

	vr1,vr21,vsys1,pa1,inc1 = np.asarray([]),np.asarray([]),np.asarray([]),np.asarray([]),np.asarray([])
	vr0,vr20,vsys0,pa0,inc0,r0 = [],[],[],[],[],[]
	r = np.asarray([])

	for n_iter in range(iter):
	#for i in ring_position:
		#vr0,vr20,vsys0,pa0,inc0 =  np.asarray([]),np.asarray([]),np.asarray([]),np.asarray([]),np.asarray([])
		#for n_iter in range(iter):
		for i in ring_position:
			sigma = []
			sol = np.asarray(guess)
			sol[0] =sol[0]+10*random.uniform(-1,1)
			sol[1] =sol[1]+10*random.uniform(-1,1)
			sol[2] =sol[2]+e_pa*random.uniform(-1,1)
			sol[3] =sol[3]+e_inc*random.uniform(-1,1)
			sol[4] =sol[4]#+(2/pixel_scale)*random.uniform(-1,1)
			sol[5] =sol[5]#+(2/pixel_scale)*random.uniform(-1,1)
			sol[6] =sol[6]+e_vsys*random.uniform(-1,1)

			try:
				#XY_mesh, vel_val, e_vel, f_pixel = pixels(vel,evel,guess,ringpos = i, delta=4,ring = "pixel",pixel_scale=pixel_scale)
				XY_mesh, vel_val, e_vel, f_pixel = pixels(vel,evel,guess,ringpos = i, delta= 2 ,ring = "ARCSEC",pixel_scale=pixel_scale)
				if f_pixel > 0.10:

					Vrot , Vr2, Vsys,  pa, inc ,xi = fit("vsys",vel_val,e_vel,XY_mesh,sol,vary,mode,sigma)
					if Vrot>10 and Vrot<vmaxrot:

						vr0.append(Vrot)
						vr20.append(Vr2)
						vsys0.append(Vsys)
						pa0.append(pa)
						inc0.append(inc)
						r0.append(i)
					else:

						vr0.append(np.nan)
						vr20.append(np.nan)
						vsys0.append(np.nan)
						pa0.append(np.nan)
						inc0.append(np.nan)
						r0.append(i)


			#except(TypeError,ZeroDivisionError,ValueError):
			except(1):
				pass

	r_sort, pa_sort = zip(*sorted(zip(r0, pa0)))
	r_sort, inc_sort = zip(*sorted(zip(r0, inc0)))
	r_sort, vr_sort = zip(*sorted(zip(r0, vr0))) 
	r_sort, vsys_sort = zip(*sorted(zip(r0, vsys0))) 
	r_sort, vr20_sort = zip(*sorted(zip(r0, vr20))) 
	max_r = np.nanmax(r_sort)
	max_r = int(max_r)

	inc_sort = np.asarray(inc_sort)
	pa_sort = np.asarray(pa_sort)
	r_final = np.asarray(r_sort)
	vr_sort = np.asarray(vr_sort)
	vsys_sort = np.asarray(vsys_sort)
	vr20_sort = np.asarray(vr20_sort)
	r_sort = np.asarray(r_sort)

	R = np.unique(r_sort)


	pa_mean_ring = np.asarray([])
	inc_mean_ring = np.asarray([])
	vr_mean_ring = np.asarray([])
	vsys_mean_ring = np.asarray([])
	vr20_mean_ring = np.asarray([])
	r_mean_ring = np.asarray([])




	pa_e = np.asarray([])
	inc_e = np.asarray([])
	vr_e = np.asarray([])
	vsys_e = np.asarray([])
	vr20_e = np.asarray([])


	#print(len(r_sort),len(ring_position))
	#for kk in range(max_r+1):
	for kk in R:
		mask = r_sort == kk
		pa_i = pa_sort[mask]
		inc_i = inc_sort[mask]
		vr_i = vr_sort[mask]
		vsys_i = vsys_sort[mask]
		vr20_i = vr20_sort[mask]

		if len(pa_i)>0:
			vr_mean_ring = np.append(vr_mean_ring,np.nanmean(vr_i))
			vsys_mean_ring = np.append(vsys_mean_ring,np.nanmean(vsys_i))
			vr20_mean_ring = np.append(vr20_mean_ring,np.nanmean(vr20_i))
			pa_mean_ring = np.append(pa_mean_ring,np.nanmean(pa_i))
			inc_mean_ring = np.append(inc_mean_ring,np.nanmean(inc_i))
			r_mean_ring = np.append(r_mean_ring,kk)

			vr_e = np.append(vr_e,np.nanstd(vr_i))
			vr20_e = np.append(vr20_e,np.nanstd(vr20_i))
			pa_e = np.append(pa_e,np.nanstd(pa_i))
			inc_e = np.append(inc_e,np.nanstd(inc_i))

		#else:
		#	vr_mean_ring = np.append(vr_mean_ring,20)
		#	pa_mean_ring = np.append(pa_mean_ring,median_pa)
		#	inc_mean_ring = np.append(inc_mean_ring,median_inc)
		#	r_mean_ring = np.append(r_mean_ring,kk)





	return vr_e,vr20_e,pa_e,inc_e,vsys_e
	#return vr0,vr20,vsys0,pa0,inc0,r




def RC_sigma_rnd(vel,evel,nrings,guess0,vary,sigma = [],mode = "rotation", delta=1,ring = "pixel",rstart = 4,iter = 5, ring_position = [] ,pixel_scale = 1):

	rstart = int(rstart/pixel_scale)
	vrot0,vr20,pa0,inc0,X0,Y0,vsys0 = guess0
	guess = [vrot0,vr20,pa0,inc0,X0,Y0,vsys0]
	guess_copy = np.copy(guess)

	n_sigma = len(sigma)

	if n_sigma != 0:
		e_vrot,e_vr2,e_pa,e_inc,e_x0,e_y0,e_vsys = sigma


	[ny,nx] = vel.shape

	x = np.arange(0, nx, 1)
	y = np.arange(0, ny, 1)
	mesh = np.meshgrid(x,y,sparse=True)



	from pixel_params import pixels
	import fit_params
	from fit_params import fit
	from fit_params import fit_polynomial
	from fit_params import fit_linear


	vary = [True,True,True,True,False,False,True]
	#ring_position = np.arange(rstart,nrings,2) 
	#ring_position = np.arange(rstart,nrings,4)

	vr1,vr21,vsys1,pa1,inc1 = np.asarray([]),np.asarray([]),np.asarray([]),np.asarray([]),np.asarray([])
	vr1,vr21,vsys1,pa1,inc1,r1 = [],[],[],[],[],[]
	r = np.asarray([])

	for n_iter in range(iter):
	#for i in ring_position:
		vr0,vr20,vsys0,pa0,inc0 =  np.asarray([]),np.asarray([]),np.asarray([]),np.asarray([]),np.asarray([])
		#for n_iter in range(iter):
		for i in ring_position:
			sigma = []
			sol = np.asarray(guess)
			sol[0] =sol[0]+10*random.uniform(-1,1)
			sol[1] =sol[1]+10*random.uniform(-1,1)
			sol[2] =sol[2]+e_pa*random.uniform(-1,1)
			sol[3] =sol[3]+e_inc*random.uniform(-1,1)
			sol[4] =sol[4]#+(2/pixel_scale)*random.uniform(-1,1)
			sol[5] =sol[5]#+(2/pixel_scale)*random.uniform(-1,1)
			sol[6] =sol[6]+e_vsys*random.uniform(-1,1)

			try:
				XY_mesh, vel_val, e_vel, f_pixel = pixels(vel,evel,guess,ringpos = i, delta=4,ring = "pixel",pixel_scale=pixel_scale)
				if f_pixel > 0.10:

					Vrot , Vr2, Vsys,  pa, inc ,xi = fit("vsys",vel_val,e_vel,XY_mesh,sol,vary,mode,sigma)
					if Vrot>10 and Vrot<vmaxrot:

						vr0 = np.append(vr0,Vrot)
						vr20 = np.append(vr20,Vr2)
						vsys0 = np.append(vsys0,Vsys)
						pa0 = np.append(pa0,pa)
						inc0 = np.append(inc0,inc)
						if n_iter == 0:
							r1.append(i)

					else:

						vr0 = np.append(vr0,Vrot)
						vr20 = np.append(vr20,Vr2)
						vsys0 = np.append(vsys0,Vsys)
						pa0 = np.append(pa0,pa)
						inc0 = np.append(inc0,inc)
						if n_iter == 0:
							r1.append(i)
				#else: print(i,"aqtuiiii")

			except(TypeError,ZeroDivisionError,ValueError):
				pass

		#if len(vr0) > 0 and  len(vr20)>0 and len(vsys0) >0 and len(pa0) >0 and len(inc0) >0:
		#if 1>0:


		#vr1=np.append(vr1,vr0)
		#vr21 = np.append(vr21,vr20)
		#vsys1 = np.append(vsys1,vsys0)
		#inc1 = np.append(inc1,inc0)
		#pa1 = np.append(pa1,pa0)

			#if n_iter == 0:
			#	r1.append(i)

		if len(vr0) !=0:
			vr1.append(vr0)
			vr21.append(vr20)
			vsys1.append(vsys0)
			inc1.append(inc0)
			pa1.append(pa0)

	return vr1,vr21,vsys1,pa1,inc1,r1
	#return vr0,vr20,vsys0,pa0,inc0,r






def RC_emcee(vel,evel,nrings,guess0,vary,sigma = [],mode = "rotation", delta=1,ring = "pixel",rstart = 4,iter = 5, pos = 2,pixel_scale = 1):

	rstart = int(rstart/pixel_scale)
	vrot0,vr20,pa0,inc0,X0,Y0,vsys0 = guess0
	guess = [vrot0,vr20,pa0,inc0,X0,Y0,vsys0]
	guess_copy = np.copy(guess)

	n_sigma = len(sigma)

	if n_sigma != 0:
		e_vrot,e_vr2,e_pa,e_inc,e_x0,e_y0,e_vsys = sigma


	[ny,nx] = vel.shape

	x = np.arange(0, nx, 1)
	y = np.arange(0, ny, 1)
	mesh = np.meshgrid(x,y,sparse=True)



	from pixel_params import pixels
	from emcee_fit import EMCEE
	import fit_params
	from fit_params import fit
	from fit_params import fit_polynomial
	from fit_params import fit_linear


	vary = [True,True,True,True,False,False,True]
	ring_position = np.arange(rstart,nrings,2) 



	vr1,vr21,vsys1,pa1,inc1 = [],[],[],[],[]
	r = []
	for i in ring_position:
		vr0,vr20,vsys0,pa0,inc0 = [],[],[],[],[]



		for n_iter in range(iter):
			#sigma = []
			sol = np.asarray(guess)
			sol[0] =sol[0]+10*random.uniform(-1,1)
			sol[1] =sol[1]+10*random.uniform(-1,1)
			sol[2] =sol[2]+e_pa*random.uniform(-1,1)
			sol[3] =sol[3]+e_inc*random.uniform(-1,1)
			sol[4] =sol[4]+(2/pixel_scale)*random.uniform(-1,1)
			sol[5] =sol[5]+(2/pixel_scale)*random.uniform(-1,1)
			sol[6] =sol[6]+e_vsys*random.uniform(-1,1)

			try:
				XY_mesh, vel_val, e_vel, f_pixel = pixels(vel,evel,sol,ringpos = i, delta=4,ring = "pixel",pixel_scale=pixel_scale)
				if f_pixel > 0.50:
					res = EMCEE(XY_mesh, vel_val, e_vel,guess,sigma)
			except(TypeError,ZeroDivisionError,ValueError):pass
			#except(1):pass

	#return vr1,vr21,vsys1,pa1,inc1,r
	return 0






