import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits

import cmap_califa
import cmap_vfield
califa=cmap_vfield.CALIFA()
import lmfit
import operator
from lmfit import Model
from lmfit import Parameters, fit_report, minimize
from matplotlib.gridspec import GridSpec



from axis import AXIS
#from bisymetric_model_objective import RC_sigma 
#from bisymetric_model_objective_v2 import RC_sigma 
from bisymetric_model import RC_sigma 
from bisymetric_model import RC_sigma_rnd2 
from bisymetric_model import RC_emcee
#from bisymetric_model_objective_v3 import RC_sigma_iter
from scipy.optimize import curve_fit
from kinematic_centre_vsys import KC
from functools import reduce
from CBAR import colorbar as cb
path = "/nfs/ofelia/disk-b/manga/data/v2_7_1/Pipe3D"

#vel = "stars"
vel = "gas"



c = 3e5
#in CU


def SN(flux,eflux,sn):
	A = np.divide(flux,eflux)
	A[A<sn] = np.nan
	return np.divide(A,A)


def rotcur(galaxy,z_star,PA,INC,X0,Y0):

	p1,p2 = [pos for pos, char in enumerate(galaxy) if char == "-"]
	direc = galaxy[p1+1:p2]

	hdu0 = fits.open("%s/%s/%s.Pipe3D.cube.fits.gz"%(path,direc,galaxy))
	#data_extensions = hdu0[2].data
	data = hdu0[3].data
	hdr = hdu0[0].header
	pixel_size = abs(hdr["CD2_2"]*3600)

	[nz,ny,nx]=data.shape
	ext = [nx/2., -nx/2,-ny/2.,ny/2.]

	ha_flux = data[45]
	e_ha = data[249]

	mask_SN = SN(ha_flux,e_ha,5)
	M_ones=np.ones((ny,nx))*mask_SN
	mask_good=np.ones((ny,nx))*mask_SN

	for i in range(nx):
		for j in range(ny):
			check_ones=M_ones[j-1:j+2,i-1:i+2]
			not_ones=np.nansum(check_ones)
			if not_ones>=7:
				mask_good[j][i]=1
			else:
				mask_good[j][i]=np.nan

	if vel == "gas":
		vel_ha = data[102]
		e_vel_ha= data[330]

		#hdu = fits.open("%s/%s.ELINES.cube.fits.gz"%(path,galaxy))
		#data = hdu[0].data
		#vel_ha =data[0] 
		#e_vel_ha =data[0] 

	if vel == "stars":
		e_vel_ha= data[300]
		hdu = fits.open("%s/%s.SSP.cube.fits.gz"%(path,galaxy))
		ssp = hdu[0].data
		#ssp = pyfits.getdata("./CALIFA/dataproducts/%s.SSP.cube.fits.gz"%galaxy)
		vel_ha = ssp[13]
		#e_vel_ha= ssp[14]


	e_vel_ha[e_vel_ha>10]=np.nan
	mask_vel=np.divide(e_vel_ha,e_vel_ha)


	vel_ha = vel_ha*mask_vel
	vel_ha[vel_ha==0]=np.nan

	XK,YK,VSYS,e_vsys = KC(vel_ha,X0,Y0)
	VSYS = z_star*c


	"""
	print ("...................................................................................................")
	print ("..............................%s............................................................"%galaxy)
	print ("........................Examine initial values: ...................................................")
	print ("Guess : PA,INC,X0,Y0:", PA,INC,XK,YK,VSYS)
	print ("...................................................................................................")

	print ("...................................................................................................")
	print ("................... step 1: All free parameters: estiamate X0,Y0, Vsys ..........................." )
	print ("...................................................................................................")
	"""

	nrings = int(36)
	guess = [50,10,PA,INC,XK,YK,VSYS]
	vary = [True, True, True, True,False, False, True]
	mode_ring = ["pixel", "arcsec", "broad"]
	mode_ring = ["pixel"]



	def first():
		pa,r,x0,y0,vsys = [],[],[],[],[]
		pa2,inc2,x02,y02,vsys2 = [],[],[],[],[]

		delta_arcsec = int(2/1)
		median_pa,median_inc,median_vsys,sigma_pa,sigma_inc,sigma_vsys,R,vrot,vrad,model = RC_sigma(vel_ha,e_vel_ha,nrings,guess,vary,delta=delta_arcsec,ring = "pixel",mode = "bisymetric", iter = 1,model = True,pixel_scale = pixel_size)
		
		return median_pa,median_inc,median_vsys,sigma_pa,sigma_inc,sigma_vsys,R,vrot,vrad,model


	median_pa,median_inc,median_vsys,sigma_pa,sigma_inc,sigma_vsys,R,vrot,vr2,model = first()



	def sec():
		vary = [ True, True, True, True,False, False, True]
		guess = [60,0,median_pa,median_inc,XK,YK,median_vsys]
		sigma = [0,0,2*sigma_pa,2*sigma_inc,0,0,2*sigma_vsys]
		delta_arcsec = int(2/1)

		vr_e,vr20_e,pa_e,inc_e,vsys_e = RC_sigma_rnd2(vel_ha,e_vel_ha,nrings,guess,vary,sigma,mode = "bisymetric", delta=delta_arcsec,ring = "pixel",iter = 10, ring_position = R,pixel_scale = pixel_size)
		return vr_e,vr20_e,pa_e,inc_e,vsys_e




	def emcee():
		vary = [ True, True, True, True,False, False, True]
		guess = [60,0,pa_med,inc_med,XK,YK,vsys_med]
		sigma = [0,0,5*e_pa,5*e_inc,0,0,5*e_vsys]
		delta_arcsec = int(2/1)

		print("sigma:", sigma)

		a= RC_emcee(vel_ha,e_vel_ha,nrings,guess,vary,sigma,mode = "bisymetric", delta=delta_arcsec,ring = "pixel",iter = 5,pixel_scale = pixel_size)
		return a


	#c = emcee()

	e_vr,e_vr2,e_pa,e_inc,e_vsys = sec()


	mask_model = np.divide(model,model)
	fig=plt.figure(figsize=(6,2))
	gs2 = GridSpec(1, 3)
	gs2.update(left=0.06, right=0.62,top=0.83,hspace=0.01,bottom=0.15,wspace=0.)


	ax=plt.subplot(gs2[0,0])
	ax1=plt.subplot(gs2[0,1])
	ax2=plt.subplot(gs2[0,2])


	im0 = ax.imshow(vel_ha - median_vsys,cmap = califa, origin = "l",vmin = -250,vmax = 250, aspect = "auto", extent = ext, interpolation = "nearest")
	im2 = ax1.imshow(model,cmap = califa, origin = "l", aspect = "auto", vmin = -250,vmax = 250, extent = ext, interpolation = "nearest")

	residual = (vel_ha*mask_model - median_vsys)- model
	im2 = ax2.imshow(residual,cmap = califa, origin = "l", aspect = "auto",vmin = -50,vmax = 50, extent = ext, interpolation = "nearest")


	AXIS(ax,tickscolor = "k")
	AXIS(ax1,tickscolor = "k",remove_yticks= True)
	AXIS(ax2,tickscolor = "k",remove_yticks= True)



	ax.set_ylabel(r'$\Delta$ Dec (pix)',fontsize=8,labelpad=0)
	ax.set_xlabel(r'$\Delta$ RA (pix)',fontsize=8,labelpad=0)
	ax1.set_xlabel(r'$\Delta$ RA (pix)',fontsize=8,labelpad=0)
	ax2.set_xlabel(r'$\Delta$ RA (pix)',fontsize=8,labelpad=0)

	ax.text(0.05,0.9, "VLOS", fontsize = 7, transform = ax.transAxes)
	ax1.text(0.05,0.9,"MODEL", fontsize = 7, transform = ax1.transAxes)
	ax2.text(0.05,0.9,"RESIDUAL",fontsize = 7, transform = ax2.transAxes)


	ax.set_facecolor('#e8ebf2')
	ax1.set_facecolor('#e8ebf2')
	ax2.set_facecolor('#e8ebf2')


	gs2 = GridSpec(1, 1)
	gs2.update(left=0.68, right=0.995,top=0.83,bottom=0.15)
	ax3=plt.subplot(gs2[0,0])

	ax3.errorbar(R,vrot, yerr=e_vr, fmt='s', color = "k",markersize = 3)
	ax3.plot(R,vrot, color = "k",linestyle='-', alpha = 0.6)

	ax3.plot(R,vr2, color = "skyblue",linestyle='-', alpha = 0.6)
	ax3.errorbar(R,vr2, yerr=e_vr2, fmt='s', color = "skyblue",markersize = 3)

	#ax3.set_ylim(-50,300)
	ax3.set_ylim(int(np.nanmin(vr2))-20,int(np.nanmax(vrot))+20)
	ax3.plot([0,np.nanmax(R)],[0,0],color = "k",linestyle='-', alpha = 0.6)
	ax3.set_xlabel('r (arcsec)',fontsize=8,labelpad = 0)
	ax3.set_ylabel('V$_\mathrm{ROT}$ (km/s)',fontsize=8,labelpad = 0)
	ax3.set_facecolor('#e8ebf2')

	AXIS(ax3,tickscolor = "k")


	cb(im0,ax,orientation = "horizontal", colormap = califa, bbox= (0,1.12,1,1),width = "100%", height = "5%",label_pad = -23, label = "(km/s)",font_size=7)
	cb(im2,ax2,orientation = "horizontal", colormap = califa, bbox= (0,1.12,1,1),width = "100%", height = "5%",label_pad = -23, label = "(km/s)",font_size=7)
	plt.savefig("./plots/kin_model_%s.png"%galaxy,dpi = 300)
	#plt.savefig("./kin_model_%s.pdf"%galaxy)
	plt.clf()
	#plt.show()

	try:
		from fit_rotcurve import Rturn_Vmax
		vmax, r_turn,beta,gamma =  Rturn_Vmax(R,vrot)

		def vrot_fit(r_sky,v_max,R_turn,beta,gamma):

				x=R_turn/r_sky
				A = (1+x)**beta
				B = (1+x**gamma)**(1./gamma)
				v=v_max*A/B

				return v

		fig=plt.figure(figsize=(2.5,2.5))
		ax = plt.subplot(111)
		ax.scatter(R,vrot,s = 10, marker = "o", c = "k")
		ax.plot(R,vrot_fit(R,vmax,r_turn,beta,gamma),"b-",alpha = 0.3)
		AXIS(ax,tickscolor = "k")
		ax.set_xlabel('r (arcsec)',fontsize=8,labelpad = 0)
		ax.set_ylabel('V$_\mathrm{ROT}$ (km/s)',fontsize=8,labelpad = 0)
		ax.set_ylim(-5,int(np.nanmax(vrot))+20)
		ax.set_facecolor('#e8ebf2')
		plt.savefig("./vflat/vflat_r_turn_%s.png"%galaxy,dpi = 300)
		#plt.savefig("./vflat_rotcur_%s.pdf"%galaxy)
		plt.clf()
		#plt.show()
		return galaxy,XK,YK,median_pa,sigma_pa,median_inc,sigma_inc,median_vsys,sigma_vsys,vmax, r_turn,beta,gamma
	except(1):
		return galaxy,XK,YK,median_pa,sigma_pa,median_inc,sigma_inc,median_vsys,sigma_vsys,0,0,0,0
