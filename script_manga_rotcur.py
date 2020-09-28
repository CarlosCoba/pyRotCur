import numpy as np
from manga_bisymetric import rotcur


ruta = "../get_proc_elines_MaNGA.clean.csv"
manga_id = np.genfromtxt(ruta,usecols =  0, dtype = str, delimiter = ",")

z_Star = np.genfromtxt(ruta,usecols =  26, dtype = float, delimiter = ",",filling_values="0")
Center = np.genfromtxt(ruta,usecols =  [179,180], dtype = float, delimiter = ",")
P_A = np.genfromtxt(ruta,usecols =  142, dtype = float, delimiter = ",")-90
INCL =  np.genfromtxt(ruta,usecols =  6, dtype = float, delimiter = ",")


n = len(manga_id)
print(n)
for i in range(6457,n):
#for i in range(1):

	try:
		galaxy = manga_id[i]
		#galaxy = manga_id[130]
		#galaxy = manga_id[160]
		index = np.where(manga_id == galaxy)[0][0]
		z_star = z_Star[index]
		X0,Y0 = Center[index]
		PA = P_A[index]
		INC = INCL[index]

		galaxy,XK,YK,median_pa,sigma_pa,median_inc,sigma_inc,median_vsys,sigma_vsys,vmax, r_turn,beta,gamma = rotcur(galaxy,z_star,PA,INC,X0,Y0)
		print(i,galaxy,XK,YK,median_pa,sigma_pa,median_inc,sigma_inc,median_vsys,sigma_vsys,vmax, r_turn,beta,gamma)


	except(ValueError,AttributeError,TypeError,ZeroDivisionError,OSError):
	#except(1):


		try:
			z_star = z_Star[index]
			X0,Y0 = Center[index]
			PA = -P_A[index]
			INC = INCL[index]

			galaxy,XK,YK,median_pa,sigma_pa,median_inc,sigma_inc,median_vsys,sigma_vsys,vmax, r_turn,beta,gamma = rotcur(galaxy,z_star,PA,INC,X0,Y0)
			print(i,galaxy,XK,YK,median_pa,sigma_pa,median_inc,sigma_inc,median_vsys,sigma_vsys,vmax, r_turn,beta,gamma)


		except(AttributeError,ValueError,TypeError,OSError):
		#except(1):
			print(i,galaxy,0,0,0,0,0,0,0,0,0,0,0,0)

