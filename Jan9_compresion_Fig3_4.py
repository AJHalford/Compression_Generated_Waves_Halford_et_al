###############################################################################################
import numpy as np
import datetime as dt
from datetime import timedelta
#This will likely need to be updated to point where the
#cdf libaray is on your computer
import os
os.environ["CDF_LIB"] =  "/Applications/cdf/cdf37_1-dist/lib"
#os.putenv("CDF_LIB", "~/CDF/LIB")
from spacepy import pycdf
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.mlab as mlab
import plasmaconst as pc
import matplotlib.dates as mdates
import matplotlib as mpl

from matplotlib.pyplot import figure 
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
###############################################################################################
###############################################################################################
#Here we are defining the start and stop times and other event times
starttime = dt.datetime(2014, 1, 9, 19, 55)
plttimestart = dt.datetime(2014, 1, 9, 20, 5)
endtime = dt.datetime(2014, 1, 9, 20, 35)
plttimeend = dt.datetime(2014, 1, 9, 20, 20)

compression = dt.datetime(2014, 1, 9, 20, 10, 15)
wave = dt.datetime(2014, 1, 9, 20, 11, 30)

compression_end = dt.datetime(2014, 1, 9, 20, 18, 00)
wave_end = dt.datetime(2014, 1, 9, 20, 14, 45)
##############################################################################################

# these are colorblind friendly on their own and many colormaps work well. But be sure to test on 
#https://www.color-blindness.com/coblis-color-blindness-simulator/

#First we are defining the primary colors used for these maps
cdi = '#093145'
cli = '#3c6478'
cda = '#107896'
cla = '#43abc9'
cdk = '#829356'
clk = '#b5c689'
cdd = '#bca136'
cld = '#efd469'
cdc = '#c2571a'
clc = '#f58b4c'
cdr = '#9a2617'
clr = '#cd594a'
clg = '#F3F4F6'
cdg = '#8B8E95'

#Now we are putting the ones together that we want to go from one to another
greycolors = [clg, cdg]                                          
greencolors = [clg, clk, cdk]
yellowcolors = [clg,cld, cdd]
redcolors = [clg, clr, cdr]
hotcolors = [cld, cdd, cdc, cdr]
colors = [cdi,  cdk, cld,  cdc,  cdr]
bluecolors = [clg, cla, cda, cdi]
trycolors = [cdi, cli, cdd, cld, clg]#[cdi, cli, cla, cld]#[cla, cda, cdi, cdr, cdc, clc]

#finally we make the actual maps
bluermap = mpl.colors.LinearSegmentedColormap.from_list("", bluecolors)
pltmap = mpl.colors.LinearSegmentedColormap.from_list("", hotcolors)
greenmap = mpl.colors.LinearSegmentedColormap.from_list("", greencolors)
yellowmap = mpl.colors.LinearSegmentedColormap.from_list("", yellowcolors)
redmap = mpl.colors.LinearSegmentedColormap.from_list("", redcolors)
trymap = mpl.colors.LinearSegmentedColormap.from_list("", trycolors)

#We chose trymap for this set of figures
colormap = trymap
#Here we are setting the text size for the figures. 
txtsize = 12
mpl.rcParams['font.size'] = txtsize
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&





#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#&&&&&&&&&&&&&&&&&&&&&&&&&&&  Figure 3 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#Here we are making the plot of how the temperature anisotropy changes due to the compression. 

Ao = np.arange(200)/60. + 0.1
dL = np.arange(100)/100. *5.
print( len(Ao), len(dL))
A = np.zeros( [len(Ao), len(dL)])
temp = []
for i in range(len(Ao)):
	for j in range(len(dL)):
		A[i,j] = dL[j]**0.55*(Ao[i]+1.) - 1.
		temp = np.append([temp],  [ dL[j]**0.55*(Ao[i]+1.) - 1.])

fig=plt.figure(figsize=(18,6))
gs=GridSpec(1,2, hspace=0.0, wspace = 0.4)
ax=fig.add_subplot(gs[0,0])

sc = ax.pcolor(dL, Ao, A, cmap = trymap, vmin = 0, vmax = 8) #abs(A).min(), vmax = abs(A).max()*0.75 )#cm.RdBu cm.jet
cbar = plt.colorbar(sc)
cbar.ax.set_ylabel(r'A at L')
plt.ylabel(r'A at L$_{o}$')
plt.xlabel(r'L$_{o}$/L')
plt.ylim(min(Ao), (max(Ao)))
plt.xlim(min(dL),max(dL))


Lo = 6.8
L = 5.8
Ao = np.arange(100)/3. + 0.01
A = (Lo/L)**0.55*(Ao+1.) -1.

ax=fig.add_subplot(gs[0,1])
ax.plot(Ao, A)
plt.ylim(0,2.5)
plt.xlim(0,2)
plt.ylabel(r'A at L = 5.8')
plt.xlabel(r'A$_{o}$ at L = 6.8')
plt.tight_layout()
plt.savefig('figures/figure_3.png')    
plt.close()

##&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
##&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
##&&&&&&&&&&&&&&&&&&&&&&&&&&&  Figure 4 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
##&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
##&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
##**********************************************************************
##Now we'll create the EMIC growth plot
##**********************************************************************
const =pc.plasmaSI()
constcgs = pc.plasmaCGS()
npts = 5000
S = np.zeros([5, npts])
Xm = np.zeros([5, 3])
pltomegaf = np.zeros(10)  
for loop in range(len(S[:,0])):     
    if loop == 0:
        A = np.array([0.75, 0.55, 0.55], dtype = 'd')

    if loop == 1:
        A = np.array([0.80, 0.60, 0.60], dtype = 'd')

    if loop == 2:
        A = np.array([0.85, 0.65, 0.65], dtype = 'd')

    if loop == 3:
        A = np.array([0.90, 0.70, 0.70], dtype = 'd')

    if loop == 4:
        A = np.array([0.95, 0.75, 0.75], dtype = 'd')

                                # Parallel thermal velocity 
                                #alpha = (2.*kb/(mj*mp))^.5
    Tperpj =  np.array([11.64, 12.99, 9.41], dtype = 'd')*10.**3.*1.6*10.**(-19.) #here we first convert to joules
    Tperp = np.array([11.64, 12.99, 9.41], dtype = 'd')*10.**3.     #the array is in keV, I've kept it in keV because Tperp isn't
                                #used anywhere else in the program,
                                #and since we tend to measure things
                                #in keV it seemed easiest to keep it
                                #in those units.
    denc = np.array([19.85, 1.40, 8.05 ], dtype = 'd')*100.**3. # in units of m^-3

    denw = np.array([0.42, 0.03, 0.17], dtype = 'd')*100.**3. # in units of m^-3

                                #equatorial magnetic field 
                                #bo = 300.d * 10^(-9.) * 10^(4.) ;gauss
    Bo =  160.0 * 10**(-9.) #tesla
    #Bo = 0.007*10**(-4.) #using Kozyra et al 1984  L = 4.5 see table 2 and figure 13

                                #speed of light 
                                #c = double(2.9979*10^8.) *10^2. ;cm/s 
    #c = double(2.9979*10^8.)      ;m/s

                                #mass of particles
                                #me = 9.1d*10^(-31.)*10^3. ;g this is the mass of the electron
    #me = 9.1d*10^(-31.)           ;Kg this is the mass of the electron
                                #mp = 1836.0d*9.1d*10^(-31.)*10^3. ;g this is the mass of the proton
    mp = 1836.0*9.1*10**(-31.)   #Kg this is the mass of the proton

                                #charge
                                #q = 1.61d*10^(-19.)*3.d*10^9. ;statcolumbs
    q = 1.61*10**(-19.)           #columbs

                                #permittivity of free space
                                #epso = 1. ;
    epso = 8.8542*10**(-12.)      #F/m

                                #permeability of free space
                                #epso = 1. ;
    muo = 4.*np.pi*10**(-7.)        #F/m

                                #Boltzmann constant
                                #kb = 1.3807d*10^(-16.) ;erg/k
    #  kb = 1.3807d*10^(-23.)        #j/K

                                #proton cyclotron frequency 
    omegap = q*Bo/(mp)
    omegaf = q*Bo/(2.*np.pi*mp)
    pltomegaf[loop] = omegaf

                                #proton plasma frequency warm and then cold
                                #wppw = (denc(0)*q^2./(mp))^.5   ;C^2/(m^2*kg*F)
    wppw = (denw[0]*q**2./(mp*epso))**.5  #C^2/(m^2*kg*F)
                                #wppc = (denw(0)*q^2./(mp))^.5    ;C^2/(m^2*kg*F)
    wppc = (denc[0]*q**2./(mp*epso))**.5 #C^2/(m^2*kg*F)


                                #delta (ratio of the cold to warm proton plasma frequencies. )
    delta2 = wppc**2./wppw**2.       #unitless def. from Kozyra et al 1984.
    delta = np.sum(denc)/np.sum(denw[0])       #this is what Steve says, but what I had reduces to that.
    #print 'delta used = ', delta # they give you the same thing. 
    #print 'my delta = ', delta2



                                #the atomic mass of the particles
    mj = np.array([1.0, 4.0, 16.0], dtype = 'd')      #in atomic mass units
  
                                  #particle plasma frequency 
                                #wpw = (denc*q^2./(mj*mp))^.5  ;C^2/(m^2*kg*F)
    wpw = (denw*q**2./(mj*mp*epso))**.5 #C^2/(m^2*kg*F)
                                #wpc = (denw*q^2./(mj*mp))^.5 ;C^2/(m^2*kg*F)
    wpc = (denc*q**2./(mj*mp*epso))**.5 #C^2/(m^2*kg*F)


    Tpar = (Tperp/(A + 1.0))*1.6*10.**(-19.) # the array is still in keV then converted to Joules
    Tparj =  (Tperpj/(A + 1.0))              #this is where we use the tperp which was first converted to joules and then found Tpar. 
    alpha = (2.0*Tpar/(mj*mp))**.5            #I removed the boltzman constant 
    #print 'tpar(keV then Joules)  =', Tpar   #here I just check that the two methods for finding the parallel temp is the same thing. 
    #print 'tparj(joules from tperp) = ', Tparj
  
                                #alpha = ((2.0d*Tpar/(mj*mp))*10^7.)^.5 ;I removed the boltzman constant CGS???????

                                #normalized mass density ratio of the ion mass to the mass of the
                                #proton over the atomic number. The second array (all ones) is the z
                                #varible which is the ion charge. In this case we are assuming that it
                                #is always 1.
    #  M = mj/([1.0d, 2.0d, 8.0d]*mp)   #this should also be unitless
    #   M = [1./21., 4./21., 16./21.]                                
    M = mj
                                #The ratio of the real part of the complex dispersion relation
                                #to the proton cyclotron frequency. This is a normalized thing so as
                                #that goes from about 0 - 1 so that's what I'm going to try for now.
    X = np.arange(npts, dtype = 'd')/npts + .00001 #this should be unitless. The + .00001 is to get rid of 
                                #the problem of when x = .5
                                #when I had x = findgen(50)/50.d then I got something closer to what
                                #the paper had. 


                                #normalized ratio of warm to cold plasma frequencies*********** IS
                                #In Kozyra et al 1984 it's the
                                #ratio of the warm or cold plasma
                                #frequency to the warm proton plasma
                                #frequency times what ever M is.  **********
    #  nuw = (M)*(wpw/wppw)^2.         #unitless; this is from Kozyra et al 1984
    #  nuc = (M)*(wpc/wppw)^2.         #unitless; this is from Kozyra et al 1984
    nuc = (denc/denw[0]) # * (M/mj)         #reduces to the above
    nuw = (denw/denw[0]) #*(M/mj)           #reduces to the above

                                #beta = (8.d*!pi*nuw*kb*Tpar)/Bo ;this
                                #might need to be multiplied by 1.6.d*10^(-19.) to convert to jules
    beta = (denw*Tpar)/(Bo**2.0/(2.0*muo))#* 1.6*10**(-19.) #this might need to be multiplied by 1.6.d*10^(-19.) to convert to jules


                                #these are the bits that form the group velocity and mu which is the
                                #ratio of the dispersion relation to the proton cyclotron
                                #frequency. This is done just because it's easier then writting
                                #the same thing over and over again. 
    expon = np.zeros([len(M), len(X)])
    expnum = np.zeros([len(M), len(X)])
    expdenom = np.zeros(len(X))
    sum1 = np.zeros(len(X))
    sum2 = np.zeros(len(X))
    sum3 = np.zeros(len(X))
    insum1 = np.zeros([len(M), len(X)])
    insum2 = np.zeros([len(M), len(X)])
    insum3 = np.zeros([len(M), len(X)])
    frac1 = np.zeros([len(M), len(X)])
                                #frac4 = expon
    brack1 = np.zeros([len(M), len(X)])
                                #brack2 = expon


                                #here we find the inside of the sum in the exponent in S
    for i in range(len(M)):
        insum2[i,:] = ((nuw[i] + nuc[i]) * M[i]) / (1.0 - M[i] * X)
                                #here we sum the sum in the exponent
                                #in S which is for only the heavy ions
                                #Helium and oxygen which in this case

    print( 'insum2[0,100]', insum2[0,100])
    print( 'nuw[0], nuc[0], M[0], X[0]', nuw[0], nuc[0], M[0], X[0])
                                #are in spots array([Hydrogen, helium, oxygen])
    for ii in range(len(X)):
        sum2[ii] = insum2[1,ii]+insum2[2,ii]
                                #here we find the denominator in the exponent (it has the same array
                                #demensions as x)
                                #expdenom = ((1.d+delta)/(1.d - x))+sum2
    expdenom = ((1.0 + delta)/(1.0 - X)) + sum2
                                #here we find the numerator in the exponent, and then find the
                                #exponent for that array value of M, then we also find the array
                                #values of M in the first part of the first sum in S
    for j in range(len(M)): 
        expnum[j,:] = (-1.0*nuw[j]/M[j])*((M[j]*X-1.0)**2./(beta[j]*X**2.))   
        expon[j,:] = expnum[j,:]/expdenom
                                #   frac1(j,*) = ((nuw(j)*sqrt(!pi))/(m(j)^2.*alpha(j)))*((A(j) + 1.d)*(1.d-M(j)*X)-1.d)
        frac1[j,:] = ((nuw[j]*np.sqrt(np.pi))/(M[j]**2.*alpha[j]))*((A[j] + 1.0)*(1.0-M[j]*X)-1.0)
        
                                #here we find the entire inside of that first sum in S
    for jj in range(len(M)):
        insum1[jj,:] = frac1[jj,:]*np.exp(expon[jj,:])
                                #here we go through and sum that first sum at every x point.
    for jjj in range(len(X)):
        sum1[jjj] = np.sum(insum1[:,jjj])

                                #now on to the second squiggly brackets in S
                                #for ij = 0l, n_elements(m)-1 do insum3(ij,*) = (nuw(ij)+nuc(ij))*(m(ij)/(1.d-m(ij)*x))
    for ij in range(len(M)):
        insum3[ij,:] = (nuw[ij]+nuc[ij])*(M[ij]/(1.0 - M[ij]*X))
                                #Again this sum is only done over the
                                #heavy ions just as in the sum in the
                                #exponent. 
    for iij in range(len(X)):
        sum3[iij] = insum3[1,iij] + insum3[2,iij]
    frac3 = (1.0 + delta)/(1.0 - X)
    brack2 = ((2.0*X**2.)*(frac3 + sum3))


                                #now finally we can calculate S
    S[loop, :] = sum1/brack2
    #print sum1[100], brack2[100]

    #Stop band calc
    print( A, M, Xm)
    for ijj in range(len(A)):
        Xm[loop, ijj] = A[ijj]/(M[ijj]*(1. + A[ijj]))
    print( 'after loop', A, M, Xm)


## #**********************************************************************************
## #Now we're going to plot the EMIC wave growth and EMIC wave together.
## #**********************************************************************************

pltlabel = ['A = [0.75, 0.55, 0.55]','A = [0.80, 0.60, 0.60]','A = [0.85, 0.65, 0.65]','A = [0.90, 0.70, 0.70]','A = [0.95, 0.75, 0.75]','']
fig, ax = plt.subplots(ncols = 2, sharey = True)
fig.set_size_inches(20, 8.5)

ax[0].plot(S[0, :]*10**9., X* pltomegaf[0], label = pltlabel[2], linewidth = 1.5, c = cli)#,label = 'A = [1.0, 0.5, 0.9]', c = 'y')
ax[0].plot(S[1, :]*10**9., X* pltomegaf[1], label = pltlabel[2], linewidth = 1.5, c = clk)#,label = 'A = [0.9, 0.4, 0.8]', c = 'b')
ax[0].plot(S[2, :]*10**9., X* pltomegaf[2], label = pltlabel[2], linewidth = 1.5, c = cdr)#,label = 'A = [0.8, 0.3, 0.7]', c = 'r')
ax[0].plot(S[3, :]*10**9., X* pltomegaf[3], label = pltlabel[2], linewidth = 1.5, c = cdc)#,label = 'A = [0.7, 0.2, 0.6]', c = 'g')
ax[0].plot(S[4, :]*10**9., X* pltomegaf[4], label = pltlabel[2], linewidth = 1.5, c = cld)



ax[0].plot(np.arange(1000)/1000.*250., np.zeros(1000) + (pltomegaf[0]), 'k--', linewidth= 1)
ax[0].plot(np.arange(1000)/1000.*250., np.zeros(1000) + (pltomegaf[0]/4),'k--', linewidth= 1)
ax[0].plot(np.arange(1000)/1000.*250., np.zeros(1000) + (pltomegaf[0]/16.), 'k--', linewidth= 1)

ax[0].fill_between(np.arange(1000)/1000.*250., (pltomegaf[0]), np.max(Xm[:,0]* omegaf), color = cdi)# alpha=0.25, color='blue')
ax[0].fill_between(np.arange(1000)/1000.*250., (pltomegaf[0]/4.), np.max(Xm[:,1]* omegaf), color = cdi)# alpha=0.25, color='red')
ax[0].fill_between(np.arange(1000)/1000.*250., (pltomegaf[0]/16.), np.max(Xm[:,2]* omegaf),  color = cdi)#alpha=0.25, color='green')

ax[0].legend(loc='upper right', frameon=False)
ax[0].set_xlim(0,10)
ax[0].set_ylim(0.1,3)
ax[0].set_ylabel('Frequency Hz')
ax[0].set_xlabel(r'EMIC Growth $\times 10^{9}$')



RBSPB_mag = 'data/rbsp-b_magnetometer_hires-gsm_emfisis-L3_20140109_v1.3.1.cdf'

data = pycdf.CDF(RBSPB_mag) #open(nearrealmag_ephm, 'r')
temptimeMAG = data['Epoch'][:]



#Here we are creating the plot of the wave power spectrogram. 
index = np.where((temptimeMAG >= starttime ) & (temptimeMAG <= endtime))
B = data['Mag'][:]
Bx = B[index,0][0,:]
By = B[index,1][0,:]
Bz = B[index,2][0,:]
t = np.arange(len(Bx))
Bmag = np.array([np.sqrt(Bz[i]**2 + By[i]**2 + Bx[i]**2) for i in range(len(Bx))])
BH = np.array([np.sqrt(By[i]**2 + Bx[i]**2) for i in range(len(Bx))])



Pxx, Freq, bins, cax = ax[1].specgram(BH, NFFT=1024*2, Fs=64, Fc=0, detrend='linear',window=mlab.window_hanning, noverlap=256*4,cmap=trymap, xextent=None, pad_to=None, sides='default',scale_by_freq=None, mode='default', scale='default')#viridis
fig.colorbar(cax)
cax.set_clim(vmin=-30, vmax=0.)
xmin = (plttimestart - starttime).seconds
xmax = (plttimeend - starttime).seconds
ax[1].set_xlim(xmin, xmax)
ax[1].set_xlabel('9 Jan. 2014 UT')
ax[1].set_ylim(.1, 3)
tt = t/64.#np.linspace(bins[0], bins[-1], num = len(t))

Hc = (const['e']*Bmag*10**-9)/(2.*np.pi*const['m_p'])
Hec = (const['e']*Bmag*10**-9)/(2.*np.pi*4.*const['m_p'])
Hoc = (const['e']*Bmag*10**-9)/(2.*np.pi*16.*const['m_p'])
ax[1].plot(tt, Hc, 'w')
ax[1].plot(tt, Hec, 'w')
ax[1].plot(tt, Hoc, 'w')
a=ax[1].get_xticks().tolist()

dtlabels = [starttime +  dt.timedelta(seconds = a[i] ) for i in range(len(a))]

newlabels = [dtlabels[i].strftime('%H:%M') for i in range(len(dtlabels))]
ax[1].set_xticklabels(newlabels)
#plt.show()
plt.tight_layout()
plt.savefig('figures/figure_4.png')
plt.close()






## #**********************************************************************
## #***********The chorus and Hiss wave growth Figure 5 and 6  ***********
## #**********************************************************************


#From Ondrej the relevant parameters for the chorus and hiss wave growth values

f =np.array([0.0600000, 0.101818, 0.143636, 0.185455, 0.227273, 0.269091, 0.310909, 0.352727, 0.394545, 0.436364, 0.478182, 0.520000, 0.561818, 0.603636, 0.645455, 0.687273, 0.729091, 0.770909, 0.812727, 0.854545, 0.896364, 0.938182, 0.980000, 1.02182, 1.06364, 1.10545, 1.14727, 1.18909, 1.23091, 1.27273, 1.31455, 1.35636, 1.39818, 1.44000, 1.48182, 1.52364, 1.56545, 1.60727, 1.64909, 1.69091, 1.73273, 1.77455, 1.81636, 1.85818, 1.90000, 1.94182, 1.98364, 2.02545, 2.06727, 2.10909, 2.15091, 2.19273, 2.23455, 2.27636, 2.31818, 2.36000, 2.40182, 2.44364, 2.48545, 2.52727, 2.56909, 2.61091, 2.65273, 2.69455, 2.73636, 2.77818, 2.82000, 2.86182, 2.90364, 2.94545, 2.98727, 3.02909, 3.07091, 3.11273, 3.15455, 3.19636, 3.23818, 3.28000, 3.32182, 3.36364, 3.40545, 3.44727, 3.48909, 3.53091, 3.57273, 3.61455, 3.65636, 3.69818, 3.74000, 3.78182, 3.82364, 3.86545, 3.90727, 3.94909, 3.99091, 4.03273, 4.07455, 4.11636, 4.15818, 4.20000])


a1 = np.array([4.06180*10**-5, 0.00536292, 0.0379852, 0.105913, 0.195673, 0.291108, 0.381291, 0.460574, 0.526838, 0.579970,  0.620867, 0.650856,  0.671378, 0.683831, 0.689492, 0.689496, 0.684826, 0.676327, 0.664721, 0.650618, 0.634534, 0.616904, 0.598093, 0.578411, 0.558114, 0.537419, 0.516507,  0.495529,  0.474611, 0.453856, 0.433353, 0.413170, 0.393367, 0.373990,  0.355076, 0.336655,  0.318749, 0.301375, 0.284545,  0.268267, 0.252544,  0.237378, 0.222768,  0.208709, 0.195196, 0.182224,  0.169783,  0.157863,  0.146457,  0.135552, 0.125137, 0.115200, 0.105730, 0.0967148, 0.0881408, 0.0799959, 0.0722675, 0.0649430, 0.0580098,  0.0514553, 0.0452671, 0.0394330, 0.0339409, 0.0287786, 0.0239346, 0.0193970,  0.0151547, 0.0111963,  0.00751100,  0.00408798, 0.000916751, -0.00201295, -0.00471117, -0.00718776, -0.00945226, -0.0115140, -0.0133822, -0.0150657, -0.0165732, -0.0179130, -0.0190936, -0.0201227, -0.0210083, -0.0217579, -0.0223789, -0.0228784,  -0.0232634, -0.0235407, -0.0237168,  -0.0237982, -0.0237909, -0.0237010,  -0.0235343, -0.0232965,  -0.0229930,  -0.0226290, -0.0222098, -0.0217402,  -0.0212251, -0.0206690])

a2 = np.array([ -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -0.000000, -1.08461*10**-39, -5.96130*10**-35, -4.73254*10**-31, -8.66138*10**-28, -5.07037*10**-25, -1.20313*10**-22, -1.37916*10**-20, -8.72152*10**-19, -3.37049*10**-17, -8.62443*10**-16, -1.55721*10**-14, -2.08818*10**-13, -2.16817*10**-12, -1.80378*10**-11, -1.23702*10**-10, -7.16116*10**-10, -3.57023*10**-9, -1.55912*10**-8, -6.05138*10**-8, -2.11364*10**-7, -6.71573*10**-7, -1.95932*10**-6, -5.29193*10**-6, -1.33271*10**-5, -3.14924*10**-5, -7.02185*10**-5, -0.000148463, -0.000298963, -0.000575651, -0.00106358, -0.00189162, -0.00324776, -0.00539680, -0.00869985, -0.0136341, -0.0208123, -0.0309994, -0.0451258, -0.0642947, -0.0897819, -0.123028, -0.165620, -0.219269, -0.285769, -0.366959, -0.464669, -0.580670, -0.716614, -0.873976, -1.05400, -1.25765, -1.48555, -1.73797, -2.01480, -2.31548, -2.63907, -2.98419, -3.34909, -3.73161, -4.12927, -4.53928, -4.95860, -5.38398, -5.81203, -6.23925,  -6.66213, -7.07716, -7.48090, -7.87004, -8.24143, -8.59212, -8.91943, -9.22091, -9.49441, -9.73810, -9.95048, -10.1303, -10.2768, -10.3894, -10.4678, -10.5121, -10.5226, -10.5001, -10.4452, -10.3592, -10.2433, -10.0990, -9.92784, -9.73162, -9.51219, -9.27147, -9.01145])


b1 = np.array([8.74410*10**-41, 5.87103*10**-23, 1.55488*10**-15,1.74601*10**-11, 6.08307*10**-9,3.33026*10**-7,6.05643*10**-6, 5.42092*10**-5, 0.000299413, 0.00117303, 0.00356842,0.00896168, 0.0193963, 0.0373200, 0.0653173, 0.105807,0.160768, 0.231534, 0.318677, 0.421984, 0.540495, 0.672601,0.816177, 0.968718,1.12748,1.28963, 1.45232, 1.61284, 1.76865, 1.91744, 2.05718,2.18614, 2.30289, 2.40632, 2.49559,2.57015,2.62968, 2.67412, 2.70359,2.71842,2.71905,2.70609, 2.68022, 2.64223, 2.59296,2.53331,2.46418, 2.38651,2.30124, 2.20930, 2.11160,2.00902, 1.90244,1.79266,1.68048, 1.56663, 1.45181, 1.33667,1.22180,1.10778, 0.995105,0.884242, 0.775612, 0.669589, 0.566508, 0.466663, 0.370309, 0.277666, 0.188915, 0.104210, 0.0236706,-0.0526111, -0.124570, -0.192164, -0.255377,-0.314213, -0.368692,-0.418857, -0.464764,-0.506481,-0.544095,-0.577699,-0.607400, -0.633311,-0.655556,-0.674263,-0.689568,-0.701609, -0.710531,-0.716480,-0.719604, -0.720056,-0.717985,-0.713544,-0.706885,-0.698158,-0.687514,-0.675100,-0.661064, -0.645548])


b2 = np.array([0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.09026*10**-37, 1.66174*10**-34, 1.14664*10**-31, 4.02626*10**-29, 7.90934*10**-27, 9.39682*10**-25, 7.20211*10**-23, 3.75868*10**-21, 1.39785*10**-19, 3.85057*10**-18, 8.11932*10**-17, 1.34828*10**-15, 1.80688*10**-14, 1.99615*10**-13, 1.85198*10**-12, 1.46659*10**-11, 1.00564*10**-10, 6.04668*10**-10, 3.22402*10**-9, 1.53968*10**-8, 6.64470*10**-8, 2.61221*10**-7, 9.42163*10**-7, 3.13794*10**-6, 9.70689*10**-6, 2.80371*10**-5, 7.59757*10**-5, 0.000193998, 0.000468610, 0.00107472, 0.00234791, 0.00490094, 0.00980159, 0.0188294, 0.0348271, 0.0621550, 0.107244, 0.179229, 0.290608, 0.457880, 0.702039, 1.04885, 1.52875, 2.17631, 3.02907,  4.12587, 5.50436,  7.19819, 9.23346, 11.6251, 14.3730, 17.4587, 20.8421, 24.4590, 28.2205, 32.0121, 35.6955, 39.1110, 42.0815, 44.4173, 45.9227, 46.4022, 45.6680, 43.5472, 39.8884, 34.5684, 27.4974,  18.6233, 7.93494, -4.53633, -18.7167,  -34.4911, -51.7048, -70.1683, -89.6609,  -109.937, -130.733, -151.769,  -172.763, -193.430, -213.493, -232.684, -250.756,  -267.480])

hissgrowth = a1+a2
chorusgrowth = b1+b2

## **********************************************************************
## ********************The chorus wave plot Figure 5 ********************
## **********************************************************************
## *********************************************************************************
# Here we are going to plot the chorus wave observed at either RBSPB or A (which ever one is selected).
# **********************************************************************************
fig = plt.figure(figsize=[10, 5])
ax2 = fig.add_subplot(1, 2, 1,)
ax2.semilogy(chorusgrowth, f*10**3., c = 'steelblue')
ax2.set_xlim(0,60)
ax2.set_ylim(10, 10**4)
ax2.set_ylabel('frequency (Hz)')
ax2.set_xlabel(r'S(s$^{-1}$)')



#here we are setting the line style/markers for plotting the results. 
style = '--'


# *********************************************************************************
# Here we are going to plot the Chorus wave observed at RBSPB
# **********************************************************************************

# Here we are reading in the survey data
RBSPB_mag = 'data/rbsp-b_wna-survey_emfisis-L4_20140109_v1.2.2.cdf'
#Now we are setting the start and end times for the interval 
#we want to look at and what we will plot
starttime = dt.datetime(2014, 1, 9, 19, 55)
plttimestart = dt.datetime(2014, 1, 9, 20, 5)
endtime = dt.datetime(2014, 1, 9, 20, 35)
plttimeend = dt.datetime(2014, 1, 9, 20, 20)


# Here we are creating the plot of the wave power spectrogram. 
data = pycdf.CDF(RBSPB_mag) 
time = data['Epoch'][:]
Bsum = data['bsum'][:]
Bmag = Bsum
freq = data['WFR_frequencies'][0,:]
Buvw = data['Buvw'][:]
B = np.sqrt(Buvw[:,0]**2 + Buvw[:,1]**2 + Buvw[:,2]**2)

#Finding the electron cyclotron frequency
ec = (const['e']*B*10**-9)/(2.*np.pi*const['m_e'])

#Here are the times for the compression and all in mdate format for the plot labels
plttime = mdates.date2num(time)
starttime = mdates.date2num(plttimestart)
endtime = mdates.date2num(plttimeend)

ax3 = fig.add_subplot(1, 2, 2)
xx, yy = np.meshgrid(plttime, freq)
im = ax3.pcolormesh(xx,yy,Bmag.T, cmap = colormap,norm = LogNorm(vmin = 10**(-10), vmax = 10**(-4)))
ax3.set_ylim(10**1,10**4)
ax3.set_xlim(starttime, endtime)
cb = fig.colorbar(im, ax=ax3)
#cb = plot.colorbar()
ax3.set_yscale('log')
cb.set_ticks([10**(-10), 10**(-8), 10**(-6),10**(-4)])
cb.set_ticklabels([-10, -8, -6, -4 ])
cb.ax.set_ylabel(r'Log(nT$^{2}$/Hz)')
ax3.set_yticklabels([])


## Here we are overplotting the electron cycltron frequency and the half electron cyclotron frequency
ax3.plot(plttime, ec, '--', c = clg, label = r'a)')
ax3.plot(plttime, ec/2., '--', c = clg)
ax3.set_xlabel('9 January 2014')
date_format = mdates.DateFormatter('%H:%M')
ax3.xaxis.set_major_formatter(date_format)
plt.tight_layout()
plt.savefig('figures/figure_5.png')
plt.close()


## #******************************************************
## #***********The Hiss wave growth Figure 6 ***********
## #******************************************************

## #*********************************************************************************
## #Here we are going to plot the chorus wave observed at either RBSPB or A (which ever one is selected).
## #**********************************************************************************
## print 'starting chorus wave bit'
fig = plt.figure(figsize=[10, 5])
ax4 = fig.add_subplot(1, 2, 1,)

ax4.semilogy(hissgrowth, f*10**3., c = 'steelblue')
ax4.set_ylim(10, 10**4)
ax4.set_xlim(0,3)
ax4.set_ylabel('frequency (Hz)')
ax4.set_xlabel(r'S(s$^{-1}$)')

starttime = dt.datetime(2014, 1, 9, 19, 55)
plttimestart = dt.datetime(2014, 1, 9, 20, 5)
endtime = dt.datetime(2014, 1, 9, 20, 35)
plttimeend = dt.datetime(2014, 1, 9, 20, 20)
RBSPA_mag = 'data/rbsp-a_wna-survey_emfisis-L4_20140109_v1.5.3.cdf' 

#Here we are creating the plot of the wave power spectrogram. 
data = pycdf.CDF(RBSPA_mag) #open(nearrealmag_ephm, 'r')
temptimeMAG = data['Epoch'][:]


index = np.where((temptimeMAG >= starttime ) & (temptimeMAG <= endtime))
Bsum = data['bsum'][:]
Bmag = Bsum
freq = data['WFR_frequencies'][0,:]
time = temptimeMAG

plttime = mdates.date2num(time)

starttime = mdates.date2num(plttimestart)
endtime = mdates.date2num(plttimeend)

#fig, ax = plt.subplots(nrows = 3, sharex = True)

ax5 = fig.add_subplot(1, 2, 2)
xx, yy = np.meshgrid(plttime, freq)
plt.pcolormesh(xx,yy,Bmag.T, cmap = trymap,norm = LogNorm(vmin = 10**(-10), vmax = 10**(-4)))
plt.yscale("log", nonposy='clip')
plt.ylim(10,10000)
plt.xlim(starttime, endtime)
plt.colorbar()

plt.xlim(starttime, endtime)
date_format = mdates.DateFormatter('%H:%M')
ax5.xaxis.set_major_formatter(date_format)
ax5.set_xlabel('9 January 2014')



plt.tight_layout()
plt.savefig('figures/figure_6.png')
plt.close()



