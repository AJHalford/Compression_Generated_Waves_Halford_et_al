#This program plots the barrel data on a blue marble polar projection map.
#It wouldn't take much to make this more general but for the moment it's plotting full days and just for a specific set of payloads. If you want to change the payloads make sure to also change the set of if statements at the begining of the for loop.

#This program also only seems to run in python and not ipython due to problems getting mpl_toolkits.basemap to install on mac osX. Hopefully this problem will be solved in future releases, but currently hasn't been.

#To run this program, in the comand line (not inside of python or ipython) type $python Jan_9_map.py

#version 1 
#Date - Jan 29 2016
#By Alexa Halford Alexa.Halford@gmail.com

import numpy as np
import datetime as dt
from datetime import timedelta
import h5py
import scipy
import os
os.environ["CDF_LIB"] =  "/Applications/cdf/cdf37_1-dist/lib"
#os.putenv("CDF_LIB", "~/CDF/LIB")
from spacepy import pycdf
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade

#This just stores then the info about who wrote this code and what version it is in 
__version__ = 1.0
__author__ = 'A.J. Halford'
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

#Payloads I want plotted
payload =  np.array(['2I',  '2W', '2T', '2K', '2L', '2X', 'RBSP-A', 'RBSP-B'])
#Days I want position plotted for
day = 9
dt = plttimestart
##############################################################################################
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
fig = plt.figure(figsize=[10, 5])
ortho = ccrs.Orthographic(central_longitude=0, central_latitude=90)
geo = ccrs.Geodetic()

#This will be the L and MLT plot
ax1 = fig.add_subplot(1, 2, 1, projection='polar')# projection=ccrs.Orthographic(0, 90))
#ax1 = fig.add_axes([0.1, 0.1, .4, .4], polar=True)
#Here we're making the L MLT plot 
#Here are the dummy variables used to help create the color bar

#now we are going to create a unit cirle that will be plotted on each figure.
unitr = [1 for i in range(3600)]
unitr2 = [2 for i in range(3600)]
unitr4 = [4 for i in range(3600)]
unitr6 = [6 for i in range(3600)]
unitr8 = [8 for i in range(3600)]
unitr10 = [10 for i in range(3600)]
unitt = np.arange(3600)*np.pi/1800.


ax1.plot(unitt, unitr, c = cdd)
ax1.plot(unitt, unitr2, c = cdc)
ax1.plot(unitt, unitr4, c = cdr)
ax1.plot(unitt, unitr6, c = cli)
ax1.plot(unitt, unitr8, c = cdi)
#ax1.plot(unitt, unitr10, c = cdi)


r6 = [1 for i in range(180)]
theta6 = np.arange(180)*np.pi/180. - np.pi/2
ax1.fill(theta6, r6, c = cdi)
place = np.array([0,np.pi/2., np.pi, 1.5*np.pi])
lab = ['','dawn', 'noon', 'dusk']
plt.xticks(place, lab)#, )
ax1.set_ylim(0, 10)
ax1.set_xlim(0,2.*np.pi)

ax1.set_frame_on(False)
#ax1.axes.get_xaxis().set_visible(False)
#ax1.axes.get_yaxis().set_visible(False)

#This will be the geographic plot
ax = fig.add_subplot(1, 2, 2, projection=ccrs.Orthographic(180, -90))
ax.coastlines(zorder=3)
ax.stock_img()
ax.gridlines()
ax.add_feature(Nightshade(dt))

for i in range(len(payload)):
	print('starting to plot payload ', payload[i])
	#Here we are just defining the different colors to be used. This could also be done outside the loop in an array and then use mkrcolor[i] in the plotting portion, This was just an easy way for me to make sure I knew which payload was which color... The program would run faster if done the way mentioned above, but it's pretty quick already. Save to change when doing a larger project. 
	if payload[i] == '2I':
		mkrcolor = cdk #'navy'
			#Make sure we have the right sting. If I were to do this for a longer period of time, the same would need to be done for the month ect....
		if day < 10:
			strday = '0' + np.str(np.int(day))
		else:
			strday = np.str(day)
		#Read in the data and get the relavent data bits.
		payload_ephm = 'data/bar_'+payload[i]+'_l2_ephm_201401'+strday+'_v04.cdf'
		data = pycdf.CDF(payload_ephm)
		temptimeMAG = data['Epoch'][:]
		templat = data['GPS_Lat'][:]
		templon = data['GPS_Lon'][:]
		tempalt = data['GPS_Alt'][:]
		tempL2 = data['L_Kp2'][:]
		tempMLT2 = data['MLT_Kp2_T89c'][:]
		mkshape = 'o'
		
	if payload[i] == '2W':
		mkrcolor = cdi #'lightblue'
		#Make sure we have the right sting. If I were to do this for a longer period of time, the same would need to be done for the month ect....
		if day < 10:
			strday = '0' + np.str(np.int(day))
		else:
			strday = np.str(day)
		#Read in the data and get the relavent data bits.
		payload_ephm = 'data/bar_'+payload[i]+'_l2_ephm_201401'+strday+'_v04.cdf'
		data = pycdf.CDF(payload_ephm)
		temptimeMAG = data['Epoch'][:]
		templat = data['GPS_Lat'][:]
		templon = data['GPS_Lon'][:]
		tempalt = data['GPS_Alt'][:]
		tempL2 = data['L_Kp2'][:]
		tempMLT2 = data['MLT_Kp2_T89c'][:]
		mkshape = 'o'
		
	if payload[i] == '2T':
		mkrcolor = cli #'blue'
			#Make sure we have the right sting. If I were to do this for a longer period of time, the same would need to be done for the month ect....
		if day < 10:
			strday = '0' + np.str(np.int(day))
		else:
			strday = np.str(day)
		#Read in the data and get the relavent data bits.
		payload_ephm = 'data/bar_'+payload[i]+'_l2_ephm_201401'+strday+'_v04.cdf'
		data = pycdf.CDF(payload_ephm)
		temptimeMAG = data['Epoch'][:]
		templat = data['GPS_Lat'][:]
		templon = data['GPS_Lon'][:]
		tempalt = data['GPS_Alt'][:]
		tempL2 = data['L_Kp2'][:]
		tempMLT2 = data['MLT_Kp2_T89c'][:]
		mkshape = 'o'
		
	if payload[i] == '2K':
		mkrcolor = cdc #'red'
		#Make sure we have the right sting. If I were to do this for a longer period of time, the same would need to be done for the month ect....
		if day < 10:
			strday = '0' + np.str(np.int(day))
		else:
			strday = np.str(day)
		#Read in the data and get the relavent data bits.
		payload_ephm = 'data/bar_'+payload[i]+'_l2_ephm_201401'+strday+'_v04.cdf'
		data = pycdf.CDF(payload_ephm)
		temptimeMAG = data['Epoch'][:]
		templat = data['GPS_Lat'][:]
		templon = data['GPS_Lon'][:]
		tempalt = data['GPS_Alt'][:]
		tempL2 = data['L_Kp2'][:]
		tempMLT2 = data['MLT_Kp2_T89c'][:]
		mkshape = 'o'
		
	if payload[i] == '2L':
		mkrcolor = cdd #'gold'
		#Make sure we have the right sting. If I were to do this for a longer period of time, the same would need to be done for the month ect....
		if day < 10:
			strday = '0' + np.str(np.int(day))
		else:
			strday = np.str(day)
		#Read in the data and get the relavent data bits.
		payload_ephm = 'data/bar_'+payload[i]+'_l2_ephm_201401'+strday+'_v04.cdf'
		data = pycdf.CDF(payload_ephm)
		temptimeMAG = data['Epoch'][:]
		templat = data['GPS_Lat'][:]
		templon = data['GPS_Lon'][:]
		tempalt = data['GPS_Alt'][:]
		tempL2 = data['L_Kp2'][:]
		tempMLT2 = data['MLT_Kp2_T89c'][:]
		mkshape = 'o'
		
	if payload[i] == '2X':
		mkrcolor = cdr #'orange'
		#Make sure we have the right sting. If I were to do this for a longer period of time, the same would need to be done for the month ect....
		if day < 10:
			strday = '0' + np.str(np.int(day))
		else:
			strday = np.str(day)
		#Read in the data and get the relavent data bits.
		payload_ephm = 'data/bar_'+payload[i]+'_l2_ephm_201401'+strday+'_v04.cdf'
		data = pycdf.CDF(payload_ephm)
		temptimeMAG = data['Epoch'][:]
		templat = data['GPS_Lat'][:]
		templon = data['GPS_Lon'][:]
		tempalt = data['GPS_Alt'][:]
		tempL2 = data['L_Kp2'][:]
		tempMLT2 = data['MLT_Kp2_T89c'][:]
		mkshape = 'o'
	
	if payload[i] == 'RBSP-A':
		mkrcolor = cli #'orange'
		mkshape = '^'
		payload_ephm = 'data/rbspa_def_MagEphem_T89Q_20140109_v1.0.0.h5'
		rbsp_ephm = h5py.File(payload_ephm)
		temptimeMAG= rbsp_ephm['IsoTime'][:]
		temptimeMAG = np.array([dt.strptime(temptimeMAG[j].decode("utf-8") , '%Y-%m-%dT%H:%M:%SZ') for j in range(len(temptimeMAG))])
		tempL2 = rbsp_ephm['Lm_eq'][:]
		tempMLT2 = rbsp_ephm['CDMAG_MLT'][:]
		temp_latlon = rbsp_ephm['Pfs_geod_LatLon'][:]
		templat = temp_latlon[:, 0]
		templon = temp_latlon[:, 1]
		tempalt = rbsp_ephm['Pfs_geod_Height'][:]

	if payload[i] == 'RBSP-B':
		mkrcolor = cld #'orange'
		mkshape = '^'
		payload_ephm = 'data/rbspb_def_MagEphem_T89Q_20140109_v1.0.0.h5'
		rbsp_ephm = h5py.File(payload_ephm)
		temptimeMAG= rbsp_ephm['IsoTime'][:]
		temptimeMAG = np.array([dt.strptime(temptimeMAG[j].decode("utf-8") , '%Y-%m-%dT%H:%M:%SZ') for j in range(len(temptimeMAG))])
		tempL2 = rbsp_ephm['Lm_eq'][:]
		tempMLT2 = rbsp_ephm['CDMAG_MLT'][:]
		temp_latlon = rbsp_ephm['Pfs_geod_LatLon'][:]
		templat = temp_latlon[:, 0]
		templon = temp_latlon[:, 1]
		tempalt = rbsp_ephm['Pfs_geod_Height'][:]

	#Making sure that we are only plotting times that have good data since bad data is identified with numbers and not nans. 
	good = np.where(tempalt > 0.)
	time = temptimeMAG[good]
	mlt = tempMLT2[good]
	alt = tempalt[good]
	lat = templat[good]
	lon = templon[good]
	Lvalue = tempL2[good]
	MLT = tempMLT2[good]

	good = np.where(lat > -90)
	time = time[good]
	mlt = mlt[good]
	alt = alt[good]
	lat = lat[good]
	lon = lon[good]
	Lvalue = tempL2[good]
	MLT = tempMLT2[good]
	
	good = np.where(lon > -180)
	time = time[good]
	mlt = mlt[good]
	alt = alt[good]
	lat = lat[good]
	lon = lon[good]    
	Lvalue = tempL2[good]
	MLT = tempMLT2[good]
	
	plttime = np.where((time>plttimestart) & (time < plttimeend))
	time = time[plttime]
	mlt = mlt[plttime]
	alt = alt[plttime]
	lat = lat[plttime]
	lon = lon[plttime]
	Lvalue = tempL2[plttime]
	MLT = tempMLT2[plttime]

	#Here we are making the L and MLT plot
	cs = ax1.scatter(MLT[0]*np.pi/12 , Lvalue[0], 45, marker= mkshape, color = mkrcolor, edgecolors = cdi, label = payload[i])
	print('the L and MLT values are ', Lvalue[0], MLT[0])
	#Here we are making the geographic map 
	print('the marker is at ', lon[0], lat[0])
	cs = ax.scatter(lon[0], lat[0], 35, marker= mkshape, color = mkrcolor,transform=ccrs.Geodetic(), edgecolors = cdi,label = payload[i])
	#plt.annotate(payload[i], xy=(lon[0], lat[0]), color = mkrcolor ,transform=ccrs.Geodetic())
    #now that we've mapped all the different days label the payload at the end. 
    #points = ortho.transform_points(geo,lon[0]+2, lat[0]+2)
    
#ax1.legend(ncol=3)
ax.legend(ncol=3)
# 
#txtcolor = 'lightslategray'
# #Add in some stations
# #Halley VI
# x, y= map(-26.66, -75.58)
# cs = map.scatter(x,y,10,  marker='h',color='g')
# plt.annotate('Halley VI', xy=(x, y), color = txtcolor )
# #SANAE IV
# x,y = map(-2.84,-74.67)
# cs = map.scatter(x,y,10,  marker='h',color='g')
# plt.annotate('SANAE IV', xy=(x, y), color = txtcolor )
# #South pole
# x,y = map(-180,-90)
# cs = map.scatter(x,y,10,  marker='h',color='g')
# plt.annotate('South Pole', xy=(x, y), color = txtcolor )
# #McMurdo
# x,y = map(166.7, -77.9)
# cs = map.scatter(x,y,10,  marker='h',color='g')
# plt.annotate('McMurdo', xy=(x, y), color = txtcolor )
# #AGO P1
# x,y = map(129.6,-83.9)
# cs = map.scatter(x,y,10,  marker='h',color='g')
# plt.annotate('P1', xy=(x, y), color = txtcolor )
# #AGO P2
# x,y = map(-46.4,-85.7)
# cs = map.scatter(x,y,10,  marker='h',color='g')
# plt.annotate('P2', xy=(x, y), color = txtcolor )
# #AGO P3
# x,y = map(28.6,-82.8)
# cs = map.scatter(x,y,10,  marker='h',color='g')
# plt.annotate('P3', xy=(x, y), color = txtcolor )
# #AGO P4
# x,y = map(96.8,-82.0)
# cs = map.scatter(x,y,10,  marker='h',color='g')
# plt.annotate('P4', xy=(x, y), color = txtcolor )
# #AGO P5
# x,y = map(123.5,-77.2)
# cs = map.scatter(x,y,10,  marker='h',color='g')
# plt.annotate('P5', xy=(x, y), color = txtcolor )
# #AGO P6
# x,y = map(130,-69.5)
# cs = map.scatter(x,y,10,  marker='h',color='g')
# plt.annotate('P6', xy=(x, y), color = txtcolor )

#plt.show()
plt.savefig('figures/figure_2.png')
#plt.close()

#
