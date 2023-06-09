# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 10:05:42 2023

@author: Francesco Mirani
"""

import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib import colors
import PIL
from PIL import Image
from mpl_point_clicker import clicker
from mpl_interactions import zoom_factory, panhandler
from typing import Tuple
import math
from time import sleep, perf_counter
from scipy import signal

# Fubction to get the index of the closest value in array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Functions to add and remove points from figure
def point_added_cb(position: Tuple[float, float], klass: str):
    x, y = position
    print(f"New point of class {klass} added at {x=}, {y=}")

def point_removed_cb(position: Tuple[float, float], klass: str, idx):
    x, y = position
    suffix = {'1': 'st', '2': 'nd', '3': 'rd'}.get(str(idx)[-1], 'th')
    print(f"The {idx}{suffix} point of class {klass} with position {x=:.2f}, {y=:.2f}  was removed")

# Function to convert the proton energis to positions
def postion_from_energy(Ener, mass, Z, L_B, L_S, B):
    Ener = Ener*1.60218e-13 # MeV to Joule
    q=1.6*10**-19; # charge
    velocity = np.sqrt(2*Ener/mass) # m/s
    R = mass*velocity/(Z*q*B)
    position_B = R - np.sqrt(R**2 - L_B**2) + L_B*L_S/np.sqrt(R**2 - L_B**2)   
    return position_B


###########################
# Set analysis parameters #
###########################
# import the TPS image
Name_folder_and_file = 'Data/SHOT 3.tif'
img = np.asarray(Image.open(Name_folder_and_file))
# Detector params
B = 0.2 #T
Len_B = 0.104 # magnet size [m]
Len_S = 0.105 # distance magnet sensor [m]
delta=1.01*10**4; # pixel/m
# Image cuts
y_start = 460
y_end = 990
x_start = 860
x_end = 1130
threshold = 1 # units of std over the bg
width_traces = 20 # number of pixels that includes the trace
carbon_charge = 6 # charge state of C ions
# Transpose image (X-ray spot must be in the botton-right positions and parabolas with positive concavity)
img = np.transpose(img)
img = img[x_start:x_end, y_start:y_end]


##################
# Find max index #
##################
# Average over the region where the number of maximum adjacent pixels is higher
x_max_vect = np.argwhere(img == img.max())[0:,1]
y_max_vect = np.argwhere(img == img.max())[0:,0]
x_max_list = [[x_max_vect[0]]]
y_max_list = [[y_max_vect[0]]]
id_list = 0
id_list_longer = 0
for i in np.arange(1,len(x_max_vect)):    
    if ((y_max_list[-1][id_list] == y_max_vect[i] - 1) or (y_max_list[-1][id_list] == y_max_vect[i])):
        y_max_list[-1].append(y_max_vect[i])
        x_max_list[-1].append(x_max_vect[i])
        id_list = id_list + 1
    else:
        if ((len(y_max_list) > 1) and  (len(y_max_list[-1]) > len(y_max_list[id_list_longer]))):
            id_list_longer = len(y_max_list) - 1
        y_max_list.append([y_max_vect[i]])
        x_max_list.append([x_max_vect[i]])
        id_list = 0
if ((len(y_max_list) > 1) and  (len(y_max_list[-1]) > len(y_max_list[id_list_longer]))):
    id_list_longer = len(y_max_list) - 1
x_max = round(np.mean(x_max_list[id_list_longer]))
y_max = round(np.mean(y_max_list[id_list_longer]))
######################
# Plot the MCP image #
######################
fig, (ax1, ax2, ax3) = plt.subplots(1,3)
fig.set_figheight(5)
fig.set_figwidth(16)
ax1.set_xlabel("Pixel")
ax1.set_ylabel("Pixel")
ax1.set_title("1) Select edges of the region with parabolas")
ax2.set_xlabel("Parabola coefficient")
ax2.set_ylabel("Occurrences")
ax2.set_title("2) Select edges of the intervals for the peaks")
ax3.set_xlabel("E [MeV]")
ax3.set_ylabel("Average signal [arb. units]")
ax3.set_title("3) Select the maximum energies")
pos = ax1.imshow(img, cmap=cm.magma, 
                norm=colors.LogNorm(vmin = img[1:-1, 1:-1].min(), vmax = img[1:-1, 1:-1].max()),
                aspect = "auto")
#fig.colorbar(pos, ax=ax1, location='top', anchor=(0.5, 0.3), shrink=0.6)
ax1.scatter(x_max, y_max, color ="green", edgecolors = "black", linewidths = 0.5) 
fig.tight_layout(w_pad=8.0)
# add zooming and middle click to pan
zoom_factory(ax1)
ph = panhandler(fig, button=2)
# Get window of data interactively
klicker = clicker(
   ax1,
   ["Edges"],
   markers=["x"]
)
klicker.on_point_added(point_added_cb)
klicker.on_point_removed(point_removed_cb)
# Get the limit for the plot signals
while True:
    limits_data = klicker.get_positions().get('Edges')
    plt.pause(0.01)
    if len(limits_data) > 3:
        break
x_window_min = round(min(limits_data[:,0]))
x_window_max = round(max(limits_data[:,0]))
y_window_min = round(min(limits_data[:,1]))
y_window_max = round(max(limits_data[:,1]))
# plot rectangle containing the selected region
ax1.plot([x_window_min, x_window_max], [y_window_min, y_window_min], color ="black", linewidth = 0.5) 
ax1.plot([x_window_min, x_window_max], [y_window_max, y_window_max], color ="black", linewidth = 0.5) 
ax1.plot([x_window_min, x_window_min], [y_window_min, y_window_max], color ="black", linewidth = 0.5) 
ax1.plot([x_window_max, x_window_max], [y_window_min, y_window_max], color ="black", linewidth = 0.5) 
fig.canvas.draw()
fig.canvas.flush_events()
######################################################################
# Perform the Hough transfor of the data over bachground + threshold #
######################################################################
# Hough transofrmation with y-yc=a*(x-xc)^2 in selected region
backgr_mean=np.mean(img); 
backgr_std=np.mean(img); 
v=[];
for i in np.arange(x_window_min, x_window_max):  
    for j in np.arange(y_window_min, y_window_max):  
        if img[j,i] > backgr_mean + backgr_std*threshold:
            a = (j - y_max)/(i - x_max)**2
            if (a < 0 and a != -math.inf):
                v.append(a)
v = np.array(v)
# Histogram of the "a" coff. (peaks identify the parabolas)
n_bins_a = round(abs((x_end - x_start)*(y_end - y_start)/100))
h=np.histogram(v, n_bins_a, (min(v[v > -10])*1.1, 0))
Hough_hist_x = h[1]
Hough_x = np.zeros(len(h[1]) - 1)
for i in np.arange(1,len(Hough_hist_x)):    
    Hough_x[i - 1] = (Hough_hist_x[i - 1] + Hough_hist_x[i])/2
Hough_y = h[0]
# Make histogram of the occurrances of "a"
ax2.step(Hough_x,Hough_y, linewidth=2, color = "darkblue")
ax2.set_xlim([min(Hough_x), max(Hough_x)])
ax2.set_ylim([0, max(Hough_y)*1.1])
# Get window of data interactively
klicker = clicker(
   ax2,
   ["Extr."],
   markers=["x"]
)
klicker.on_point_added(point_added_cb)
klicker.on_point_removed(point_removed_cb)
#####################################
# Get average slope of the parabola #
#####################################
# Select interactively the intervals and perform average mean over curves
flag_mark_rig = 0
while True:
    peaks_pos = klicker.get_positions().get('Extr.')
    plt.pause(0.01)
    numerator = Hough_x*Hough_y
    if len(peaks_pos) == 2 and flag_mark_rig == 0:
        slope_1 = sum(numerator[(Hough_x > peaks_pos[0,0]) & (Hough_x < peaks_pos[1,0])])/sum(Hough_y[(Hough_x > peaks_pos[0,0]) & (Hough_x < peaks_pos[1,0])])
        ax2.fill_between(Hough_x, max(Hough_y)*1.1, where = (Hough_x > peaks_pos[0,0]) & (Hough_x < peaks_pos[1,0]),
                color='purple', alpha=0.1)
        ax2.plot([slope_1, slope_1], [0, max(Hough_y)*1.1], color ="black", linewidth = 0.5) 
        flag_mark_rig = 1;
        fig.canvas.draw()
        fig.canvas.flush_events()
    elif len(peaks_pos) == 4:
        slope_2 = sum(numerator[(Hough_x > peaks_pos[2,0]) & (Hough_x < peaks_pos[3,0])])/sum(Hough_y[(Hough_x > peaks_pos[2,0]) & (Hough_x < peaks_pos[3,0])])
        ax2.fill_between(Hough_x, max(Hough_y)*1.1, where = (Hough_x > peaks_pos[2,0]) & (Hough_x < peaks_pos[3,0]),
                color='purple', alpha=0.1)
        ax2.plot([slope_2, slope_2], [0, max(Hough_y)*1.1], color ="black", linewidth = 0.5) 
        fig.canvas.draw()
        fig.canvas.flush_events()
        break
# Re-order slopes (largest in abs for carbon ions)
slope = [slope_1, slope_2]
slope.sort()
slope_p = slope[1]
slope_c = slope[0]
x_par = np.linspace(x_window_min, x_window_max,500)
# Add parabolas to plot for protons exluding velues outside region of interest
y_par_prot = y_max + slope_p*(-x_par + x_max)**2
x_par_prot = x_par[y_par_prot > y_window_min]
y_par_prot = y_par_prot[y_par_prot > y_window_min]
ax1.plot(x_par_prot, y_par_prot, color ="black", linewidth = 0.5) 
# Add parabolas to plot for carbons exluding velues outside region of interest
y_par_carb = y_max + slope_c*(-x_par + x_max)**2
x_par_carb = x_par[y_par_carb > y_window_min]
y_par_carb = y_par_carb[y_par_carb > y_window_min]
ax1.plot(x_par_carb, y_par_carb, color ="black", linewidth = 0.5) 
fig.canvas.draw()
fig.canvas.flush_events()
############################################
# Retrieve the signals along the parabolas #
############################################
# Mean signal over trace for protons - bg
cumulate_signal_prot = np.array([])
cumulate_signal_bg = np.array([])
x_eval_sig_prot = np.array([round(x_par_prot[0])])
while x_eval_sig_prot[-1] < x_par_prot[-1]:  
    y_eval_signal = y_par_prot[find_nearest(x_eval_sig_prot[-1], x_par_prot)]
    y_eval_signal = round(y_eval_signal - width_traces/2)
    cumulate_signal_prot = np.append(cumulate_signal_prot,0)
    cumulate_signal_bg = np.append(cumulate_signal_bg,0)
    n_pixels_signals =0
    n_pixels_bg =0
    for i in np.arange(y_eval_signal,y_eval_signal + width_traces):
        if(img[i, x_eval_sig_prot[-1]] > backgr_mean + backgr_std*threshold):
            cumulate_signal_prot[-1] = cumulate_signal_prot[-1] + img[i, x_eval_sig_prot[-1]]
            n_pixels_signals = n_pixels_signals + 1
        else:
            cumulate_signal_bg[-1] = cumulate_signal_bg[-1] + img[i, x_eval_sig_prot[-1]]
            n_pixels_bg = n_pixels_bg + 1
    if(n_pixels_signals != 0):
        cumulate_signal_prot[-1] = cumulate_signal_prot[-1]/n_pixels_signals - cumulate_signal_bg[-1]/n_pixels_bg
    x_eval_sig_prot = np.append(x_eval_sig_prot, x_eval_sig_prot[-1] + 1)
x_eval_sig_prot = x_eval_sig_prot[0:-1]
# Mean signal over trace for carbon - bg
cumulate_signal_carb = np.array([])
cumulate_signal_bg = np.array([])
x_eval_sig_carb = np.array([round(x_par_carb[0])])
while x_eval_sig_carb[-1] < x_par_carb[-1]:  
    y_eval_signal = y_par_carb[find_nearest(x_eval_sig_carb[-1], x_par_carb)]
    y_eval_signal = round(y_eval_signal - width_traces/2)
    cumulate_signal_carb = np.append(cumulate_signal_carb,0)
    cumulate_signal_bg = np.append(cumulate_signal_bg,0)
    n_pixels_signals =0
    n_pixels_bg =0
    for i in np.arange(y_eval_signal,y_eval_signal + width_traces):
        if(img[i, x_eval_sig_carb[-1]] > backgr_mean + backgr_std*threshold):
            cumulate_signal_carb[-1] = cumulate_signal_carb[-1] + img[i, x_eval_sig_carb[-1]]
            n_pixels_signals = n_pixels_signals + 1
        else:
            cumulate_signal_bg[-1] = cumulate_signal_bg[-1] + img[i, x_eval_sig_prot[-1]]
            n_pixels_bg = n_pixels_bg + 1
    if(n_pixels_signals != 0):
        cumulate_signal_carb[-1] = cumulate_signal_carb[-1]/n_pixels_signals - cumulate_signal_bg[-1]/n_pixels_bg
    x_eval_sig_carb = np.append(x_eval_sig_carb, x_eval_sig_carb[-1] + 1)
x_eval_sig_carb = x_eval_sig_carb[0:-1]
#####################################################
# Get calibration curve for protons and carbon ions #
#####################################################
m_p=1.66*10**-27;  # proton mass
m_c=1.99*10**-26;  # proton mass
Ener_calib = np.linspace(0.1,300,3000)
Pos_calib_prot = np.zeros(len(Ener_calib))
Pos_calib_carb = np.zeros(len(Ener_calib))
for i in np.arange(len(Ener_calib)):  
    Pos_calib_prot[i] = postion_from_energy(Ener_calib[i], m_p, 1, Len_B, Len_S, B)
    Pos_calib_carb[i] = postion_from_energy(Ener_calib[i], m_c, carbon_charge, Len_B, Len_S, B)
Pos_calib_prot = Pos_calib_prot[::-1]
Pos_calib_carb = Pos_calib_carb[::-1]
Ener_calib = Ener_calib[::-1]
##############
# Get specta #
##############
# Convert pixels to energy and get spectrum for protons
x_pixels_pos_prot = (x_max-x_eval_sig_prot)/delta
Energie_protons = np.zeros(len(x_pixels_pos_prot))
for i in np.arange(len(x_pixels_pos_prot)):  
    Energie_protons[i] = np.interp(x_pixels_pos_prot[i], Pos_calib_prot, Ener_calib)
# Convert pixels to energy and get spectrum for protons
x_pixels_pos_carb = (x_max-x_eval_sig_carb)/delta
Energie_carbons = np.zeros(len(x_pixels_pos_carb))
for i in np.arange(len(x_pixels_pos_carb)):  
    Energie_carbons[i] = np.interp(x_pixels_pos_carb[i], Pos_calib_carb, Ener_calib)
# plot proton spectrum
ax3.plot(Energie_protons, cumulate_signal_prot, linewidth = 1.5, drawstyle = "steps-mid", color = "steelblue", label = "Protons")
ax3.fill_between(Energie_protons, cumulate_signal_prot, step = "mid", color = "steelblue", alpha = 0.2)
ax3.plot(Energie_carbons, cumulate_signal_carb, linewidth = 1.5, drawstyle = "steps-mid", color = "crimson", label = "C ions")
ax3.fill_between(Energie_carbons, cumulate_signal_carb, step = "mid", color = "crimson", alpha = 0.2)
ax3.set_xlim([0, 1.1*max([max(Energie_protons[cumulate_signal_prot > 0]), max(Energie_carbons[cumulate_signal_carb > 0])])])
ax3.set_yscale("log")
ax3.legend(loc="upper right")
fig.canvas.draw()
fig.canvas.flush_events()
#########################
# Select maximum energy #
#########################
# Get window of data interactively
klicker = clicker(
   ax3,
   ["Max."],
   markers=["x"]
)
klicker.on_point_added(point_added_cb)
klicker.on_point_removed(point_removed_cb)
while True:
    Max_en = klicker.get_positions().get('Max.')
    plt.pause(0.01)
    if len(Max_en) > 1:
        break
# plot lines
lim_ax3 = ax3.get_ylim()
ax3.plot([Max_en[0,0], Max_en[0,0]], ax3.get_ylim(), color ="black", linewidth = 0.5) 
ax3.plot([Max_en[1,0], Max_en[1,0]], ax3.get_ylim(), color ="black", linewidth = 0.5) 
#######################
# Save image and data #
#######################
fig.savefig(Name_folder_and_file[:-4] + ' results.png')
np.savetxt(Name_folder_and_file[:-4] + ' results_prot.txt', np.column_stack([Energie_protons, cumulate_signal_prot]))
np.savetxt(Name_folder_and_file[:-4] + ' results_carb.txt', np.column_stack([Energie_carbons, cumulate_signal_carb]))