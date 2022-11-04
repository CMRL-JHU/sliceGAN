import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

path_main_output = './pipeline_output';

EBSDGS = "/home/brayan.murgas/Experimental/Microstructures/Al/ebsd_Al7050/Al7050YZ_small_HAGB5_RXGOS_GS_fit/IndFiles/Al7050YZ_small_HAGB5_RXGOS_GS_fit_2D_Rx_allG_ECD.dat"

dream3d_11  = "./Ng11/5-grainDiameter.txt"
dream3d_20  = "./Ng20/5-grainDiameter.txt"
dream3d_53  = "./Ng53/5-grainDiameter.txt"
dream3d_109 = "./Ng109/5-grainDiameter.txt"
dream3d_154 = "./Ng154/5-grainDiameter.txt"
dream3d_209 = "./Ng209/5-grainDiameter.txt"
dream3d_308 = "./Ng308/5-grainDiameter.txt"
dream3d_402 = "./Ng402/5-grainDiameter.txt"
dream3d_493 = "./Ng493/5-grainDiameter.txt"
dream3d_602 = "./Ng602/5-grainDiameter.txt"

OutName = path_main_output + "/6-KS_test_GS"

#EBSDgbDiso = "/home/brayan.murgas/Experimental/Microstructures/Al/ebsd_Al7050/Al7050YZ_small_HAGB5_RXGOS_GS_fit/IndFiles/Al7050YZ_small_HAGB5_RXGOS_GS_fit_2D_Rx_allG_disorientation.txt"

#dream3d_11  = "./Ng11/5-misorientationList.txt"
#dream3d_20  = "./Ng20/5-misorientationList.txt"
#dream3d_53  = "./Ng53/5-misorientationList.txt"
#dream3d_109 = "./Ng109/5-misorientationList.txt"
#dream3d_154 = "./Ng154/5-misorientationList.txt"
#dream3d_209 = "./Ng209/5-misorientationList.txt"
#dream3d_308 = "./Ng308/5-misorientationList.txt"
#dream3d_402 = "./Ng402/5-misorientationList.txt"
#dream3d_493 = "./Ng493/5-misorientationList.txt"
#dream3d_602 = "./Ng602/5-misorientationList.txt"

#OutName = path_main_output + "/6-KS_test_D"

#files = [EBSDGS, dream3d_11, dream3d_20, dream3d_53, dream3d_109, dream3d_154]
#labels = ["EBSD", "$N_G$=10", "$N_G$=19", "$N_G$=52", "$N_G$=108", "$N_G$=153"]
#linesList = ["", "s", "*", "x", "D", "."]
#files = [EBSDGS, dream3d_209, dream3d_308, dream3d_402, dream3d_493, dream3d_602]
#labels = ["EBSD", "$N_G$=208", "$N_G$=307", "$N_G$=401", "$N_G$=492", "$N_G$=601"]
#linesList = ["", "", "", "", "", ""]

files = [EBSDGS, dream3d_11, dream3d_20, dream3d_53, dream3d_109, dream3d_154, dream3d_209, dream3d_308, dream3d_402, dream3d_493, dream3d_602]
#files = [EBSDgbDiso, dream3d_11, dream3d_20, dream3d_53, dream3d_109, dream3d_154, dream3d_209, dream3d_308, dream3d_402, dream3d_493, dream3d_602]
labels = ["EBSD", "$N_G$=10", "$N_G$=19", "$N_G$=52", "$N_G$=108", "$N_G$=153", "$N_G$=208", "$N_G$=307", "$N_G$=401", "$N_G$=492", "$N_G$=601"]
linesList = ["", "", "", "", "", "", "", "", "", "", ""]
#linesList = ["_", "s", "*", "x", "D", ".", "o", "3", "h", "+", "^"]

#files = [EBSDGS, dream3d_11]
#labels = ["EBSD", "$N_G$=11"]

labels1 = ["Grain size"]
#labels1 = ["Disorientation"]
linesList1 = ["o"]

nbBins = 20

xLabel = "Grain size [$\mu m$]"
#xLabel = "Disorientation [$°$]"
yLabel = "CDF"

xLabel1 = "$N_G$"
yLabel1 = "CDF (KS)"

NbGrains=[10,19,52,108,153,208,307,401,492,601] 

Xm = 150
#Xm = 200

sourceData = []
DataY=[]
DataCDF=[]

maxValueToHist=-9999999999999999999
minValueToHist=999999999999999999

## Read data
for file in files:
    #print(file)
    source = open(file,"r")
    sourceData = source.readlines()
    DataY.append([])
    DataCDF.append([])
    y = 1. * np.arange(len(sourceData)) / (len(sourceData) - 1)
    for i in range(1,len(sourceData)):
        valueY = float(sourceData[i])
        DataY[len(DataY)-1].append(valueY)
    DataY[len(DataY)-1].sort()
    DataCDF[len(DataY)-1]=y.tolist()
    for i in range(0,len(DataY[len(DataY)-1])):
        value=DataY[len(DataY)-1][i]
        if value > maxValueToHist : 
            maxValueToHist=value
        if value < minValueToHist : 
            minValueToHist=value
#    print(len(DataY))
#    print(len(DataY[len(DataY)-1]))
#    print(DataY[len(DataY)-1])
#    print(DataCDF[len(DataY)-1])
maxValueToHist = maxValueToHist*1.05
print("Max Value =" + str(maxValueToHist))
print("Min Value =" + str(minValueToHist))

KSvals=[]
#NbGrains=[]
for ifn in range(1,len(DataY)):
    ksvals= stats.ks_2samp(DataY[0], DataY[ifn])
    KSvals.append(ksvals.statistic)
#    NbGrains.append(len(DataY[ifn]))
print(KSvals)
print(NbGrains)

#plt.clf()
#fig= plt.figure() #(figsize=(12,8), dpi=150)
#plt.grid(b=True, linestyle=':', linewidth=1)
#axes = plt.gca()
#plt.xlabel(xLabel, fontsize=18)
#plt.ylabel(yLabel, fontsize=18)
#axes.set_xlim([minValueToHist*0.9,maxValueToHist*1.4]) #set_xlim([xMin,xMax])
##plt.title(r"$"+'t='+str(step*timeStepPlot+initialTimePlot)+"$", fontsize=24)
#for ifn in range(0, len(DataY)):
#    plt.plot(DataY[ifn], DataCDF[ifn][1:], label=labels[ifn], marker=linesList[ifn]) #color=colorList[ifn],
##plt.legend(prop={'size': 24})
#plt.legend(prop={'size': 15}, ncol=1)
#plt.tick_params(axis = 'both', which = 'major', labelsize = 15) # Make bigger the tick of the axes (both X and Y)
#plt.subplots_adjust(left=0.15, bottom=0.13, right=0.95, top=0.95)
#plt.savefig(OutName+"_CDF.pdf", format='pdf')
#plt.savefig(OutName+"_CDF.png", format='png')
#plt.close()

plt.clf()
fig= plt.figure() #(figsize=(12,8), dpi=150)
plt.grid(b=True, linestyle=':', linewidth=1)
axes = plt.gca()
plt.xlabel(xLabel1, fontsize=18)
plt.ylabel(yLabel1, fontsize=18)
#plt.title(r"$"+'t='+str(step*timeStepPlot+initialTimePlot)+"$", fontsize=24)
########### Set limits
#axes.set_xlim([-10,630])
#axes.set_ylim([0.01,0.28])
axes.set_xlim([-10,630])
axes.set_ylim([0.04,0.34])
########### Define rectangle
plt.axvline(Xm, linewidth=1.0, color="black")
rect=mpatches.Rectangle((150,0.00),480,0.35, #(200,0.00),430,0.28, 
                        #fill=False,
                        alpha=0.1,
                        #color="purple",
                       #linewidth=2,
                       facecolor="black")
plt.gca().add_patch(rect)
########### Plot
plt.plot(NbGrains, KSvals, label=labels1[0], marker=linesList1[0]) #color=colorList[ifn],
########### Text
text = axes.text(
            400, #415,
            0.15,
#            "%.1f" % Xm,
            "Convergence",
            transform=axes.transData,
            size=15,
            rotation=0,
            ha="center",
            va="top",
            #weight="bold",
            bbox={'facecolor': 'white', 'alpha': 0.8, 'edgecolor': 'black', 'pad': 6},
#            bbox={'facecolor': 'white', 'alpha': 1.0, 'pad': 10},
        )
########### 
plt.legend(prop={'size': 24})
#plt.legend(prop={'size': 15}, ncol=2)
plt.tick_params(axis = 'both', which = 'major', labelsize = 15) # Make bigger the tick of the axes (both X and Y)
plt.subplots_adjust(left=0.15, bottom=0.13, right=0.95, top=0.95)
plt.savefig(OutName+".pdf", format='pdf')
plt.savefig(OutName+".png", format='png')
plt.close()

#function [misorientations_target, YMisorientationReference] = initialize_RapidKS(misorientations_target_Unsorted)
#%%% format misorientations, initialize misorientation reference
#%%% for Max's RapidKS test
#% why are two extra values added to the misorientations here? -- josh
#misorientations_target_nonUnique = sort([0;misorientations_target_Unsorted;63]); % Max Pinz
#% why do we need unique floating point indecies between 0 and 1? --josh
#YMisorientationReference_nonUnique = 0:1/(numel(misorientations_target_nonUnique)-1):1; % Max Pinz
#% okay so we have a unique issue here that needs to be managed  % Max Pinz
#[misorientations_target,Ia,Ic] = unique(misorientations_target_nonUnique); % Max Pinz
#YMisorientationReference = YMisorientationReference_nonUnique(Ia); % Max Pinz
#end

#def initialize_RapidKS(data):
#    




