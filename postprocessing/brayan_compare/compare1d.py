import h5py
import numpy as np
import matplotlib.pyplot as plt
    
def plot_comparison(files, path_data, OutName, nbBins, labels, xLabel, yLabel):
    
    size_figure = [6.4, 4.8]
    dpi = 150
    
    sourceData = []
    DataY=[]

    maxValueToHist=-9999999999999999999
    minValueToHist=999999999999999999

    ## Read data
    for file in files:

        if file.rsplit('.',1)[-1] == "dream3d": # now it can pull from dream3d! --josh
            with h5py.File(file, 'r') as f:
                data = f[path_data][...]
                # all dream.3d data except for linked lists have a component dimension
                # this check ensures that we only take the first dimension
                # which is necessary for aspect ratios
                if len(data.shape) > 1:
                    data = data[:, 0]
                # all linked lists are transposed from the normal top-down direction
                # this check ensures they are formatted the same
                else:
                    data = data.T
                DataY.append(np.sort(data, axis=0))
                minValueToHist = min(minValueToHist, DataY[-1][1:].min())
                maxValueToHist = max(maxValueToHist, DataY[-1]    .max())

        else:
            #print(file)
            source = open(file,"r")
            sourceData = source.readlines()
            DataY.append([])
            for i in range(1,len(sourceData)):
                valueY = float(sourceData[i])
                DataY[len(DataY)-1].append(valueY)
            DataY[len(DataY)-1].sort()
            for i in range(0,len(DataY[len(DataY)-1])):
                value=DataY[len(DataY)-1][i]
                if value > maxValueToHist : 
                    maxValueToHist=value
                if value < minValueToHist : 
                    minValueToHist=value
                    
    maxValueToHist = maxValueToHist*1.05
    print("Max Value =" + str(maxValueToHist))
    print("Min Value =" + str(minValueToHist))
    ## computing ranges
    rangeSize=(maxValueToHist-minValueToHist)/nbBins
    print("maxValue="+str(maxValueToHist) + " minValue="+str(minValueToHist)+ " rangeSize="+str(rangeSize))
    ranges=[None]*(nbBins+1)
    ranges[0]=minValueToHist
    for i in range(1,nbBins+1):
        ranges[i]=ranges[i-1]+rangeSize
        print(ranges[i])
    ranges=np.asarray(ranges)
    print(ranges)

    # Computing  distributions
    Distributions=[]
    for ifn in range(0,len(DataY)):
        Distributions.append([])
        Distributions[ifn]=[0.0]*(nbBins)
        irange = 0
        for i in range(0,len(DataY[ifn])):
            while DataY[ifn][i] > ranges[irange+1]:
                irange+=1
            Distributions[ifn][irange]+=1
        for i in range(0,len(Distributions[ifn])):
            Distributions[ifn][i]/=len(DataY[ifn])
            Distributions[ifn][i]*=100.0
        print(Distributions[ifn])

    plt.clf()
    width = (rangeSize/len(DataY))*0.9
    #fig= plt.figure() #(figsize=(12,8), dpi=150)
    fig = plt.figure(figsize=size_figure, dpi=dpi)
    
    plt.grid(b=True, linestyle=':', linewidth=1)
    axes = plt.gca()
    plt.xlabel(xLabel, fontsize=18)
    plt.ylabel(yLabel, fontsize=18)
    #plt.title(r"$"+'t='+str(step*timeStepPlot+initialTimePlot)+"$", fontsize=24)
    for ifn in range(0, len(DataY)):
        plt.bar(ranges[:-1]+width*ifn, Distributions[ifn], width = width, label=labels[ifn]) #color=colorList[ifn],
    plt.legend(prop={'size': 24})
    plt.tick_params(axis = 'both', which = 'major', labelsize = 15) # Make bigger the tick of the axes (both X and Y)
    plt.subplots_adjust(bottom=0.13, right=0.95, top=0.95)
    plt.tight_layout()
    plt.savefig(OutName+".pdf", format='pdf')
    plt.savefig(OutName+".png", format='png')
    plt.savefig(OutName+"_transparent.png", format='png', transparent=True)
    plt.close()









