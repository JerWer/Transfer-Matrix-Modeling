#! python3

"""

based on transfer matrix python package tmm, written by Steven Byrnes, http://sjbyrnes.com
main code to generate the matrices and calculate the currents written by Gabriel Christmann
readaptation and completion by Jérémie Werner (EPFL PVlab 2014-2018, CU Boulder 2019-2020), 
latest update: April 2020

"""

#%%
import os
import sys
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg")

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QAction
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from scipy.interpolate import interp1d
from scipy import integrate
import numpy as np

import csv

import tmm.tmm_core as tm
import pickle
import math

from TMsimulGUI import Ui_TransferMatrixModeling

admittance0=2.6544E-3
echarge = 1.60218e-19
planck = 6.62607e-34
lightspeed = 299792458

file = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'spectratxtfiles','AM15G.txt'), encoding='ISO-8859-1')
am15g = file.readlines()
file.close()
dataWave = []
dataInt = []
for i in range(len(am15g)):
    pos = am15g[i].find('\t')
    dataWave.append(float(am15g[i][:pos]))
    dataInt.append(float(am15g[i][pos+1:-1]))
SpectIntensFct = interp1d(dataWave,dataInt)
#%%
def eta(index,imaginary_angle,TEorTM):
    res=index*np.cos(imaginary_angle)
    if(TEorTM==0):
        return admittance0*(np.abs(res.real)-1j*np.abs(res.imag))
    else:
        return admittance0*index*index/(np.abs(res.real)-1j*np.abs(res.imag))

def interp2w(value,abscis,ordo):
    return np.interp(value,abscis,ordo)   

def getgenspec(spectrgen,EQE):
    retval=[]
    for ii,wl in enumerate(EQE[0]):
        retval.append(EQE[1][ii]*interp2w(wl,spectrgen[0],spectrgen[1]))
    return [EQE[0],retval]
 
def getcurrent(spectrgen,EQE):
    retval=0
    for ii,wl in enumerate(EQE[0]):
        retval+=EQE[1][ii]*interp2w(wl,spectrgen[0],spectrgen[1])
    return retval

def calcCurrent(x,y,xmin,xmax):
    f = interp1d(x, y, kind='cubic')
    x2 = lambda x: AM15GParticlesinnm(x)*f(x)
    return echarge/10*integrate.quad(x2,xmin,xmax)[0]

def AM15GParticlesinnm(x):
    return (x*10**(-9))*SpectIntensFct(x)/(planck*lightspeed)

        
def elta(index,imaginary_angle,dd,ll):
    res= index*np.cos(imaginary_angle)
    return 2*np.pi*dd*(np.abs(res.real)-1j*np.abs(res.imag))/ll

def importindex(filename):
    index=[[],[]]
    with open(filename, 'rt', encoding='ISO-8859-1') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar="'")
        iterdata=iter(spamreader)
        next(iterdata)
        for row in iterdata:
            index[0].append(float(row[0]))
            index[1].append(float(row[1]))
    return index
def plotdata(graph1,*nextgraphs):
    plt.figure()
    plt.plot(graph1[0],graph1[1])
    for graph in nextgraphs:
        plt.plot(graph[0],graph[1])

def plotdatas(graph1,*nextgraphs):
    plt.figure()
    plt.scatter(graph1[0],graph1[1],s=80,marker='s',color='b')
    for graph in nextgraphs:
        plt.scatter(graph[0],graph[1],s=80,marker='^',color='k')

def niceplotdata(abscisse,ordonnee,graph1,*nextgraphs):
    plt.figure()
    plt.plot(graph1[0][0],graph1[0][1],label=graph1[1])
    for graph in nextgraphs:
        plt.plot(graph[0][0],graph[0][1],label=graph[1])
    plt.xlabel(abscisse)
    plt.ylabel(ordonnee)
    plt.legend(framealpha=0,fancybox=False)
    plt.tight_layout()

def nicescatdata(abscisse,ordonnee,graph1,*nextgraphs):
    plt.figure()
    plt.scatter(graph1[0][0],graph1[0][1],label=graph1[1],s=80,marker='s',color='b')
    for graph in nextgraphs:
        plt.scatter(graph[0][0],graph[0][1],label=graph[1],s=80,marker='^',color='k')
    plt.xlabel(abscisse)
    plt.ylabel(ordonnee)
    plt.legend(framealpha=0,fancybox=False)
    plt.tight_layout()

def writefile(name,xx,yy):
    with open(name, 'w', encoding='ISO-8859-1') as myfile:
        text=''
        for n,x in enumerate(xx):
            text=text+str(x)+'\t'+str(yy[n])+'\n'
        myfile.write(text)
            
def scatdatas(graph1,*nextgraphs):
    plt.figure()
    plt.scatter(graph1[0],graph1[1],s=80,marker='s',color='b')
    for graph in nextgraphs:
        plt.scatter(graph[0],graph[1],s=80,marker='s',color='b')
    plt.tight_layout()
    
def importindex2(filename):
    index=[[],[],[],[]]
    if filename.split('.')[1]=="csv":
        if 'nk_' in filename:
            with open(filename, 'rt', encoding='ISO-8859-1') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar="'")
                iterdata=iter(spamreader)
                next(iterdata)
                for row in iterdata:
                    index[0].append(float(row[0]))#wavelength
                    index[1].append(float(row[1])+1j*float(row[2]))#complex notation n+ik
                    index[2].append(float(row[1]))#n
                    index[3].append(float(row[2]))#k
            return index
        elif 'pvlighthouse' in filename:
            #get boundary conditions: largest of both x columns first value, smallest of both x columns last value
            #interpolate both n and k, with smallest step size of both columns
            #put in index 
            csvdata=[[],[],[],[]]
            with open(filename, 'rt', encoding='ISO-8859-1') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar="'")
                iterdata=iter(spamreader)
                next(iterdata)
                for row in iterdata:
                    if row[0]!='':
                        csvdata[0].append(float(row[0]))
                    if row[1]!='':
                        csvdata[1].append(float(row[1]))
                    if row[2]!='':
                        csvdata[2].append(float(row[2]))
                    if row[3]!='':
                        csvdata[3].append(float(row[3]))
            
            xmin=max([min(csvdata[0]),min(csvdata[2])])
            xmax=min([max(csvdata[0]),max(csvdata[2])])
            stepsize=min([t - s for s, t in zip(csvdata[0], csvdata[0][1:])]+[t - s for s, t in zip(csvdata[2], csvdata[2][1:])])
            fn=interp1d(csvdata[0],csvdata[1], kind='cubic')
            fk=interp1d(csvdata[2],csvdata[3], kind='cubic')
            for i in range(int(math.ceil(xmin)),int(round(xmax)),int(stepsize)):
                index[0].append(i)#wavelength
                index[1].append(fn(i)+1j*fk(i))#complex notation n+ik
                index[2].append(fn(i))#n
                index[3].append(fk(i))#k
            return index
            
    elif filename.split('.')[1]=="txt":
        filetoread = open(filename,"r", encoding='ISO-8859-1')
        filerawdata = filetoread.readlines()
        for row in filerawdata:
            if row[0]!="#" and row!="\n":
                index[0].append(float(row.split("\t")[0]))#wavelength
                index[1].append(float(row.split("\t")[1])+1j*float(row.split("\t")[2]))#complex notation n+ik
                index[2].append(float(row.split("\t")[1]))#n
                index[3].append(float(row.split("\t")[2]))#k
        return index
    elif filename.split('.')[1]=="nk":
        filetoread = open(filename,"r", encoding='ISO-8859-1')
        filerawdata = filetoread.readlines()
        for row in filerawdata:
            if row[0]!="#" and row!="\n":
                try:
                    index[0].append(float(row.split("\t")[0]))#wavelength
                    index[1].append(float(row.split("\t")[1])+1j*float(row.split("\t")[2]))#complex notation n+ik
                    index[2].append(float(row.split("\t")[1]))#n
                    index[3].append(float(row.split("\t")[2]))#k
                except:
                    pass
        return index

def importAM(filename):
    index=[[],[]]
    with open(filename, 'rt', encoding='ISO-8859-1') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar="'")
        for row in spamreader:
            index[0].append(float(row[0]))
            index[1].append(float(row[1]))
    return index

def importAM2(filename):
    index=[[],[]]
    with open(filename, 'rt', encoding='ISO-8859-1') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar="'")
        iterdata=iter(spamreader)
        next(iterdata)
        for row in iterdata:
            index[0].append(float(row[0]))
            index[1].append(float(row[1]))
    return index

def plotall(structure, genspec, lay1,lay2):
    huss=structure.geninalllays(genspec,1,0,1,'s')
    print("Generation in perovskite: ",huss[lay1])
    print("Generation in Si: ",huss[lay2])
    spec1=structure.absspectruminlay(lay1,genspec[0],1,0,1,'s')
    spec2=structure.absspectruminlay(lay2,genspec[0],1,0,1,'s')
    spec3=structure.calculateRTrange(genspec[0],1,0,1,'s')
    plotdata(spec1,spec2,[spec3[0],1-np.asarray(spec3[1])-np.asarray(spec3[2])],[spec3[0],np.asarray(spec1[1])+np.asarray(spec2[1])])

def plotAllLayers(structure,genspec, actlay1,actlay2):
    huss=structure.geninalllays(genspec,1,0,1,'s')
    print("Generation in perovskite: ",huss[actlay1])
    print("Generation in Si: ",huss[actlay2])
    spec1=structure.absspectruminlay(actlay1,genspec[0],1,0,1,'s')
    spec2=structure.absspectruminlay(actlay2,genspec[0],1,0,1,'s')
    spec3=structure.calculateRTrange(genspec[0],1,0,1,'s')
    
    plt.figure()
    
    for item in range(structure.lengthstructure()):
        spec0=structure.absspectruminlay(item+1,genspec[0],1,0,1,'s')
        plt.plot(spec0[0],spec0[1])
    spec=[spec3[0],1-np.asarray(spec3[1])-np.asarray(spec3[2])]
    spec4=[spec3[0],np.asarray(spec1[1])+np.asarray(spec2[1])]
    plt.plot(spec[0],spec[1])
    plt.plot(spec4[0],spec4[1])
    plt.savefig("test.png", dpi=300, transparent=False) 

def plotAllLayersTriple(structure,genspec, actlay1,actlay2,actlay3):
    huss=structure.geninalllays(genspec,1,0,1,'s')
    print("Generation in top: ",huss[actlay1])
    print("Generation in middle: ",huss[actlay2])
    print("Generation in bottom: ",huss[actlay3])
    
    spectop=structure.absspectruminlay(actlay1,genspec[0],1,0,1,'s')
    specmid=structure.absspectruminlay(actlay2,genspec[0],1,0,1,'s')
    specbot=structure.absspectruminlay(actlay3,genspec[0],1,0,1,'s')
    specR=structure.calculateRTrange(genspec[0],1,0,1,'s')
    
    plt.figure()
    
    
    for item in range(structure.lengthstructure()):
        spec0=structure.absspectruminlay(item+1,genspec[0],1,0,1,'s')
        plt.plot(spec0[0],spec0[1])
    
    spec=[specR[0],1-np.asarray(specR[1])-np.asarray(specR[2])]
    spec4=[specR[0],np.asarray(spectop[1])+np.asarray(specmid[1])+np.asarray(specbot[1])]
    plt.plot(spec[0],spec[1])
    plt.plot(spec4[0],spec4[1])
    plt.savefig("test.png", dpi=300, transparent=False) 
    
class multilayer:
    """Class defining a multilayer structure by its name, its components"""
    def __init__(self,*layers):
        self.name=""
        self.structure=[]
        for layer in layers:
            self.structure.append(layer)
    def plotnprofile(self,wavelength):
        zpos=[]
        zn=[]
        pos=0
        for layer in self.structure:
            zpos.append(pos)
            if layer.thickness>1000:#limiting the thickness of layers for visual clarity on the graph
                thick=1000
            else:
                thick=layer.thickness
            pos+=thick
            zpos.append(pos)
            zn.append(layer.material.indexatWL(wavelength))
            zn.append(layer.material.indexatWL(wavelength))
#        plt.plot(zpos,zn)
        return [zpos,zn]
    
    def lengthstructure(self):
        return len(self.structure)

    def addlayer(self,layer):
        self.structure.append(layer)
        
    def addlayerinpos(self,position,layer):
        self.structure.insert(position,layer)
    
    def addbilayers(self,layer_1,layer_2,pairs):
        for uselessindex in range(pairs):
            self.structure.append(layer_1)
            self.structure.append(layer_2)
        
    def addDBR(self,material_1,material_2,wavelength,pairs):
        layer_1=layer(material_1,wavelength/(material_1.indexatWL(wavelength).real*4))
        layer_2=layer(material_2,wavelength/(material_2.indexatWL(wavelength).real*4))
        self.addbilayers(layer_1,layer_2,pairs)
        
    def updatematerial(self,material):
        for materials in self.structure:
            if(material.name==materials.name):
                materials=material
                
    def erasestructure(self):
        self.structure=[]

#    def createTM(self,incident_medium_n,incident_angle,wavelength,TEorTM):
#        transfer_matrix=ml.eye(2)*(1+0j)
#        ndlist=self.makendlist(wavelength)
#        for [n,d] in ndlist:
#            angleincurrentlayer=np.arcsin((1+0j)*incident_medium_n*np.sin(incident_angle*np.pi/180)/n)
#            etac=eta(n,angleincurrentlayer,TEorTM)
#            eltac=elta(n,angleincurrentlayer,d,wavelength)
#            intermediate_matrix=transfer_matrix
#            transfer_matrix=intermediate_matrix*np.matrix([[np.cos(eltac),1j*np.sin(eltac)/etac],[1j*np.sin(eltac)*etac,np.cos(eltac)]])
#        return transfer_matrix
    
    def makendlist(self,wavelength):
        ndlist=[]
        for layers in self.structure:
            ndlist.append([layers.material.indexatWL(wavelength),layers.thickness])
        return ndlist
            
#    def calculateRT(self,incident_medium_n,incident_angle,output_medium_n,wavelength,TEorTM):
#        angleinoutputmedium=np.arcsin((1+0j)*incident_medium_n*np.sin(incident_angle*np.pi/180)/output_medium_n)
#        etai=eta(incident_medium_n,incident_angle*np.pi/180,TEorTM)
#        etao=eta(output_medium_n,angleinoutputmedium,TEorTM)
#        transfer_matrix=self.createTM(incident_medium_n,incident_angle,wavelength,TEorTM)
#        [[bb],[cc]]=(transfer_matrix*np.matrix([[1],[etao]])).tolist()
#        return [np.absolute((etai*bb-cc)/(etai*bb+cc))**2,4*etai.real*etao.real/(np.absolute(etai*bb+cc)**2)]
    
    def calculatetmm(self,incident_medium_n,incident_angle,output_medium_n,wavelength,TEorTM):
        d_list=[np.inf]
        n_list=[incident_medium_n]
        for layers in self.structure:
            d_list.append(layers.thickness)
            n_list.append(layers.material.indexatWL(wavelength))
        d_list.append(np.inf)
        n_list.append(output_medium_n)
        return tm.coh_tmm(TEorTM, n_list, d_list, incident_angle*np.pi/180, wavelength)

    def calculatetmminc(self,incident_medium_n,incident_angle,output_medium_n,wavelength,TEorTM):
        d_list=[np.inf]
        n_list=[incident_medium_n]
        c_list=['i']
        for layers in self.structure:
            d_list.append(layers.thickness)
            n_list.append(layers.material.indexatWL(wavelength))
            #print(layers.material.indexatWL(wavelength))
#            print('huss')
            
            c_list.append(layers.coherence)
        d_list.append(np.inf)
        n_list.append(output_medium_n)
        c_list.append('i')
#        print(n_list)
#        print(d_list)
#        print(c_list)
        return tm.inc_tmm(TEorTM, n_list, d_list, c_list, incident_angle*np.pi/180, wavelength)

    def calculatetmmforwlrange(self,wlrange,incident_medium_n,incident_angle,output_medium_n,TEorTM):
        return [self.calculatetmminc(incident_medium_n,incident_angle,output_medium_n,wl,TEorTM) for wl in wlrange]
        
    def calculateRTrange(self,wlrange,incident_medium_n,incident_angle,output_medium_n,TEorTM):
        huss=self.calculatetmmforwlrange(wlrange,incident_medium_n,incident_angle,output_medium_n,TEorTM)
        RR=[]
        TT=[]
        for ii,transc in enumerate(huss):
            RR.append(transc['R'])
            TT.append(transc['T'])
        return [wlrange,RR,TT]
    
    def absinalllayers(self,wl,incident_medium_n,incident_angle,output_medium_n,TEorTM):
        return tm.inc_absorp_in_each_layer(self.calculatetmminc(incident_medium_n,incident_angle,output_medium_n,wl,TEorTM))

    def absspectruminlay(self,layer,wlrange,incident_medium_n,incident_angle,output_medium_n,TEorTM):
        absinlay=[]
        for wl in wlrange:
            absinlay.append(self.absinalllayers(wl,incident_medium_n,incident_angle,output_medium_n,TEorTM)[layer])
        return [wlrange,absinlay]
    
    def generationinlay(self,layer,generationspectrum,incident_medium_n,incident_angle,output_medium_n,TEorTM):
        geninlay=0
        for nn,wl in enumerate(generationspectrum[0][1:]):
            #print(wl-generationspectrum[0][nn])
            #geninlay+=self.absinalllayers(wl,incident_medium_n,incident_angle,output_medium_n,TEorTM)[layer]*generationspectrum[1][nn]*(wl-generationspectrum[1][nn-1])
            geninlay+=self.absinalllayers(wl,incident_medium_n,incident_angle,output_medium_n,TEorTM)[layer]*generationspectrum[1][nn+1]*(wl-generationspectrum[0][nn])
        return geninlay

    def geninalllays(self,generationspectrum,incident_medium_n,incident_angle,output_medium_n,TEorTM):
        gen=np.zeros(len(self.structure)+2)
        for nn,wl in enumerate(generationspectrum[0][1:]):
            #rint(np.array(self.absinalllayers(wl,incident_medium_n,incident_angle,output_medium_n,TEorTM)))
            gen+=np.array(self.absinalllayers(wl,incident_medium_n,incident_angle,output_medium_n,TEorTM))*generationspectrum[1][nn+1]*(wl-generationspectrum[0][nn])
        return gen
    
    def scanlayerthickness(self,layertoscan0,scanrange,generationspectrum,incident_medium_n,incident_angle,output_medium_n,TEorTM):
        layertoscan=layertoscan0-1
        retval=[scanrange]
        for ii in range(len(self.structure)+2):
            retval.append([])
        for thick in scanrange:
            self.structure[layertoscan].thickness=thick
            gen=self.geninalllays(generationspectrum,incident_medium_n,incident_angle,output_medium_n,TEorTM)
            for ii in range(len(self.structure)+2):
                retval[ii+1].append(gen[ii])
            print(thick)
        return retval

    def scanlayerthickness2D(self,layer1toscan0,layer2toscan0,scanrange1,scanrange2,generationspectrum,incident_medium_n,incident_angle,output_medium_n,TEorTM):
        layer1toscan=layer1toscan0-1
        layer2toscan=layer2toscan0-1
        
        DATA=[]
        
        for thick1 in scanrange1:
            self.structure[layer1toscan].thickness=thick1
            print(thick1)
            print("")
            retval=[scanrange2]
            for ii in range(len(self.structure)+2):
                retval.append([])
            for thick2 in scanrange2:
                self.structure[layer2toscan].thickness=thick2
                gen=self.geninalllays(generationspectrum,incident_medium_n,incident_angle,output_medium_n,TEorTM)
                for ii in range(len(self.structure)+2):
                    retval[ii+1].append(gen[ii])
                print(thick2)
            DATA.append([thick1,[list(x) for x in zip(*retval)]])
            print("")
        return DATA
            
    
    
    def getcurrentinlayerstextu(self,currentspectrum):
        r1TE=np.transpose(np.asarray([tm.inc_absorp_in_each_layer(self.calculatetmminc(1,54.74,1,x,'s')) for x in currentspectrum[0]]))
        r2TE=np.transpose(np.asarray([tm.inc_absorp_in_each_layer(self.calculatetmminc(1,15.79,1,x,'s')) for x in currentspectrum[0]]))
        r1TM=np.transpose(np.asarray([tm.inc_absorp_in_each_layer(self.calculatetmminc(1,54.74,1,x,'p')) for x in currentspectrum[0]]))
        r2TM=np.transpose(np.asarray([tm.inc_absorp_in_each_layer(self.calculatetmminc(1,15.79,1,x,'p')) for x in currentspectrum[0]]))
        listlayers=[]
        for ii,layer in enumerate(r1TE):
            layerTE=[]
            layerTM=[]
            for jj,wlpos in enumerate(layer):
                if(ii==0):
                    layerTE.append(wlpos*r2TE[ii][jj])
                    layerTM.append(r1TM[ii][jj]*r2TM[ii][jj])
                else:
                    layerTE.append(wlpos+r1TE[0][jj]*r2TE[ii][jj])
                    layerTM.append(r1TM[ii][jj]+r1TM[0][jj]*r2TM[ii][jj])
            listlayers.append([np.trapz(np.asarray(layerTE)*np.asarray(currentspectrum[1]),x=currentspectrum[0])/10,np.trapz(np.asarray(layerTM)*np.asarray(currentspectrum[1]),x=currentspectrum[0])/10])
            
        return listlayers
        
        
        
        
class layer:
    """Class defining a layer by its material and thickness"""
    def __init__(self,material,thickness,coherence):
        self.name=""
        self.material=material
        self.thickness=thickness
        self.coherence=coherence
        
class material:
    """Class defining a material by its name and refractive index"""
    def __init__(self,name,indextabl):
        
        self.name=name
        self.indextable=[sorted(indextabl[0]),[x for _,x in sorted(zip(indextabl[0],indextabl[1]))]]
        
    def indexatWL(self,wavelength):
        return np.interp(wavelength,self.indextable[0],[x.real for x in self.indextable[1]])+1j*np.interp(wavelength,self.indextable[0],[x.imag for x in self.indextable[1]])

#%%

LARGE_FONT= ("Verdana", 16)
SMALL_FONT= ("Verdana", 10)

stackDir			= os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'TMstacks')# cell stacks predifined with layer names and thicknesses
matDir			= os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'matdata')	# materials data
resDir        = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'results')
matPrefix		= 'nk_'		# materials data prefix
matHeader		= 1				# number of lines in header
colorstylelist = ['white', 'red', 'blue', 'brown', 'green','cyan','magenta','olive','navy','orange','gray','aliceblue','antiquewhite','aqua','aquamarine','azure','beige','bisque','blanchedalmond','blue','blueviolet','brown','burlywood','cadetblue','chartreuse','chocolate','coral','cornflowerblue','cornsilk','crimson','darkblue','darkcyan','darkgoldenrod','darkgray','darkgreen','darkkhaki','darkmagenta','darkolivegreen','darkorange','darkorchid','darkred','darksalmon','darkseagreen','darkslateblue','darkslategray','darkturquoise','darkviolet','deeppink','deepskyblue','dimgray','dodgerblue','firebrick','floralwhite','forestgreen','fuchsia','gainsboro','ghostwhite','gold','goldenrod','greenyellow','honeydew','hotpink','indianred','indigo','ivory','khaki','lavender','lavenderblush','lawngreen','lemonchiffon','lightblue','lightcoral','lightcyan','lightgoldenrodyellow','lightgreen','lightgray','lightpink','lightsalmon','lightseagreen','lightskyblue','lightslategray','lightsteelblue','lightyellow','lime','limegreen','linen','magenta','maroon','mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen','mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred','midnightblue','mintcream','mistyrose','moccasin','navajowhite','navy','oldlace','olive','olivedrab','orange','orangered','orchid','palegoldenrod','palegreen','paleturquoise','palevioletred','papayawhip','peachpuff','peru','pink','plum','powderblue','purple','red','rosybrown','royalblue','saddlebrown','salmon','sandybrown','seagreen','seashell','sienna','silver','skyblue','slateblue','slategray','snow','springgreen','steelblue','tan','teal','thistle','tomato','turquoise','violet','wheat','white','whitesmoke','yellow','yellowgreen']


#get list of stack names
stacklist=os.listdir(stackDir)
stackNameList=[]
for item in stacklist:
    stackNameList.append(item.split('.')[0])

matlist=os.listdir(matDir)
owd = os.getcwd()
os.chdir(matDir)
matnamelist=[]
material_list={}
material_list_dat={}
for item in matlist:
    if item.split('.')[1]=="csv":
        if 'nk_' in item:
            name=item.split('_')[1].split('.')[0]
            matnamelist.append(name)
            indeximported=importindex2(item)
            material_list[name]=material(name,indeximported)
            material_list_dat[name]=indeximported
        elif 'pvlighthouse' in item:    
            name='pvlh_'+item.split('_')[1].split('.')[0]
            matnamelist.append(name)
            indeximported=importindex2(item)
            material_list[name]=material(name,indeximported)
            material_list_dat[name]=indeximported
    elif item.split('.')[1]=="txt":
        if "interpolated" not in item:
            name=item.split('.')[0]
            matnamelist.append(name)
            material_list[name]=material(name,importindex2(item))
            material_list_dat[name]=importindex2(item)
    elif item.split('.')[1]=="nk":
        if "interpolated" not in item:
            name=item.split('.')[0]
            matnamelist.append("s_"+ name)
            material_list["s_"+ name]=material("s_"+ name,importindex2(item))
            material_list_dat["s_"+ name]=importindex2(item)
matnamelist.sort(key=lambda x: x.lower())
os.chdir(owd)
#with open('material_list.pk', 'wb') as fichier:
#    mon_pickler = pickle.Pickler(fichier)
#    mon_pickler.dump(material_list)

#with open('material_list.pk','rb') as fichier:
#    mon_depickler = pickle.Unpickler(fichier)
#    material_list=pickle.load(mon_depickler)

with open('material_list.pickle', 'wb') as fichier:
    pickle.dump(material_list,fichier,pickle.HIGHEST_PROTOCOL)

with open('material_list.pickle','rb') as fichier:
    material_list=pickle.load(fichier)

AM1p5Gc=importAM2(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'spectratxtfiles','AM1.5Gdf.csv'))

numberofLayer = 0

MatThickActList=[] #list of list with [Material, thickness, Active?, Incoherent?] info in order of stack

#%%#######################
class Help(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        global matnamelist
        
        self.resize(1000, 500)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QtCore.QSize(900, 200))
        self.setWindowTitle("Information")
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.label = QtWidgets.QLabel()
        self.label.setText("""
More info about transfer matrix modeling: https://en.wikipedia.org/wiki/Transfer-matrix_method_(optics)
                             
##################
based on transfer matrix python package tmm, written by Steven Byrnes, http://sjbyrnes.com
main code to generate the matrices and calculate the currents written by Gabriel Christmann (PV-Center, CSEM)
readaptation and completion by Jérémie Werner (EPFL PVlab 2014-2018, CU Boulder 2019-2020), 
latest update: April 2020
                              
##################
IQE=100%
the light is considered entering/exiting from/to an incoherent medium with refractive index of 1 (Air), and perpendicular to the device plane.
all layers are considered flat.""")

        self.gridLayout.addWidget(self.label)
        
class Nkplotwin(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        global matnamelist
        
        self.resize(831, 556)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QtCore.QSize(831, 556))
        self.setWindowTitle("Plot NK data")
        
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.gridLayout.setObjectName("gridLayout")
        self.listWidget_nkplot = QtWidgets.QListWidget(self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.listWidget_nkplot.sizePolicy().hasHeightForWidth())
        self.listWidget_nkplot.setSizePolicy(sizePolicy)
        self.listWidget_nkplot.setObjectName("listWidget_nkplot")
        self.gridLayout.addWidget(self.listWidget_nkplot, 0, 1, 1, 1)
        self.frame = QtWidgets.QFrame(self)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.widget_nkplot = QtWidgets.QWidget(self.frame)
        self.widget_nkplot.setObjectName("widget_nkplot")
        self.gridLayout_nkplot = QtWidgets.QGridLayout(self.widget_nkplot)
        self.gridLayout_nkplot.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_nkplot.setObjectName("gridLayout_nkplot")
        self.gridLayout_2.addWidget(self.widget_nkplot, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)
        
        self.fig1 = Figure()
        self.NKgraph = self.fig1.add_subplot(111)
        self.addmpl(self.fig1,self.gridLayout_nkplot, self.widget_nkplot)
        
        self.listWidget_nkplot.itemClicked.connect(self.updateNkgraph)
        
        
        for item in matnamelist:
            self.listWidget_nkplot.addItem(item)
            
        
        
    def addmpl(self, fig, whereLayout, whereWidget):
        self.canvas = FigureCanvas(fig)
        whereLayout.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, 
                whereWidget, coordinates=True)
        whereLayout.addWidget(self.toolbar)   
        
    def updateNkgraph(self,a):
        global matnamelist, material_list_dat
        
        try:
            w=material_list_dat[matnamelist[self.listWidget_nkplot.currentRow()]][0]
            n=material_list_dat[matnamelist[self.listWidget_nkplot.currentRow()]][2]
            k=material_list_dat[matnamelist[self.listWidget_nkplot.currentRow()]][3]
        except ValueError:
            print("valueerror")
            w=[]
            n=[]
            k=[]
        if len(w)==len(n) and len(w)==len(k):
            pass
        else:
            w=[]
            n=[]
            k=[]

        self.widget_nkplot.close()
        self.widget_nkplot = QtWidgets.QWidget(self.frame)
        self.widget_nkplot.setObjectName("widget_nkplot")
        self.gridLayout_nkplot = QtWidgets.QGridLayout(self.widget_nkplot)
        self.gridLayout_nkplot.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_nkplot.setObjectName("gridLayout_nkplot")
        self.gridLayout_2.addWidget(self.widget_nkplot, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)
        self.fig1 = Figure()
        self.NKgraph = self.fig1.add_subplot(111)
        self.addmpl(self.fig1,self.gridLayout_nkplot, self.widget_nkplot)
        
        
        self.NKgraph.set_xlabel("Wavelength (nm)")
        self.NKgraph.set_ylabel("n")
        self.NKgraph.plot(w,n,'r')
        self.NKgraph.tick_params(axis='y', labelcolor='r')
        
        self.NKgraph2 = self.NKgraph.twinx()
        
        self.NKgraph2.set_ylabel("k")
        self.NKgraph2.plot(w,k,'b')
        self.NKgraph2.tick_params(axis='y', labelcolor='b')
        
        self.fig1.tight_layout()
        self.fig1.canvas.draw() 

reordered=0
class reorderwin(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        global MatThickActList, numberofLayer
        
        self.resize(300, 300)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QtCore.QSize(200, 200))
        self.setWindowTitle("Reorder stack by drag and drop")
        
        self.widget_layout = QtWidgets.QVBoxLayout()

        # Create ListWidget and add 10 items to move around.
        self.list_widget = QtWidgets.QListWidget()
        for i in range(numberofLayer):
            self.list_widget.addItem(str(i)+"_"+MatThickActList[i][1])

        # Enable drag & drop ordering of items.
        self.list_widget.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)

        self.pushButton_reorder = QtWidgets.QPushButton("Validate", self)
        self.widget_layout.addWidget(self.pushButton_reorder)
        self.pushButton_reorder.clicked.connect(self.validate)

        self.widget_layout.addWidget(self.list_widget)
        self.setLayout(self.widget_layout)
    def validate(self):
        global MatThickActList,reordered, window
        #reorder the MatThickActList according to the listbox defined order
        
        # reorderedlist=list(self.listbox.get(0,tk.END))
        reorderedlist=[self.list_widget.item(x).text() for x in range(self.list_widget.count())]
        # print(reorderedlist)
        
        numreorderedlist=[]
        for item in reorderedlist:
            numreorderedlist.append(int(item.split('_')[0]))
        
        MatThickActList = [MatThickActList[i] for i in numreorderedlist]
        window.populate()
        self.hide()
        

#%%#######################

class TMSimulation(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        global MatThickActList, numberofLayer, reordered
        self.ui = Ui_TransferMatrixModeling()
        self.ui.setupUi(self)
        
        finish = QAction("Quit", self)
        finish.triggered.connect(lambda: self.closeEvent(0))
        
        self.ui.pushButton_LoadStack.clicked.connect(self.loadstack)
        self.ui.pushButton_SaveStack.clicked.connect(self.savestack)
        self.ui.pushButton_AddLayer.clicked.connect(self.AddLayer)
        self.ui.pushButton_ReorderStack.clicked.connect(self.reorder)
        self.ui.pushButton_DeleteLayer.clicked.connect(self.DeleteLayer)
        self.ui.pushButton_CheckNK.clicked.connect(self.Nkplotstartwin)
        self.ui.pushButton_StartSimul.clicked.connect(self.simulate)
        self.ui.pushButton_Help.clicked.connect(self.Helpcall)
        
        self.populate()
        
    def closeEvent(self, event):
        """ what happens when close the program"""
        
        close = QMessageBox.question(self,
                                     "QUIT",
                                     "Are you sure?",
                                      QMessageBox.Yes | QMessageBox.No)
        if close == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
            
    def reorder(self):
        self.w=reorderwin()
        self.w.show()
    
    def Nkplotstartwin(self):
        self.w = Nkplotwin()
        self.w.show()
        # self.hide()
        
    def Helpcall(self):
        self.w = Help()
        self.w.show()
        
    def populate(self):
        global numberofLayer
        global matnamelist
        global MatThickActList
       
        self.clearLayout(self.ui.gridLayout_2)
        self.ui.scrollArea_stack = QtWidgets.QScrollArea(self.ui.frame_2)
        self.ui.scrollArea_stack.setWidgetResizable(True)
        self.ui.scrollArea_stack.setObjectName("scrollArea_stack")
        self.ui.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.ui.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 500, 400))
        self.ui.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.ui.verticalLayout = QtWidgets.QVBoxLayout(self.ui.scrollAreaWidgetContents)
        self.ui.verticalLayout.setObjectName("verticalLayout")
       
        # numberofLayer=2
        for item in range(numberofLayer):
            self.frame = QtWidgets.QFrame(self.ui.scrollAreaWidgetContents)
            self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
            self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
            self.frame.setObjectName("frame")
            self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame)
            self.horizontalLayout.setObjectName("horizontalLayout")
            
            self.checkBox_layernumb = QtWidgets.QCheckBox(self.frame)
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(self.checkBox_layernumb.sizePolicy().hasHeightForWidth())
            self.checkBox_layernumb.setSizePolicy(sizePolicy)
            self.checkBox_layernumb.setObjectName("checkBox_layernumb"+str(item+1))
            self.checkBox_layernumb.setText(str(item+1))
            self.checkBox_layernumb.setChecked(MatThickActList[item][0])
            self.horizontalLayout.addWidget(self.checkBox_layernumb)
            self.checkBox_layernumb.clicked.connect(self.updateMatThickActList)
            
            #the material of the layer
            self.comboBox_matname = QtWidgets.QComboBox(self.frame)
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(self.comboBox_matname.sizePolicy().hasHeightForWidth())
            self.comboBox_matname.setObjectName("comboBox_matname"+str(item+1))
            self.comboBox_matname.addItems(matnamelist)
            self.comboBox_matname.setCurrentText(MatThickActList[item][1])
            self.horizontalLayout.addWidget(self.comboBox_matname)
            self.comboBox_matname.currentTextChanged.connect(self.updateMatThickActList)
            
            self.spinBox_thickness = QtWidgets.QSpinBox(self.frame)
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(self.spinBox_thickness.sizePolicy().hasHeightForWidth())
            self.spinBox_thickness.setSizePolicy(sizePolicy)
            self.spinBox_thickness.setMaximum(9999999)
            self.spinBox_thickness.setObjectName("spinBox_thickness"+str(item+1))
            self.spinBox_thickness.setValue(MatThickActList[item][2])
            self.horizontalLayout.addWidget(self.spinBox_thickness)
            
            self.checkBox_activelayer = QtWidgets.QCheckBox(self.frame)
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(self.checkBox_activelayer.sizePolicy().hasHeightForWidth())
            self.checkBox_activelayer.setSizePolicy(sizePolicy)
            self.checkBox_activelayer.setObjectName("checkBox_activelayer"+str(item+1))
            self.checkBox_activelayer.setText("active?")
            self.checkBox_activelayer.setChecked(MatThickActList[item][3])
            self.horizontalLayout.addWidget(self.checkBox_activelayer)
            self.checkBox_activelayer.clicked.connect(self.updateMatThickActList)
            
            self.checkBox_incoherent = QtWidgets.QCheckBox(self.frame)
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(self.checkBox_incoherent.sizePolicy().hasHeightForWidth())
            self.checkBox_incoherent.setSizePolicy(sizePolicy)
            self.checkBox_incoherent.setObjectName("checkBox_incoherent"+str(item+1))
            self.checkBox_incoherent.setText("incoherent?")
            self.checkBox_incoherent.setChecked(MatThickActList[item][4])
            self.horizontalLayout.addWidget(self.checkBox_incoherent)
            self.checkBox_incoherent.clicked.connect(self.updateMatThickActList)
            
            self.ui.verticalLayout.addWidget(self.frame)
        
        self.ui.scrollArea_stack.setWidget(self.ui.scrollAreaWidgetContents)
        self.ui.gridLayout_2.addWidget(self.ui.scrollArea_stack, 0, 0, 1, 1)   
    
    def updateMatThickActList(self):
        global numberofLayer
        global matnamelist
        global MatThickActList
        
        MatThickActList=[]
        for i in range(numberofLayer):
            MatThickActList.append([self.findChild(QtWidgets.QCheckBox,"checkBox_layernumb"+str(i+1)).isChecked()
                                    ,self.findChild(QtWidgets.QComboBox,"comboBox_matname"+str(i+1)).currentText()
                                    ,self.findChild(QtWidgets.QSpinBox,"spinBox_thickness"+str(i+1)).value()
                                    ,self.findChild(QtWidgets.QCheckBox,"checkBox_activelayer"+str(i+1)).isChecked()
                                    ,self.findChild(QtWidgets.QCheckBox,"checkBox_incoherent"+str(i+1)).isChecked()])
        
    def clearLayout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clearLayout(item.layout())
                
    def AddLayer(self):
        global numberofLayer
        global MatThickActList
        
        numberofLayer += 1
        
        MatThickActList.append([False,'Air',0,True,True]) #material,thickness,active,conherence (coherent=0;incoherent=1=checked)
        
        self.populate()
    
    def DeleteLayer(self):
        global numberofLayer
        global MatThickActList
        
        # print('')
        # print('delete')
        # print(MatThickActList)
        
        newmatlist=[]
        for item in MatThickActList:
            if not item[0]:
                print(item[0])
                newmatlist.append(item)
        MatThickActList=newmatlist
        numberofLayer -= 1
        self.populate()
        
    def loadstack(self):
        global MatThickActList
        global numberofLayer
        
        directory=os.path.join(str(Path(os.path.abspath(__file__)).parent.parent),'TMstacks')

        path = QFileDialog.getOpenFileName(caption = 'Select stack file', directory = directory)
        
        if path!='':
            try:
                with open(path[0],'r') as file:
                    MatThickActList=[]
                    for line in file:
                        MatThickActList.append([False,
                                                line[:-1].split("\t")[0],
                                                int(line[:-1].split("\t")[1]),
                                                int(line[:-1].split("\t")[2]),
                                                int(line[:-1].split("\t")[3])])
                numberofLayer=len(MatThickActList)
                self.populate()
            except:
                QMessageBox.information(self,'Import failed', "Might not be the correct file?!...") 
    
    def savestack(self):
        global MatThickActList
        global stackNameList
                
        try:
            directory=os.path.join(str(Path(os.path.abspath(__file__)).parent.parent),'TMstacks')
        
            path = QFileDialog.getSaveFileName(caption = 'Select where to save the data', directory = directory)
            
            with open(path[0],'w') as file:
                text=''
                for item in MatThickActList:
                    text+=item[1]+'\t'+str(item[2])+'\t'+str(item[3])+'\t'+str(item[4])+'\n'
                file.write(text[:-1])
            
            stacklist=os.listdir(stackDir)
            stackNameList=[]
            for item in stacklist:
                stackNameList.append(item.split('.')[0])
        except:
            QMessageBox.information(self,'Exception', "something wrong during saving process...") 
        self.populate()
    
    def simulate(self):
        if self.ui.checkBox_1D.isChecked() and self.ui.checkBox_2D.isChecked():
            QMessageBox.information(self,'Exception', "cannot make both 1D and 2D at the same time")
        else:
            self.simulate1()
            
    def simulate1(self):
        global numberofLayer
        global matnamelist
        global MatThickActList
        
        
        #ask path to export
        directory=os.path.join(str(Path(os.path.abspath(__file__)).parent.parent),'results')
        
        path = QFileDialog.getSaveFileName(self, caption = 'Select where to save the data', directory = directory,filter = "Images (*.png)")
        f=str(path[0])
        
        #check the wavelength range and readapt to have only in decades (round to nearest decade)
        start=divmod(self.ui.spinBox_StartWave.value(),10)
        startWave=int(self.ui.spinBox_StartWave.value())
        if start[1]!=0:
            if start[1]>5:
                startWave=int((start[0]+1)*10)
            else:
                startWave=int(start[0]*10)
        end=divmod(self.ui.spinBox_EndWave.value(),10)
        EndWave=int(self.ui.spinBox_EndWave.value())
        if end[1]!=0:
            if end[1]>5:
                EndWave=int((end[0]+1)*10)
            else:
                EndWave=int(end[0]*10)

        #create structure and fill with user-defined layer properties
       
        if MatThickActList!=[]:
            if MatThickActList[0][4]:
                coher='i'
            else:
                coher='c'
            structure=multilayer(layer(material_list[MatThickActList[0][1]],MatThickActList[0][2],coher))
            if len(MatThickActList)>1:
                for i in range(1,len(MatThickActList)):
                    if MatThickActList[i][4]:
                        coher='i'
                    else:
                        coher='c'
                    structure.addlayer(layer(material_list[MatThickActList[i][1]],MatThickActList[i][2],coher))
        
        #create spectrum according to start and end Wavelength
        specttotake=[[],[]]
        for item in range(len(AM1p5Gc[0])):
            if AM1p5Gc[0][item] >= startWave and AM1p5Gc[0][item] <= EndWave:
                specttotake[0].append(AM1p5Gc[0][item])
                specttotake[1].append(AM1p5Gc[1][item])
        
        #check which layer is active and calculate current, print
        huss=structure.geninalllays(specttotake,1,0,1,'s')
        numbofactive=0
        spectofactivelayers=[]
        namesofactive=[]
        numbofnonactive=0
        spectofnonactivelayers=[]
        namesofnonactive=[]
        specR=structure.calculateRTrange(specttotake[0],1,0,1,'s')
        currents=[]
        currentsnon=[]
        totalcurrent=0
        for i in range(len(MatThickActList)):
            if MatThickActList[i][3]:
                numbofactive+=1
                spectofactivelayers.append(structure.absspectruminlay(i+1,specttotake[0],1,0,1,'s'))
                Jsc=str(MatThickActList[i][1])+': '+"%.2f"%huss[i+1]+'\n'
                totalcurrent+=huss[i+1]
                currents.append(Jsc)
                namesofactive.append(MatThickActList[i][1])
                print(Jsc)
            else:
                numbofnonactive+=1
                spectofnonactivelayers.append(structure.absspectruminlay(i+1,specttotake[0],1,0,1,'s'))
                Jsc=str(MatThickActList[i][1])+': '+"%.2f"%huss[i+1]+'\n'
                currentsnon.append(Jsc)
                namesofnonactive.append(MatThickActList[i][1])
#                print(Jsc)
        if spectofactivelayers!=[]:
            spectotal=[specR[0],np.asarray(spectofactivelayers[0][1])]
            
            for i in range(1,len(spectofactivelayers)):   
                spectotal[1]+=np.asarray(spectofactivelayers[i][1])

            #make graph and export

            ############ the figures #################
            self.fig = plt.figure(figsize=(6, 4))
            self.fig.patch.set_facecolor('white')
            self.fig1 = self.fig.add_subplot(111)   
            
            [x,y]=structure.plotnprofile(400)
            self.fig1.plot(x,y,label='400nm')
            [x,y]=structure.plotnprofile(500)
            self.fig1.plot(x,y,label='500nm')
            [x,y]=structure.plotnprofile(600)
            self.fig1.plot(x,y,label='600nm')
            [x,y]=structure.plotnprofile(700)
            self.fig1.plot(x,y,label='700nm')
            [x,y]=structure.plotnprofile(800)
            self.fig1.plot(x,y,label='800nm')
            [x,y]=structure.plotnprofile(900)
            self.fig1.plot(x,y,label='900nm')
            self.fig1.set_xlabel("Distance in device (nm)")
            self.fig1.set_ylabel("Refractive index")
            self.fig1.legend(ncol=1)
            self.fig.savefig(f[:-4]+'_n.png', dpi=300, transparent=False) 
            
            self.fig2 = plt.figure(figsize=(6, 4))
            self.fig2.patch.set_facecolor('white')
            self.fig21 = self.fig2.add_subplot(111)
            datatoexport=[]
            headoffile1=""
            headoffile2=""
            headoffile3=""
#            plt.figure()
            k=0
            for item in spectofactivelayers:
                self.fig21.plot(item[0],item[1],label=currents[k][:-1])
                
                datatoexport.append(item[0])
                datatoexport.append(item[1])
                headoffile1+="Wavelength\tIntensity\t"
                headoffile2+="(nm)\t(-)\t"
                headoffile3+=" \t"+namesofactive[k]+"\t"
                k+=1
            if len(spectofactivelayers)>1:
                self.fig21.plot(spectotal[0],spectotal[1],label="Total: "+"%.2f"%totalcurrent)
                datatoexport.append(spectotal[0])
                datatoexport.append(spectotal[1])
                headoffile1+="Wavelength\tIntensity\t"
                headoffile2+="(nm)\t(-)\t"
                headoffile3+=" \tTotal\t"

            specRR=[specR[0],1-np.asarray(specR[1])-np.asarray(specR[2])]
            datatoexport.append(specRR[0])
            datatoexport.append(specRR[1])
            headoffile1+="Wavelength\tIntensity\t"
            headoffile2+="(nm)\t(-)\t"
            headoffile3+=" \tTotal absorptance\t"
            specRR=[specR[0],np.asarray(specR[2])]
            datatoexport.append(specRR[0])
            datatoexport.append(specRR[1])
            headoffile1+="Wavelength\tIntensity\t"
            headoffile2+="(nm)\t(-)\t"
            headoffile3+=" \tTransmittance\t"
            specRR=[specR[0],1-np.asarray(specR[1])]#-np.asarray(specR[2])]
            datatoexport.append(specRR[0])
            datatoexport.append(specRR[1])
            headoffile1+="Wavelength\tIntensity\t"
            headoffile2+="(nm)\t(-)\t"
            headoffile3+=" \t1-Reflectance\t"
            self.fig21.plot(specRR[0],specRR[1],label="1-Reflectance: "+"%.2f"%calcCurrent(specR[0],specR[1],specRR[0][0],specRR[0][-1]))
            self.fig21.set_xlabel("Wavelength (nm)")
            self.fig21.set_ylabel("Light Intensity Fraction")
            self.fig21.set_xlim([specRR[0][0],specRR[0][-1]])
            self.fig21.set_ylim([0,1])
            self.fig21.legend(ncol=1)#loc='lower right',
            self.fig2.savefig(f, dpi=300, transparent=False) 
            
            #with absorption spectra of non active layers
            k=0
            if len(spectofnonactivelayers)>0:
                for item in spectofnonactivelayers:
                    # self.fig1.plot(item[0],item[1],label=currentsnon[k][:-1])
                    datatoexport.append(item[0])
                    datatoexport.append(item[1])
                    headoffile1+="Wavelength\tIntensity\t"
                    headoffile2+="(nm)\t(-)\t"
                    headoffile3+=" \t"+namesofnonactive[k]+"\t"
                    k+=1
#                self.fig1.set_xlabel("Wavelength (nm)")
#                self.fig1.set_ylabel("Light Intensity Fraction")
#                self.fig1.set_xlim([specRR[0][0],specRR[0][-1]])
#                self.fig1.set_ylim([0,1])
#                self.fig1.legend(ncol=1)#loc='lower right',
#                self.fig.savefig(f[:-4]+'_withParasitic.png', dpi=300, transparent=False) 
            
            headoffile1=headoffile1[:-1]+'\n'
            headoffile2=headoffile2[:-1]+'\n'
            headoffile3=headoffile3[:-1]+'\n'
            
            # self.fig.clear()
            # self.fig1 = self.fig.add_subplot(111)
            self.fig3 = plt.figure(figsize=(6, 4))
            self.fig3.patch.set_facecolor('white')
            self.fig31 = self.fig3.add_subplot(111)
            
            specRR=[specR[0],1-np.asarray(specR[1])-np.asarray(specR[2])]
            self.fig31.plot(specRR[0],specRR[1],label="Absorptance of full stack: "+"%.2f"%calcCurrent(specRR[0],specRR[1],specRR[0][0],specRR[0][-1]))
            specRR=[specR[0],1-np.asarray(specR[1])]
            Rloss=calcCurrent(specR[0],specR[1],specRR[0][0],specRR[0][-1])
            self.fig31.plot(specRR[0],specRR[1],label="1-Reflectance: "+"%.2f"%Rloss)
            specRR=[specR[0],np.asarray(specR[2])]
            Tloss=calcCurrent(specR[0],specR[2],specRR[0][0],specRR[0][-1])
            self.fig31.plot(specRR[0],specRR[1],label="Transmittance: "+"%.2f"%Tloss)
            self.fig31.set_xlabel("Wavelength (nm)")
            self.fig31.set_ylabel("Light Intensity Fraction")
            self.fig31.set_xlim([specRR[0][0],specRR[0][-1]])
            self.fig31.set_ylim([0,1])
            self.fig31.legend(ncol=1)#loc='lower right',
            self.fig3.savefig(f[:-4]+'_ART.png', dpi=300, transparent=False) 
            
            # self.fig.clear()
            # self.fig1 = self.fig.add_subplot(111) 
            self.fig4 = plt.figure(figsize=(6, 4))
            self.fig4.patch.set_facecolor('white')
            self.fig41 = self.fig4.add_subplot(111)
            
            spectotalparas=[specR[0],np.asarray(spectofactivelayers[0][1])]
            names=['']
            for i in range(1,len(spectofactivelayers)):   
                spectotalparas[1]+=np.asarray(spectofactivelayers[i][1])
            for i in range(len(spectofnonactivelayers)):   
                names.append(currentsnon[i][:-1])
                spectotalparas.append(spectotalparas[-1]+np.asarray(spectofnonactivelayers[i][1]))
            names.append('Reflectance: '+"%.2f"%Rloss)
            spectotalparas.append(spectotalparas[-1]+np.asarray(specR[1]))
            names.append('Transmittance: '+"%.2f"%Tloss)
            spectotalparas.append(spectotalparas[-1]+np.asarray(specR[2]))
            
            for i in range(1,len(spectotalparas)):
                self.fig41.plot(spectotalparas[0],spectotalparas[i],label=names[i-1],color=colorstylelist[i-1],linewidth=1)

            for i in range(1,len(spectotalparas)-1):
                self.fig41.fill_between(spectotalparas[0],spectotalparas[i],spectotalparas[i+1],facecolor=colorstylelist[i])
            
            self.fig41.set_xlim([specRR[0][0],specRR[0][-1]])
            self.fig41.set_xlabel("Wavelength (nm)")
            self.fig41.set_ylabel("Light Intensity Fraction")
            self.fig41.set_ylim([0,1])
            self.fig41.legend(ncol=1)#loc='lower right',
            self.fig4.savefig(f[:-4]+'_withParasitic2.png', dpi=300, transparent=False) 
            
            datatoexportINV=[list(x) for x in zip(*datatoexport)]
            datatoexportINVtxt=[]
            for item in datatoexportINV:
                lineinterm=""
                for i in range(len(item)-1):
                    lineinterm+=str(item[i])+"\t"
                lineinterm+=str(item[len(item)-1])+"\n"
                datatoexportINVtxt.append(lineinterm)
            
            file = open(f[:-4]+"_rawdata.txt",'w', encoding='ISO-8859-1')
            file.writelines(headoffile1)
            file.writelines(headoffile2)
            file.writelines(headoffile3)
            file.writelines(item for item in datatoexportINVtxt)
            file.close()
            # plt.close("all")
            
            file = open(f[:-4]+"_layerstack.txt",'w', encoding='ISO-8859-1')
            file.writelines(item[1]+'\t'+str(item[2])+'\t'+str(item[3])+'\t'+str(item[4])+'\n' for item in MatThickActList)
            file.close()
            
        
        if self.ui.checkBox_1D.isChecked():
            w=simul1D(structure,specttotake, f)
            w.show()
            
        elif self.ui.checkBox_2D.isChecked():
            w=simul2D(structure,specttotake, f)
            w.show()
            


class simul1D(QtWidgets.QDialog):
    def __init__(self, structure, specttotake, f):
        super().__init__()
        global numberofLayer
        global matnamelist
        global MatThickActList
        
        self.resize(350, 300)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QtCore.QSize(350, 200))
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.setWindowTitle("Define the variation ranges")
        
        self.matlistfor2Dvar=[MatThickActList[item][1]+'#'+str(item+1) for item in range(len(MatThickActList))]
        
        self.label = QtWidgets.QLabel()
        self.label.setText('from')
        self.gridLayout.addWidget(self.label)
        
        self.Startthick1 = QtWidgets.QLineEdit()
        self.Startthick1.setText(str(0))
        self.gridLayout.addWidget(self.Startthick1)
        
        self.label = QtWidgets.QLabel()
        self.label.setText('step')
        self.gridLayout.addWidget(self.label)
        
        self.Stepthick1 = QtWidgets.QLineEdit()
        self.Stepthick1.setText(str(1))
        self.gridLayout.addWidget(self.Stepthick1)
        
        self.label = QtWidgets.QLabel()
        self.label.setText('to')
        self.gridLayout.addWidget(self.label)
        
        self.Endthick1 = QtWidgets.QLineEdit()
        self.Endthick1.setText(str(10))
        self.gridLayout.addWidget(self.Endthick1)
        
        self.comboBox_matname1D = QtWidgets.QComboBox()
        self.comboBox_matname1D.addItems(self.matlistfor2Dvar)
        self.gridLayout.addWidget(self.comboBox_matname1D)
        
        self.pushButton_1D = QtWidgets.QPushButton("Validate", self)
        self.gridLayout.addWidget(self.pushButton_1D)
        self.pushButton_1D.clicked.connect(lambda: self.validate(structure,specttotake,f))
        
    def validate(self,structure,specttotake,f):
        global numberofLayer
        global matnamelist
        global MatThickActList
        
        
        d=structure.scanlayerthickness(self.matlistfor2Dvar.index(self.comboBox_matname1D.currentText())+1,range(int(self.Startthick1.text()),int(self.Endthick1.text())+1,int(self.Stepthick1.text())),specttotake,1,0,1,'s')
        dinv=[list(x) for x in zip(*d)]
        
        head="Thickness\tAir\t"
        for item in self.matlistfor2Dvar:
            head+=item+"\t"
        head+="Air\n"
        datatoexportINVtxt=[]
        for item in dinv:
            lineinterm=""
            for i in range(len(item)-1):
                lineinterm+=str(item[i])+"\t"
            lineinterm+=str(item[len(item)-1])+"\n"
            datatoexportINVtxt.append(lineinterm)
        
        file = open(f[:-4]+"_1Drawdata.txt",'w', encoding='ISO-8859-1')
        file.writelines(head)
        file.writelines(item for item in datatoexportINVtxt)
        file.close()
        
        plt.figure()
        activelist=[]
        for i in range(len(MatThickActList)):
            if MatThickActList[i][3]:
                activelist.append(i)
                plt.plot(d[0],d[i+2],label=MatThickActList[i][1])
        
        plt.xlabel(self.comboBox_matname1D.currentText()+" thickness (nm)")
        plt.ylabel("Current")
        plt.legend(ncol=1)
        plt.savefig(f[:-4]+"_1D.png", dpi=300, transparent=False)
        plt.close()
        
        self.hide()
    
    
class simul2D(QtWidgets.QDialog):
    def __init__(self, structure, specttotake, f):
        super().__init__()
        global numberofLayer
        global matnamelist
        global MatThickActList
        
        self.resize(350, 500)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumSize(QtCore.QSize(350, 200))
        self.gridLayout = QtWidgets.QGridLayout(self)
        self.setWindowTitle("Define the variation ranges")
        
        self.matlistfor2Dvar=[MatThickActList[item][1]+'#'+str(item+1) for item in range(len(MatThickActList))]
        
        self.label = QtWidgets.QLabel()
        self.label.setText('from')
        self.gridLayout.addWidget(self.label)
        
        self.Startthick1 = QtWidgets.QLineEdit()
        self.Startthick1.setText(str(0))
        self.gridLayout.addWidget(self.Startthick1)
        
        self.label = QtWidgets.QLabel()
        self.label.setText('step')
        self.gridLayout.addWidget(self.label)
        
        self.Stepthick1 = QtWidgets.QLineEdit()
        self.Stepthick1.setText(str(1))
        self.gridLayout.addWidget(self.Stepthick1)
        
        self.label = QtWidgets.QLabel()
        self.label.setText('to')
        self.gridLayout.addWidget(self.label)
        
        self.Endthick1 = QtWidgets.QLineEdit()
        self.Endthick1.setText(str(10))
        self.gridLayout.addWidget(self.Endthick1)
        
        self.comboBox_matname2D1 = QtWidgets.QComboBox()
        self.comboBox_matname2D1.addItems(self.matlistfor2Dvar)
        self.gridLayout.addWidget(self.comboBox_matname2D1)
        
        self.label = QtWidgets.QLabel()
        self.label.setText('from')
        self.gridLayout.addWidget(self.label)
        
        self.Startthick2 = QtWidgets.QLineEdit()
        self.Startthick2.setText(str(0))
        self.gridLayout.addWidget(self.Startthick2)
        
        self.label = QtWidgets.QLabel()
        self.label.setText('step')
        self.gridLayout.addWidget(self.label)
        
        self.Stepthick2 = QtWidgets.QLineEdit()
        self.Stepthick2.setText(str(1))
        self.gridLayout.addWidget(self.Stepthick2)
        
        self.label = QtWidgets.QLabel()
        self.label.setText('to')
        self.gridLayout.addWidget(self.label)
        
        self.Endthick2 = QtWidgets.QLineEdit()
        self.Endthick2.setText(str(10))
        self.gridLayout.addWidget(self.Endthick2)
        
        self.comboBox_matname2D2 = QtWidgets.QComboBox()
        self.comboBox_matname2D2.addItems(self.matlistfor2Dvar)
        self.gridLayout.addWidget(self.comboBox_matname2D2)
        
        self.pushButton_2D = QtWidgets.QPushButton("Validate", self)
        self.gridLayout.addWidget(self.pushButton_2D)
        self.pushButton_2D.clicked.connect(lambda: self.validate(structure,specttotake,f))
        
    def validate(self,structure,specttotake,f):
        global numberofLayer
        global matnamelist
        global MatThickActList
        
        d=structure.scanlayerthickness2D(self.matlistfor2Dvar.index(self.comboBox_matname2D1.currentText())+1,self.matlistfor2Dvar.index(self.comboBox_matname2D2.currentText())+1,range(int(self.Startthick1.text()),int(self.Endthick1.text())+1,int(self.Stepthick1.text())),range(int(self.Startthick2.text()),int(self.Endthick2.text())+1,int(self.Stepthick2.text())),specttotake,1,0,1,'s')
        datatoexporttxt=[]
        
        for i in range(len(d)):
            for j in range(len(d[i][1])):
                linetxt=str(d[i][0])
                for k in range(len(d[i][1][j])):
                    linetxt+="\t"+str(d[i][1][j][k])
                linetxt+="\n"
                datatoexporttxt.append(linetxt)
                
        head="Thick. "+self.comboBox_matname2D1.currentText()+"\tThick. "+self.comboBox_matname2D2.currentText()+"\tAir\t"
        for item in self.matlistfor2Dvar:
            head+=item+"\t"
        head+="Air\n"
        file = open(f[:-4]+"_2Drawdata.txt",'w', encoding='ISO-8859-1')
        file.writelines(head)
        file.writelines(item for item in datatoexporttxt)
        file.close()
        
        self.hide()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = TMSimulation()
    window.show()
    sys.exit(app.exec())      
        
