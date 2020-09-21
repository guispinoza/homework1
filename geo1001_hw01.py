#-- GEO1001.2020--hw01
#-- [Guilherme Spinoza Andreo] 
#-- [5383994]

import numpy as np																	
import matplotlib.pyplot as plt														
import pandas as pd																	
import analytic 
import thinkstats2
import thinkplot
import scipy.stats as stats
import seaborn as sns
import matplotlib.patches
import scipy
from scipy.stats import pearsonr
import csv

### Manage the excel database
locations = ["HEAT - A_final.xls","HEAT - B_final.xls","HEAT - C_final.xls","HEAT - D_final.xls", "HEAT - E_final.xls"]
df_A = pd.read_excel("HEAT - A_final.xls", header = 3, skiprows = range(4,5))
df_B = pd.read_excel("HEAT - B_final.xls", header = 3, skiprows = range(4,5))
df_C = pd.read_excel("HEAT - C_final.xls", header = 3, skiprows = range(4,5))
df_D = pd.read_excel("HEAT - D_final.xls", header = 3, skiprows = range(4,5))
df_E = pd.read_excel("HEAT - E_final.xls", header = 3, skiprows = range(4,5))





Temperature = (df_A["Temperature"], df_B["Temperature"], df_C["Temperature"], df_D["Temperature"], df_E["Temperature"])
TrueDirection = (df_A["Direction ‚ True"], df_B["Direction ‚ True"], df_C["Direction ‚ True"], df_D["Direction ‚ True"], df_E["Direction ‚ True"])
WindSpeed = (df_A["Wind Speed"], df_B["Wind Speed"], df_C["Wind Speed"], df_D["Wind Speed"], df_E["Wind Speed"])
Wetbulb = (df_A["NA Wet Bulb Temperature"], df_B["NA Wet Bulb Temperature"], df_C["NA Wet Bulb Temperature"], df_D["NA Wet Bulb Temperature"], df_E["NA Wet Bulb Temperature"])
CrossWind = (df_A["Crosswind Speed"], df_B["Crosswind Speed"], df_C["Crosswind Speed"], df_D["Crosswind Speed"], df_E["Crosswind Speed"])
WBGT = (df_A["WBGT"], df_B["WBGT"], df_C["WBGT"], df_D["WBGT"], df_E["WBGT"])
### Classifying the information

TEMP_A=df_A["Temperature"]
TEMP_B=df_B["Temperature"]
TEMP_C=df_C["Temperature"]
TEMP_D=df_D["Temperature"]
TEMP_E=df_E["Temperature"]

DIR_A=df_A["Direction ‚ True"]
DIR_B=df_B["Direction ‚ True"]
DIR_C=df_C["Direction ‚ True"]
DIR_D=df_D["Direction ‚ True"]
DIR_E=df_E["Direction ‚ True"]

WINDSP_A=df_A["Wind Speed"]
WINDSP_B=df_B["Wind Speed"]
WINDSP_C=df_C["Wind Speed"]
WINDSP_D=df_D["Wind Speed"]
WINDSP_E=df_E["Wind Speed"]

WBGT_A=df_A["WBGT"] 
WBGT_B=df_B["WBGT"] 
WBGT_C=df_C["WBGT"] 
WBGT_D=df_D["WBGT"] 
WBGT_E=df_E["WBGT"] 

CROSS_A = df_A["Crosswind Speed"]   
CROSS_B = df_B["Crosswind Speed"]  
CROSS_C = df_C["Crosswind Speed"]  
CROSS_D = df_D["Crosswind Speed"]  
CROSS_E = df_E["Crosswind Speed"]

### Compute mean statstics

# I wanted to clean this data up but I didn't have enough time

Mean = (df_A.mean(), df_B.mean(), df_C.mean(),df_D.mean(), df_E.mean())
Variance = (df_A.var(), df_B.var(), df_C.var(),df_D.var(), df_E.var())
Standard_Deviation = (df_A.std(), df_B.std(), df_C.std(),df_D.std(), df_E.std())
print(Mean, Variance, Standard_Deviation)



### Create 1 histogram for the 5 sensor temperature values, 50 and 5 bins
def hist_temp(a,b,c,d):
    
    fig, ax = plt.subplots(1,2)
    plt.subplots_adjust(wspace = 0.6)
    ax[0].set_ylabel("Frequency")
    ax[1].set_ylabel("Frequency")
    ax[0].set_xlabel("Temperature °C")
    ax[1].set_xlabel("Temperature °C")
    a.hist(ax=ax[0],bins=b)
    a.hist(ax=ax[1],bins=c)
    
    plt.suptitle('Temperatures of Sensor '+d)
    plt.show()
hist_temp(TEMP_A.astype(float),5,50,'A')
hist_temp(TEMP_B.astype(float),5,50,'B')
hist_temp(TEMP_C.astype(float),5,50,'C')
hist_temp(TEMP_D.astype(float),5,50,'D')
hist_temp(TEMP_E.astype(float),5,50,'E')

## Create 1 plot where frequency polygons for the 5 sensors values overlap with a legend
def hist_temp_sens(a,b,c,d,e):
    
    fig = plt.figure(figsize=(16,10))
    ax_a = fig.add_subplot(111)
    ax_b = fig.add_subplot(111)
    ax_c = fig.add_subplot(111)
    ax_d = fig.add_subplot(111)
    ax_e = fig.add_subplot(111)
    
    [fr_a,bins]=np.histogram(a,bins=27)
    [fr_b,bins]=np.histogram(b,bins=27)
    [fr_c,bins]=np.histogram(c,bins=27)
    [fr_d,bins]=np.histogram(d,bins=27)
    [fr_e,bins]=np.histogram(e,bins=27)

    a_cdf=np.cumsum(fr_a)
    b_cdf=np.cumsum(fr_b)
    c_cdf=np.cumsum(fr_c)
    d_cdf=np.cumsum(fr_d)
    e_cdf=np.cumsum(fr_e)

    #Create 1 plot where frequency poligons for the 5 sensors Temperature values overlap in different colors with a legend.
    
    ax_a.plot(bins[:-1],a_cdf,label= "Sensor A")
    ax_b.plot(bins[:-1],b_cdf,label= "Sensor B")
    ax_c.plot(bins[:-1],c_cdf,label= "Sensor C")
    ax_d.plot(bins[:-1],d_cdf,label= "Sensor D")
    ax_e.plot(bins[:-1],e_cdf,label= "Sensor E")
   
    plt.grid(axis='y', alpha=0.75)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Temperature °C', size=16)
    plt.ylabel('Frequency', size=16)
    plt.title("Sensors's frequency polygons", size=18)
    plt.legend(prop={"size": 14}, title="legend")
    plt.show()

hist_temp_sens(TEMP_A.astype(float),TEMP_B.astype(float),TEMP_C.astype(float),TEMP_D.astype(float),TEMP_E.astype(float))

### Generate 3 boxplots that include the 5 sensors for: Wind Speed, Wind Direction and Temperature.

plt.boxplot(Temperature)
plt.ylabel("Temperature in °C")
plt.xlabel("Sensors")
plt.title("Temperature")
plt.show()

plt.boxplot(WindSpeed)
plt.ylabel("m/s")
plt.xlabel("Sensors")
plt.title("Windspeed")
plt.show()

plt.boxplot(TrueDirection)
plt.ylabel("Wind Direction in Deg")
plt.xlabel("Sensors")    
plt.title("Wind Direction")
plt.show()

################### A2 ###############################

# Plot PMF, PDF and CDF for the 5 sensors Temperature values. 
# Describe the behaviour of the distributions, are they all similar? what about their tails?

def pmf(sample1, sample2, sample3, sample4, sample5):
    c1 = sample1.value_counts()
    p1 = c1/len(sample1)
    
    c2 = sample2.value_counts()
    p2 = c2/len(sample2)
    
    c3 = sample3.value_counts()
    p3 = c3/len(sample3)

    c4 = sample4.value_counts()
    p4 = c4/len(sample4)

    c5 = sample5.value_counts()
    p5 = c5/len(sample5)
    
    df1 = p1
    df2 = p2
    df3 = p3
    df4 = p4
    df5 = p5
    
 
    ###PMF
    c1 = df1.sort_index()
    fig = plt.figure(figsize=(17,8))
    plt.subplots_adjust(wspace = 0.2, hspace = 0.5)
    ax1 = fig.add_subplot(321)
    ax1.bar(c1.index,c1, width=0.3)
    ax1.set_ylabel("Probability")
    ax1.set_xlabel("Temperature in °C")
    plt.title('PMF, Sensor A')
    
    c2 = df2.sort_index()
    ax2 = fig.add_subplot(322)
    ax2.bar(c2.index,c2, width=0.3)
    ax2.set_ylabel("Probability")
    ax2.set_xlabel("Temperature in °C")
    plt.title('PMF, Sensor B')
    
    c3 = df3.sort_index()
    ax3 = fig.add_subplot(323)
    ax3.bar(c3.index,c3, width=0.3)
    ax3.set_ylabel("Probability")
    ax3.set_xlabel("Temperature in °C")
    plt.title('PMF, Sensor C')
    
    c4 = df4.sort_index()
    ax4 = fig.add_subplot(324)
    ax4.bar(c4.index,c4, width=0.3)
    ax4.set_ylabel("Probability")
    ax4.set_xlabel("Temperature in °C")
    plt.title('PMF, Sensor D')
    
    c5 = df5.sort_index()
    ax5 = fig.add_subplot(325)
    ax5.bar(c5.index,c5, width=0.3)
    ax5.set_ylabel("Probability")
    ax5.set_xlabel("Temperature in °C")
    plt.title('PMF, Sensor E')

    plt.show()

pmf(df_A["Temperature"].astype(float),df_B["Temperature"].astype(float),df_C["Temperature"].astype(float), df_D["Temperature"].astype(float), df_E["Temperature"].astype(float))

def pdf(sample1, sample2, sample3, sample4, sample5):
    nb = 27

    fig = plt.figure(figsize=(17,8))
    plt.subplots_adjust(wspace = 0.2, hspace = 0.5)
    
    ax1 = fig.add_subplot(321)
    ax1.hist(x=sample1.astype(float),bins=nb, density=True, color='b',alpha=0.7, rwidth=0.85)
    sns.distplot(sample1.astype(float), color='k',ax=ax1, hist = False)
    ax1.set_ylabel("Probability Density")
    ax1.set_xlabel("Temperature in °C")
    plt.title('PDF, Sensor A')

    ax2 = fig.add_subplot(322)
    ax2.hist(x=sample2.astype(float),bins=nb, density=True, color='b',alpha=0.7, rwidth=0.85)
    sns.distplot(sample2.astype(float), color='k',ax=ax2, hist = False)
    ax2.set_ylabel("Probability Density")
    ax2.set_xlabel("Temperature in °C")
    plt.title('PDF, Sensor B')
    
    ax3 = fig.add_subplot(323)
    ax3.hist(x=sample3.astype(float),bins=nb, density=True, color='b',alpha=0.7, rwidth=0.85)
    sns.distplot(sample3.astype(float), color='k',ax=ax3, hist = False)
    ax3.set_ylabel("Probability Density")
    ax3.set_xlabel("Temperature in °C")
    plt.title('PDF, Sensor C')
    
    ax4 = fig.add_subplot(324)
    ax4.hist(x=sample4.astype(float),bins=nb, density=True, color='b',alpha=0.7, rwidth=0.85)
    sns.distplot(sample4.astype(float), color='k',ax=ax4, hist = False)
    ax4.set_ylabel("Probability Density")
    ax4.set_xlabel("Temperature in °C")
    plt.title('PDF, Sensor D')
    
    ax5 = fig.add_subplot(325)
    ax5.hist(x=sample5.astype(float),bins=nb, density=True, color='b',alpha=0.7, rwidth=0.85)
    sns.distplot(sample5.astype(float), color='k',ax=ax5, hist = False)
    ax5.set_ylabel("Probability Density")
    ax5.set_xlabel("Temperature in °C")
    plt.title('PDF, Sensor E')
    
    plt.show()

pdf(df_A["Temperature"].astype(float),df_B["Temperature"].astype(float),df_C["Temperature"].astype(float), df_D["Temperature"].astype(float), df_E["Temperature"].astype(float))
   

def CDF_TEMP(a, b, c, d, e):

    fig = plt.figure(figsize=(20,8))
    plt.subplots_adjust(wspace = 0.3, hspace = 0.5)
    fig.suptitle("CDF Temperature °C")
    
    ax1 = fig.add_subplot(321)
    a1=ax1.hist(x=a.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax1.plot(a1[1][1:]-(a1[1][1:]-a1[1][:-1])/2,a1[0], color='k')
    ax1.set_ylabel("CDF")
    ax1.set_xlabel("Temperature in °C")
    plt.title('CDF, Sensor A')

    ax2 = fig.add_subplot(322)
    a2=ax2.hist(x=b.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax2.plot(a2[1][1:]-(a2[1][1:]-a2[1][:-1])/2,a2[0], color='k')
    ax2.set_ylabel("CDF")
    ax2.set_xlabel("Temperature in °C")
    plt.title('CDF, Sensor B')

    ax3 = fig.add_subplot(323)
    a3=ax3.hist(x=c.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax3.plot(a3[1][1:]-(a3[1][1:]-a3[1][:-1])/2,a3[0], color='k')
    ax3.set_ylabel("CDF")
    ax3.set_xlabel("Temperature in °C")
    plt.title('CDF, Sensor C')
    
    ax4 = fig.add_subplot(324)
    a4=ax4.hist(x=d.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax4.plot(a4[1][1:]-(a4[1][1:]-a4[1][:-1])/2,a4[0], color='k')
    ax4.set_ylabel("CDF")
    ax4.set_xlabel("Temperature in °C")
    plt.title('CDF, Sensor D')

    ax5 = fig.add_subplot(325)
    a5=ax5.hist(x=e.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax5.plot(a5[1][1:]-(a5[1][1:]-a5[1][:-1])/2,a5[0], color='k')
    ax5.set_ylabel("CDF")
    ax5.set_xlabel("Temperature in °C")
    plt.title('CDF, Sensor E')
    plt.show()

CDF_TEMP(df_A["Temperature"].astype(float),df_B["Temperature"].astype(float),df_C["Temperature"].astype(float), df_D["Temperature"].astype(float), df_E["Temperature"].astype(float))

###A2 - 2)	For the Wind Speed values, plot the pdf and the kernel density estimation. Comment on the differences


################ PDF WINDSPEED #####################

def pdf_windspeed(sample1, sample2, sample3, sample4, sample5):
    nb = 27
    fig = plt.figure(figsize=(17,8))
    plt.subplots_adjust(wspace = 0.2, hspace = 0.5)
    
    ax1 = fig.add_subplot(321)
    ax1.hist(x=sample1.astype(float),bins=nb, density=True, color='b',alpha=0.7, rwidth=0.85)
    sns.distplot(sample1.astype(float), color='k',ax=ax1, hist = False)
    ax1.set_ylabel("Probability Density")
    ax1.set_xlabel("Wind Speed (m/s)")
    plt.title('PDF, Sensor A')

    ax2 = fig.add_subplot(322)
    ax2.hist(x=sample2.astype(float),bins=nb, density=True, color='b',alpha=0.7, rwidth=0.85)
    sns.distplot(sample2.astype(float), color='k',ax=ax2, hist = False)
    ax2.set_ylabel("Probability Density")
    ax2.set_xlabel("Wind Speed (m/s)")
    plt.title('PDF, Sensor B')
    
    ax3 = fig.add_subplot(323)
    ax3.hist(x=sample3.astype(float),bins=nb, density=True, color='b',alpha=0.7, rwidth=0.85)
    sns.distplot(sample3.astype(float), color='k',ax=ax3, hist = False)
    ax3.set_ylabel("Probability Density")
    ax3.set_xlabel("Wind Speed (m/s)")
    plt.title('PDF, Sensor C')
    
    ax4 = fig.add_subplot(324)
    ax4.hist(x=sample4.astype(float),bins=nb, density=True, color='b',alpha=0.7, rwidth=0.85)
    sns.distplot(sample4.astype(float), color='k',ax=ax4, hist = False)
    ax4.set_ylabel("Probability Density")
    ax4.set_xlabel("Wind Speed (m/s)")
    plt.title('PDF, Sensor D')
    
    ax5 = fig.add_subplot(325)
    ax5.hist(x=sample5.astype(float),bins=nb, density=True, color='b',alpha=0.7, rwidth=0.85)
    sns.distplot(sample5.astype(float), color='k',ax=ax5, hist = False)
    ax5.set_ylabel("Probability Density")
    ax5.set_xlabel("Wind Speed (m/s)")
    plt.title('PDF, Sensor E')
    
    plt.show()

pdf_windspeed(df_A["Wind Speed"].astype(float),df_B["Wind Speed"].astype(float),df_C["Wind Speed"].astype(float), df_D["Wind Speed"].astype(float), df_E["Wind Speed"].astype(float)) 
     
def KDE(sample1, sample2, sample3, sample4, sample5):
    fig = plt.figure(figsize=(20,8))
    plt.subplots_adjust(wspace = 0.3, hspace = 0.5)
    fig.suptitle("Kernel Density Function")

    ax1 = fig.add_subplot(321)
    sns.distplot(sample1, hist=True, kde=True, bins=27, color = 'darkblue', 
        hist_kws={'edgecolor':'black'},
        kde_kws={'linewidth': 2.5})
    ax1.set_ylabel("KDE Values")
    ax1.set_xlabel("Wind Speed in m/s")
    plt.title('KDE, Sensor A')

    ax2 = fig.add_subplot(322)
    sns.distplot(sample2, hist=True, kde=True, bins=27, color = 'darkblue', 
        hist_kws={'edgecolor':'black'},
        kde_kws={'linewidth': 2.5})
    ax2.set_ylabel("KDE Values")
    ax2.set_xlabel("Wind Speed in m/s")
    plt.title('KDE, Sensor B')

    ax3 = fig.add_subplot(323)
    sns.distplot(sample3, hist=True, kde=True, bins=27, color = 'darkblue', 
        hist_kws={'edgecolor':'black'},
        kde_kws={'linewidth': 2.5})
    ax3.set_ylabel("KDE Values")
    ax3.set_xlabel("Wind Speed in m/s")
    plt.title('KDE, Sensor C')

    ax4 = fig.add_subplot(324)
    sns.distplot(sample4, hist=True, kde=True, bins=27, color = 'darkblue', 
        hist_kws={'edgecolor':'black'},
        kde_kws={'linewidth': 2.5})
    ax4.set_ylabel("KDE Values")
    ax4.set_xlabel("Wind Speed in m/s")
    plt.title('KDE, Sensor D')

    ax5 = fig.add_subplot(325)
    sns.distplot(sample5, hist=True, kde=True, bins=27, color = 'darkblue', 
        hist_kws={'edgecolor':'black'},
        kde_kws={'linewidth': 2.5})
    ax5.set_ylabel("KDE Values")
    ax5.set_xlabel("Wind Speed in m/s")
    plt.title('KDE, Sensor E')


    plt.show()
KDE(df_A["Wind Speed"].astype(float),df_B["Wind Speed"].astype(float),df_C["Wind Speed"].astype(float), df_D["Wind Speed"].astype(float), df_E["Wind Speed"].astype(float))
###################### A3 #####################################

## Compute the correlations between all the sensors for Temperature, WGBT, and Crosswind Speed

def corr(a,b,c,d,e,f):
    
    ab = np.interp(np.linspace(0,len(b),len(b)),np.linspace(0,len(a),len(a)),a)
    ac = np.interp(np.linspace(0,len(c),len(c)),np.linspace(0,len(a),len(a)),a)
    ad = np.interp(np.linspace(0,len(d),len(d)),np.linspace(0,len(a),len(a)),a)
    ae = np.interp(np.linspace(0,len(e),len(e)),np.linspace(0,len(a),len(a)),a)

    bc = np.interp(np.linspace(0,len(c),len(c)),np.linspace(0,len(b),len(b)),b)
    bd = np.interp(np.linspace(0,len(d),len(d)),np.linspace(0,len(b),len(b)),b)
    be = np.interp(np.linspace(0,len(e),len(e)),np.linspace(0,len(b),len(b)),b)

    cd = np.interp(np.linspace(0,len(d),len(d)),np.linspace(0,len(c),len(c)),c)
    ce = np.interp(np.linspace(0,len(e),len(e)),np.linspace(0,len(c),len(c)),c)

    de = np.interp(np.linspace(0,len(e),len(e)),np.linspace(0,len(d),len(d)),d)

    ## INTERPOLATE THE SAMPLES
    norm_ab = (ab - ab.mean())/ab.std()
    norm_ac = (ac - ac.mean())/ac.std()
    norm_ad = (ad - ad.mean())/ad.std()
    norm_ae = (ae - ae.mean())/ae.std()

    norm_bc = (bc - bc.mean())/bc.std()
    norm_bd = (bd - bd.mean())/bd.std()
    norm_be = (be - be.mean())/be.std()

    norm_cd = (cd - cd.mean())/cd.std()
    norm_ce = (ce - ce.mean())/ce.std()

    norm_de = (de - de.mean())/de.std()


    ### NORMALIZE THE SAMPLES BECAUSE THEY HAVE DIFFERENT UNITS

    b_normal= (b-b.mean())/b.std()
    c_normal= (c-b.mean())/c.std()
    d_normal= (d-b.mean())/d.std()
    e_normal= (e-b.mean())/e.std()


    ## USE PEARSON AND SPEARMAN TO COMPUTE STATISTICS
    p=[]
    s=[]

    pcoef_ab = stats.pearsonr(norm_ab,b_normal)[0]
    prcoef_ab = stats.spearmanr(norm_ab,b_normal)[0]
    p.append(pcoef_ab)
    s.append(prcoef_ab)

    pcoef_ac = stats.pearsonr(norm_ac,c_normal)[0]
    prcoef_ac = stats.spearmanr(norm_ac,c_normal)[0]
    p.append(pcoef_ac)
    s.append(prcoef_ac)
   
    pcoef_ad = stats.pearsonr(norm_ad,d_normal)[0]
    prcoef_ad = stats.spearmanr(norm_ad,d_normal)[0]
    p.append(pcoef_ad)
    s.append(prcoef_ad)

    pcoef_ae = stats.pearsonr(norm_ae,e_normal)[0]
    prcoef_ae = stats.spearmanr(norm_ae,e_normal)[0]
    p.append(pcoef_ae)
    s.append(prcoef_ae)

    pcoef_bc = stats.pearsonr(norm_bc,c_normal)[0]
    prcoef_bc = stats.spearmanr(norm_bc,c_normal)[0]
    p.append(pcoef_bc)
    s.append(prcoef_bc)

    pcoef_bd = stats.pearsonr(norm_bd,d_normal)[0]
    prcoef_bd = stats.spearmanr(norm_bd,d_normal)[0]
    p.append(pcoef_bd)
    s.append(prcoef_bd)

    pcoef_be = stats.pearsonr(norm_be,e_normal)[0]
    prcoef_be = stats.spearmanr(norm_be,e_normal)[0]
    p.append(pcoef_be)
    s.append(prcoef_be)

    pcoef_cd = stats.pearsonr(norm_cd,d_normal)[0]
    prcoef_cd = stats.spearmanr(norm_cd,d_normal)[0]
    p.append(pcoef_cd)
    s.append(prcoef_cd)

    pcoef_ce = stats.pearsonr(norm_ce,e_normal)[0]
    prcoef_ce = stats.spearmanr(norm_ce,e_normal)[0]
    p.append(pcoef_ce)
    s.append(prcoef_ce)

    pcoef_de = stats.pearsonr(norm_de,e_normal)[0]
    prcoef_de = stats.spearmanr(norm_de,e_normal)[0]
    p.append(pcoef_de)
    s.append(prcoef_de)

    xlabel=["AB", "AC", "AD", "AE", "BC", "BD", "BE", "CD", "CE", "DE"]
    ### PRODUCE THE SCATTERPLOT FOR PEARSON AND SPEARMAN
    
        
    fig = plt.figure(figsize=(15,8))
    plt.subplots_adjust(wspace = 0.4, hspace = 0.6)
    fig.suptitle("Correlations :" +f, size=18)
    

    ax1 = fig.add_subplot(121)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax1.scatter(xlabel,p)
    ax1.set_xlabel('Sensors combination', size=16)
    ax1.set_ylabel('Pearson Correlations', size=16)

    ax2 = fig.add_subplot(122)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax2.scatter(xlabel,s)
    ax2.set_xlabel('Sensors combination', size=16)
    ax2.set_ylabel('Spearman Correlations', size=16)

    
    plt.show()
    
corr(Temperature[0].astype(float),Temperature[1].astype(float),Temperature[2].astype(float),Temperature[3].astype(float),Temperature[4].astype(float),"Temperature - sensor correlation")
corr(WBGT[0].astype(float),WBGT[1].astype(float),WBGT[2].astype(float),WBGT[3].astype(float),WBGT[4].astype(float),"WBGT - sensor correlation")
corr(CrossWind[0].astype(float),CrossWind[1].astype(float),CrossWind[2].astype(float),CrossWind[3].astype(float),CrossWind[4].astype(float),"Cross Wind Speed - sensor correlation")

######################### A4 ###############################
### Plot the CDF for all the sensors and for variables Temperature and Wind Speed
#CDF

def CDF_TEMP2(a, b, c, d, e):
    fig = plt.figure(figsize=(20,8))
    plt.subplots_adjust(wspace = 0.3, hspace = 0.5)
    fig.suptitle("CDF Temperature °C")
    
    ax1 = fig.add_subplot(321)
    a1=ax1.hist(x=a.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax1.plot(a1[1][1:]-(a1[1][1:]-a1[1][:-1])/2,a1[0], color='k')
    ax1.set_ylabel("CDF")
    ax1.set_xlabel("Temperature in °C")
    plt.title('CDF, Sensor A')

    ax2 = fig.add_subplot(322)
    a2=ax2.hist(x=b.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax2.plot(a2[1][1:]-(a2[1][1:]-a2[1][:-1])/2,a2[0], color='k')
    ax2.set_ylabel("CDF")
    ax2.set_xlabel("Temperature in °C")
    plt.title('CDF, Sensor B')

    ax3 = fig.add_subplot(323)
    a3=ax3.hist(x=c.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax3.plot(a3[1][1:]-(a3[1][1:]-a3[1][:-1])/2,a3[0], color='k')
    ax3.set_ylabel("CDF")
    ax3.set_xlabel("Temperature in °C")
    plt.title('CDF, Sensor C')
    
    ax4 = fig.add_subplot(324)
    a4=ax4.hist(x=d.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax4.plot(a4[1][1:]-(a4[1][1:]-a4[1][:-1])/2,a4[0], color='k')
    ax4.set_ylabel("CDF")
    ax4.set_xlabel("Temperature in °C")
    plt.title('CDF, Sensor D')

    ax5 = fig.add_subplot(325)
    a5=ax5.hist(x=e.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax5.plot(a5[1][1:]-(a5[1][1:]-a5[1][:-1])/2,a5[0], color='k')
    ax5.set_ylabel("CDF")
    ax5.set_xlabel("Temperature in °C")
    plt.title('CDF, Sensor E')
    plt.show()

def CDF_WINDSPEED(a, b, c, d, e):
    fig = plt.figure(figsize=(20,8))
    plt.subplots_adjust(wspace = 0.3, hspace = 0.5)
    fig.suptitle("CDF Wind Speed (m/s)")
    
    ax1 = fig.add_subplot(321)
    a1=ax1.hist(x=a.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax1.plot(a1[1][1:]-(a1[1][1:]-a1[1][:-1])/2,a1[0], color='k')
    ax1.set_ylabel("CDF")
    ax1.set_xlabel("Wind Speed (m/s)")
    plt.title('CDF, Sensor A')

    ax2 = fig.add_subplot(322)
    a2=ax2.hist(x=b.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax2.plot(a2[1][1:]-(a2[1][1:]-a2[1][:-1])/2,a2[0], color='k')
    ax2.set_ylabel("CDF")
    ax2.set_xlabel("Wind Speed (m/s)")
    plt.title('CDF, Sensor B')

    ax3 = fig.add_subplot(323)
    a3=ax3.hist(x=c.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax3.plot(a3[1][1:]-(a3[1][1:]-a3[1][:-1])/2,a3[0], color='k')
    ax3.set_ylabel("CDF")
    ax3.set_xlabel("Wind Speed (m/s)")
    plt.title('CDF, Sensor C')
    
    ax4 = fig.add_subplot(324)
    a4=ax4.hist(x=d.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax4.plot(a4[1][1:]-(a4[1][1:]-a4[1][:-1])/2,a4[0], color='k')
    ax4.set_ylabel("CDF")
    ax4.set_xlabel("Wind Speed (m/s)")
    plt.title('CDF, Sensor D')

    ax5 = fig.add_subplot(325)
    a5=ax5.hist(x=e.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax5.plot(a5[1][1:]-(a5[1][1:]-a5[1][:-1])/2,a5[0], color='k')
    ax5.set_ylabel("CDF")
    ax5.set_xlabel("Wind Speed (m/s)")
    plt.title('CDF, Sensor E')
    plt.show()
     

CDF_TEMP2(df_A["Temperature"].astype(float),df_B["Temperature"].astype(float),df_C["Temperature"].astype(float), df_D["Temperature"].astype(float), df_E["Temperature"].astype(float))
CDF_WINDSPEED(df_A["Wind Speed"].astype(float),df_B["Wind Speed"].astype(float),df_C["Wind Speed"].astype(float), df_D["Wind Speed"].astype(float), df_E["Wind Speed"].astype(float))

### Compute the 95% confidence intervals for variables Temperature and Wind Speed for all the sensors

def mconfidence_interval(data):
    confidence = 0.95
    a = 1.0 * np.array(data)
    n = len(a)
    m = np.mean(a)
    standarderror = stats.sem(a)
    h = standarderror * stats.t.ppf((1 + confidence) / 2., n-1)
    start = m - h
    end = m + h

    file = open('Confidence Intervals Results.csv','a')
    
    file.write(str(start)+ " , " +str(end) +"\n")
      
    file.close()
    
    
mconfidence_interval(df_A['Temperature'].astype(float))
mconfidence_interval(df_B['Temperature'].astype(float))
mconfidence_interval(df_C['Temperature'].astype(float))
mconfidence_interval(df_D['Temperature'].astype(float))
mconfidence_interval(df_E['Temperature'].astype(float))

mconfidence_interval(df_A['Wind Speed'].astype(float))
mconfidence_interval(df_B['Wind Speed'].astype(float))
mconfidence_interval(df_C['Wind Speed'].astype(float))
mconfidence_interval(df_D['Wind Speed'].astype(float))
mconfidence_interval(df_E['Wind Speed'].astype(float))

### Save them in a table (txt or csv form)


### Test the hypothesis: the time series for Temperature and Wind Speed are the same for sensors
TEMP_A=df_A["Temperature"]
TEMP_B=df_B["Temperature"]
TEMP_C=df_C["Temperature"]
TEMP_D=df_D["Temperature"]
TEMP_E=df_E["Temperature"]


### A4 - 2 
def student_t(arr1,arr2):
    
    data = arr1, arr2
    t, p=stats.ttest_ind(data[0],data[1])
    print("t = " + str(t))
    print("p = " + str(p))


student_t(TEMP_E.astype(float).values, TEMP_D.astype(float).values)
student_t(TEMP_D.astype(float).values, TEMP_C.astype(float).values)
student_t(TEMP_C.astype(float).values, TEMP_B.astype(float).values)
student_t(TEMP_B.astype(float).values, TEMP_A.astype(float).values)

student_t(WINDSP_E.astype(float).values, WINDSP_D.astype(float).values)
student_t(WINDSP_D.astype(float).values, WINDSP_C.astype(float).values)
student_t(WINDSP_C.astype(float).values, WINDSP_B.astype(float).values)
student_t(WINDSP_B.astype(float).values, WINDSP_A.astype(float).values)

