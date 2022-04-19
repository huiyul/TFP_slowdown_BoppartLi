# -*- coding: utf-8 -*-
"""
Script name: Boppart_Li_2021_replication.py
Program written by Mitchell Ochse and Fabián Rivera Reyes
Last updated April 18, 2022


README

Replication programs for
"Productivity slowdown: reducing the measure of our ignorance"
by Timo Boppart and Huiyu Li

To obtain the replication figures: download the replication folder, open
this program in Anaconda, and run this program in its entirety.

Grey comments provide more detailed explanations for specific commands used.

The code is divided in the following sections:
        1. Preamble, imports the necessary programs, defines functions, assigns colors and labels to countries
            using dictionaries, defines base year for calculations
        2. Data Import: Imports and cleans the raw data from the data folder in dropbox
            a. This requires importing and cleaning data from the following sources:
                *Penn World Tables (pates version 10.0, as of October 2021)
                *Steg resource file
                *GDP deflator
                *Quarterly TFP data
                *Annual TFP data
        3. Plotting (from line 424 onwards): creates the necessary figures and exports them to the out/finalfigs
            folder in the replication folder
"""



#PREAMBLE
import os, math, random
import numpy as np
import pandas as pd
import statistics
from matplotlib import pyplot as plt
from scipy import stats
import statsmodels.api as sm

#Define relevant paths
mywd = os.getcwd()
data_path =  os.path.join(mywd, 'data')
output_path = os.path.join(mywd, 'output/')


#####################################################################################
#                                                                                   #
#                                                                                   #
#           Creating dictionaries to relate countries to colors and descriptions    #
#                                                                                   #
#                                                                                   #
#####################################################################################
#Base year has been chosen below
index_year = 1995

#State country abbreviations
lst = ['AUT', 'BEL', 'DNK', 'FRA', 'DEU', 'ITA', 'JPN', 'NLD', 'ESP', 'SWE', 'GBR', 'USA', 'CHN', 'IND']
#State country names in order of abbreviations
names = ['Austria', 'Belgium', 'Denmark', 'France', 'Germany', 'Italy', 'Japan', 'The Netherlands', 'Spain', 'Sweden', 'United Kingdom', 'USA','China','India']
#State colors assigned to countries in order
colors = ['b','r','olive','c','m','y','k','chocolate','teal', 'indigo','gold','g','r','y']

#Establish list of OECD countries
oecd = ['Australia','Austria','Belgium','Canada','Chile','Colombia','Czech Republic','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Iceland','Ireland'
        ,'Israel','Italy','Japan','Republic of Korea','Latvia','Lithuania','Luxembourg','Mexico','Netherlands','New Zealand','Norway','Poland','Portugal','Slovakia','Slovenia',
        'Spain','Sweden','Switzerland','Turkey','United Kingdom','United States']

#Create a dictionary relation country abbreviation and assigned color
i = 0
country_color ={}
for cn in lst:
    country_color[cn] = colors[i]
    i +=1

#Create a dictionary relating column name and variable description
title_dn = {}
title_dn['pop'] = 'Annual Population Growth'
title_dn['emp'] = 'Annual Employment Growth'
title_dn['h'] = 'Annual Total Hours Worked Growth'
title_dn['rtfpna'] = 'Annual Real TFP Growth (Constant National Prices, 2011=1)'
title_dn['rnna'] = 'Annual Capital Stock Growth (Constant Prices, 2011$USD)'
title_dn['rgdpna'] = 'Annual Real GDP Growth (Constant Prices, 2011$USD)'
title_dn['avh'] = 'Annual Average Hours Worked Growth'
title_lvl = {}
title_lvl['pop'] = 'Annual Population'
title_lvl['emp'] = 'Annual Employment'
title_lvl['h'] = 'Annual Total Hours Worked'
title_lvl['rtfpna'] = 'Annual Real TFP (Constant National Prices, 2011=1)'
title_lvl['rgdpna'] = 'Annual Real GDP (Constant Prices, 2011$USD)'
title_lvl['rnna'] = 'Annual Capital Stock (Constant Prices, 2011$USD)'
title_lvl['avh'] = 'Annual Average Hours Worked'

title_cum = {}
title_cum['pop'] = 'Cumulative Population Growth (%s = 100)' % (index_year,)
title_cum['emp'] = 'Cumulative Employment Growth (%s = 100)' % (index_year,)
title_cum['h'] = 'Cumulative Total Hours Worked Growth (%s = 100)' % (index_year,)
title_cum['rtfpna'] = 'Cumulative Real TFP Growth (Constant National Prices, %s = 100)' % (index_year,)
title_cum['rgdpna'] = 'Cumulative Real GDP Growth (Constant Prices, %s = 100)' % (index_year,)
title_cum['rnna'] = 'Cumulative Capital Stock Growth (Constant Prices, %s = 100)' % (index_year,)
title_cum['avh'] = 'Cumulative Average Hours Worked Growth (%s = 100)' % (index_year,)

plotvars_dn = {}
plotvars_dn['gdppercap_growth_ra'] = 'GDP Per Capita'
plotvars_dn['gdpperhr_growth_ra'] = 'GDP Per Hour Worked'
plotvars_dn['rtfpna_growth_ra'] = 'TFP'
plotvars_dn['rgdpna_growth_ra'] = 'GDP'
plotvars_dn['TFPG_ra'] = 'TFP'
plotvars_dn['GDP_eks_growth_ra'] = 'GDP'

#Associate each variable with a specific color
var_colors = {}
legend_colors = ['yellow','blue','red','green','red','green','green','b']
i = 0
for var in ['gdppercap_growth_ra','gdpperhr_growth_ra','rtfpna_growth_ra','rgdpna_growth_ra', 'TFPG_ra','GDP_eks_growth_ra']:
    var_colors[var] = legend_colors[i]
    i+=1












#####################################################################################
#                                                                                   #
#                                                                                   #
#           Importing and Cleaning Raw Data Files to be used,                       #
#           Also outputs the massaged data into excel files that are used later     #
#                                                                                   #
#                                                                                   #
#                                                                                   #
#####################################################################################


#Identify the varlist to be used
plotvars = ['gdpperhr_growth','rtfpna_growth','rgdpna_growth']
varlist = ["country", "year", "pop", "emp", "avh", "rtfpna", "rnna", "rgdpna", 'hc']
varlist_nocon = [x for x in varlist if x != "country" and x != "year"]
varlist_nocon.append('h')

df_pwt = pd.read_excel(io=os.path.join(data_path,'pwt100.xlsx'), sheet_name = 'Data')

#Keep only the relevant variables and copy the dataframe to a new one that we will manipulate
pwt = df_pwt[varlist].copy()
#Adjust the units
pwt.loc[:,'pop'] *= 1000000
pwt.loc[:,'emp'] *= 1000000
pwt.loc[:,'rnna'] *= 1000000
pwt.loc[:,'rgdpna'] *= 1000000
#Create new variables using the existing variables
pwt.loc[:,'h'] = pwt.loc[:,'avh']* pwt.loc[:,'emp']
pwt.loc[:,'gdppercap'] = pwt.loc[:,'rgdpna']/pwt.loc[:,'pop']
pwt.loc[:,'gdpperhr'] = pwt.loc[:,'rgdpna']/pwt.loc[:,'h']



pwt = pwt.pivot(index = 'year', columns = 'country', values = varlist_nocon + ['gdppercap'] + ['gdpperhr'])

minidx = int(pwt.index[0])
pwt = pwt[pwt.index<2020]
"""
Now, since each column is a variable measure for a certain country,
For example, a column could have the title: ('pop','Albania'), where pop is the variable that pertains to the country Albania
We iterate through the columns to calculate each growth rate
"""
for col in list(pwt.columns.values):
    try:
#        if col != 'year' :
        #Here, we find the first nonmissing year of data for a given variable/country pair.
        mincol = int(pwt[col].first_valid_index())
        #...and calculate the year difference between this date and the first date in the whole
        #dataset.
        diff = mincol - minidx
            #for each column (i.e. country var), we're going to define our corresponding 'growth' column
        pwt[(col[0]+"_growth", col[1])] = np.nan
        pwt[(col[0]+"_cumgrowth", col[1])] = np.nan
            #Note the index_year variable is defined by the user, for the desired base cum growth year
        pwt[(col[0]+"_cumgrowth", col[1])][index_year] = 100
        for t in range(1,len(pwt[col])-diff):
        #Replace each obs in the growth column with the correct growth rate between t and t-1
        #(add mincol since index starts at mincol, and mincol+1 is the first year we can calc a growth rate for)
            i = t+mincol
            pwt[(col[0]+"_growth", col[1])][i] = math.log(pwt[col][i]/pwt[col][i-1])
                #Notice due to the different iterations (forward and reverse) that may
                #be necessary to construct the cumulated growth series, we're going to begin
                #by projecting the cumulative growth series FORWARD in time.
            if i > index_year:
                pwt[(col[0]+"_cumgrowth", col[1])][i] = pwt[(col[0]+"_cumgrowth", col[1])][i-1]*math.exp(pwt[(col[0]+"_growth", col[1])][i])
            #As mentioned above, our 'col' column now has the cumulated growth series calculated
            #only for indexes geq than the index_year. Now, let's project the series backwards in time.
        for x in range(index_year, minidx, -1):
            pwt[(col[0]+"_cumgrowth", col[1])][x] = pwt[(col[0]+"_cumgrowth", col[1])][x+1]/math.exp(pwt[(col[0]+"_growth", col[1])][x+1])
    except:
        pass

pwt = pwt[sorted(pwt.columns, key = lambda x: (x[0], x[1]))]
pwt.to_excel(os.path.join(output_path,"festschrift_output.xlsx"), sheet_name="Data")
outtfp = [col for col in pwt.columns.values if col[0] == "rtfpna_cumgrowth"]
tfptbl = pwt[outtfp]
tfptbl.to_excel(os.path.join(output_path,"pwt_tfp.xlsx"), sheet_name="Data")

"""
Recall now we have two dataframes with Penn World Tables 10.0 data:
    *df_pwt- dataframe as imported
    *pwt- cleaned and adjusted data with new variables for growth and cumulative
        growth
"""

###Create Dataframe for Quarterly tfp
qtr_tfp = pd.read_excel(io=os.path.join(data_path,'quarterly_tfp_verJune2021.xlsx'), sheet_name="quarterly", skiprows = 1)
#Identify relevant portion of spreadsheet for analysis and only keep that part
qtr_tfp.drop(0, inplace=True)
qtr_tfp.drop(range(297,316), inplace=True)
#Now select desired variables
qtr_tfp = qtr_tfp[['date','dLP','dtfp','capital deepening', 'dLQ']]


ann_tfp = pd.read_excel(io=os.path.join(data_path,'quarterly_tfp_verJune2021.xlsx'), sheet_name="annual")
ann_tfp = ann_tfp.dropna()
#We want 1960-2019, 5 year trailing moving averages for the columns titled: dLP dLQ dtfp capital deepening
ann_tfp.rename(columns = {'capital deepening':'cap_dep'}, inplace = True)




#Input GDP Deflator
def_df = pd.read_excel(io=os.path.join(data_path,'us_gdpdef.xlsx'), skiprows = 10)

#Input Resource data for Hall and Jones data recreation
resource_df = pd.read_excel(io=os.path.join(data_path,'steg_resource_data.xlsx'), sheet_name = 'Data')

#State assumed values for both alpha and a
alpha = (1/3)
a = (alpha/(1-alpha))
#Copy the PWT data as imported to a new dataframe because we want this data to be in a different shape
hj_df = df_pwt.copy()
#Merge with the resource data we imported above
hj_df = hj_df.merge(resource_df, on = ['countrycode','year'], how = 'left')
#Choosing a specific year to make tables for; we choose latest available
hj_df = hj_df[hj_df['year'] == 2018]
hj_df.rename(columns = {'rgdpo':'Y','cn':'K','emp':'L'}, inplace = True)

#Create desired variables; these include variables that were used in robustness checks but not
# necessarily included in the figures specified
hj_df['Y'] = hj_df['Y']*(1-(hj_df['natural_res']/100))
hj_df['H'] = (hj_df['hc']*hj_df['L'])
hj_df['A'] = (hj_df['Y']/(hj_df['K']**(alpha)))**(1/(1-alpha))*(1/(hj_df['H']))
hj_df['ngdp'] = hj_df['Y']*hj_df.pl_gdpo
hj_df['nk'] = hj_df['rnna']/hj_df.pl_n*hj_df.pl_i
hj_df['Y_c'] = hj_df['ngdp']/hj_df.pl_con    # output in consumption units
hj_df = hj_df[['countrycode','country','year','Y','Y_c','K','H','A', 'L','nk','ngdp','pl_n','pl_gdpo','pl_i','pl_c','pl_con']]
hj_df['Y/L'] = hj_df['Y']/hj_df['L']
hj_df['Y_c/L'] = hj_df['Y_c']/hj_df['L']
hj_df['K/Y**a'] = (hj_df['K']/hj_df['Y'])**(alpha/(1-alpha))
hj_df['H/L'] = hj_df['H']/hj_df['L']
hj_df['pl_n/pl_gdpo'] = hj_df.pl_n/hj_df.pl_gdpo
hj_df['PK/PY'] = hj_df['nk']/hj_df['ngdp']
hj_df['K/Y'] = hj_df['K']/hj_df['Y']
hj_df['pl_i/pl_gdpo'] = hj_df.pl_i/hj_df.pl_gdpo
hj_df['pl_i/pl_c'] = hj_df['pl_i']/hj_df['pl_c']
hj_df['pl_i/pl_con'] = hj_df['pl_i']/hj_df['pl_con']
hj_df['ngdp_2'] = hj_df['Y']*hj_df.pl_c
hj_df['nk_2'] = hj_df.K*hj_df.pl_i
hj_df['PK/PY**a_2'] = (hj_df['nk_2']/hj_df['ngdp_2'])**(alpha/(1-alpha))
hj_df['A_nom_2'] = hj_df['Y/L']/(hj_df['H/L']*hj_df['PK/PY**a_2'])
hj_df['PK/PY**a'] = (hj_df['nk']/hj_df['ngdp'])**(alpha/(1-alpha))
hj_df['A_nom'] = hj_df['Y_c/L']/(hj_df['H/L']*hj_df['PK/PY**a'])
hj_df['ngdp_2'] = hj_df['Y']*hj_df.pl_c
hj_df['nk_2'] = hj_df.K*hj_df.pl_i
hj_df['PK/PY**a_2'] = (hj_df['nk_2']/hj_df['ngdp_2'])**(alpha/(1-alpha))
hj_df['A_nom_2'] = hj_df['Y/L']/(hj_df['H/L']*hj_df['PK/PY**a_2'])
row = hj_df[hj_df['countrycode'] == 'USA']
hj_df.loc[:,'Y':] = hj_df.loc[:,'Y':].div(hj_df.loc[hj_df.index[hj_df['countrycode'] == 'USA'],'Y':].values)






#Preparing dataframes for convergence figures
#Now let's copy the PWT data as imported for one third last dataframe (ra_df_conv), with a better
#shape for the convergence figures
df = pd.read_excel(io=output_path+'festschrift_output.xlsx',
                   sheet_name = 'Data', index_col = 0, header = [0,1])#Get a list of the columns that contain the variables we want to plot...
#Get a list of the columns that contain the variables we want to plot...
growthcols = [col for col in df.columns.values if col[0] in plotvars]
#...and subset the data to contain only these series
growth_df = df[growthcols]
#Now, construct the 'rolling average' version of the series. Just uses the pandas np rolling series.
#Notice we currently do not center this as it is a 5 period trailing moving average
growth_df = growth_df.rolling(window = 5, center = False).mean()
for col in growthcols:
    col_1, col_2 = col
    growth_df.loc[:,(col_1+"_ra", col_2)] = growth_df.loc[:,(col_1, col_2)]*100



#Create a new dataframe of just the rolling average values
ra_cols = [col for col in growth_df.columns.values if "_ra" in col[0]]
ra_df_conv = growth_df[ra_cols]

###Construct the OECD average###
#To do this, we weight each country by its real GDP value
#This first finds each country's real gdp (rgdpna) value and creates a list of these
#Note we subset to check whether or not the country is in the oecd list of countries.
wghts = df[[x for x in df.columns.values if x[0] == 'rgdpna' and x[1] in oecd]].dropna(axis = 1, how = 'any')
#Now, we can keep only the countries in the oecd list
ra_oecd = ra_df_conv[[x for x in ra_df_conv.columns.values if x[1] in oecd]]
#merge in their respective unmodified rgdpna 'weights'
ra_oecd = ra_oecd.merge(wghts, left_index = True, right_index= True)
#unstack the data and drop missing values
ra_oecd = ra_oecd.stack().reset_index().groupby(['year', 'country']).mean().dropna(axis = 0, how ='any')
#ra_oecd = ra_oecd.stack().reset_index()
#Create a new dataset to merge the OECD averages into
oecd_avg = pd.DataFrame(index = ra_oecd.index.unique())
#oecd_avg = pd.DataFrame(ra_oecd.index.get_level_values('year').unique())
for l in list(set([x[0] for x in ra_cols])):
    #Calculate the weighted annual averageof each series for the OECD countries. Weights should be individual country GDPs
    test = ra_oecd.groupby('year').apply(lambda x: np.average(x['%s' %(l)], weights = x['rgdpna'])).to_frame().rename(columns= {0:('%s' %(l), 'OECD Avg')})
    #Merge into this new dataset
    oecd_avg = oecd_avg.merge(test, left_index = True, right_index = True)
#and finally recombine the OECD values into the main ra_df.
pd.MultiIndex.from_tuples(oecd_avg.columns)
oecd_avg.columns = pd.MultiIndex.from_tuples(oecd_avg.columns)
ra_df_conv = ra_df_conv.merge(oecd_avg, left_index = True, right_index = True)


###Construct the nonOECD average###
#To do this, we weight each country by its real GDP value
#This first finds each country's real gdp (rgdpna) value and creates a list of these
#Note we subset to check whether or not the country is in the oecd list of countries.
wghts = df[[x for x in df.columns.values if x[0] == 'rgdpna' and x[1] not in oecd]].dropna(axis = 1, how = 'any')
#Now, we can keep only the countries in the oecd list
ra_nonoecd = ra_df_conv[[x for x in ra_df_conv.columns.values if x[1] not in oecd]]
#merge in their respective unmodified rgdpna 'weights'
ra_nonoecd = ra_nonoecd.merge(wghts, left_index = True, right_index= True)
#unstack the data and drop missing values
ra_nonoecd = ra_nonoecd.stack().groupby(['year', 'country']).mean().dropna(axis = 0, how ='any')
#ra_oecd = ra_oecd.stack().reset_index()
#Create a new dataset to merge the OECD averages into
nonoecd_avg = pd.DataFrame(index = ra_nonoecd.index.unique())
#oecd_avg = pd.DataFrame(ra_oecd.index.get_level_values('year').unique())
for l in list(set([x[0] for x in ra_cols])):
    #Calculate the weighted annual averageof each series for the OECD countries. Weights should be individual country GDPs
    test = ra_nonoecd.groupby('year').apply(lambda x: np.average(x['%s' %(l)], weights = x['rgdpna'])).to_frame().rename(columns= {0:('%s' %(l), 'nonOECD Avg')})
    #Merge into this new dataset
    nonoecd_avg = nonoecd_avg.merge(test, left_index = True, right_index = True)
#and finally recombine the OECD values into the main ra_df.
pd.MultiIndex.from_tuples(nonoecd_avg.columns)
nonoecd_avg.columns = pd.MultiIndex.from_tuples(nonoecd_avg.columns)
#and finally recombine the OECD values into the main ra_df.
ra_df_conv = ra_df_conv.merge(nonoecd_avg, left_index = True, right_index = True)
ra_df_conv = ra_df_conv.reset_index(level=0, inplace=False)
ra_df_conv = ra_df_conv.reset_index(level=0, inplace=False)
ra_df_conv = ra_df_conv.drop_duplicates(subset = [('year', '')], keep ='first')

ra_df_conv_cols = [col for col in ra_df_conv.columns.values if col != ('country', '')]
ra_df_conv = ra_df_conv[ra_df_conv_cols]
ra_df_conv.set_index(('year', ''), inplace = True)


#Preparing dataframes for Figures 1 and 2
#Copy the Cleaned pwt dataframe onto df
df = pwt.copy()
#create the 'rolling average' (ra) version of the desired variables
#Get a list of the columns that contain the variables we want to plot...
growthcols = [col for col in df.columns.values if col[0] in plotvars]
#...and subset the data to contain only these series
growth_df = df[growthcols]
#Now, construct the 'rolling average' version of the series. Just uses the pandas np rolling series.
#Notice we currently do not center this as it is a 5 period trailing moving average
growth_df = growth_df.rolling(window = 5, center = False).mean()

for col in growthcols:
    col_1, col_2 = col
    growth_df.loc[:,(col_1+"_ra", col_2)] = growth_df.loc[:,(col_1, col_2)]*100

#Create a new dataframe of just the rolling average values
ra_cols = [col for col in growth_df.columns.values if "_ra" in col[0]]
ra_df = growth_df[ra_cols]

###Construct the OECD average###
#To do this, we weight each country by its real GDP value
#This first finds each country's real gdp (rgdpna) value and creates a list of these
wghts = df[[x for x in df.columns.values if x[0] == 'rgdpna' and x[1] in oecd]].dropna(axis = 1, how = 'any')
#Now, we can keep only the countries in the oecd list
ra_oecd = ra_df[[x for x in ra_df.columns.values if x[1] in oecd]]
#merge in their respective unmodified rgdpna 'weights'
ra_oecd = ra_oecd.merge(wghts, left_index = True, right_index= True)
#unstack the data and drop missing values
ra_oecd = ra_oecd.stack().reset_index().groupby(['year', 'country']).mean().dropna(axis = 0, how ='any')
#ra_oecd = ra_oecd.stack().reset_index()
#Create a new dataset to merge the OECD averages into
oecd_avg = pd.DataFrame(index = ra_oecd.index.unique())
#oecd_avg = pd.DataFrame(ra_oecd.index.get_level_values('year').unique())
for l in list(set([x[0] for x in ra_cols])):
    #Calculate the weighted annual averageof each series for the OECD countries. Weights should be individual country GDPs
    test = ra_oecd.groupby('year').apply(lambda x: np.average(x['%s' %(l)], weights = x['rgdpna'])).to_frame().rename(columns= {0:('%s' %(l), 'OECD Avg')})
    #Merge into this new dataset
    oecd_avg = oecd_avg.merge(test, left_index = True, right_index = True)
#and finally recombine the OECD values into the main ra_df.
pd.MultiIndex.from_tuples(oecd_avg.columns)
oecd_avg.columns = pd.MultiIndex.from_tuples(nonoecd_avg.columns)
ra_df = ra_df.merge(oecd_avg, left_index = True, right_index = True)








#####################################################################################
#                                                                                   #
#                                                                                   #
#           Plotting                                                                #
#                                                                                   #
#                                                                                   #
#####################################################################################

#Create the panel plots with the different selections of countries#
#We essentially have two loops going on here, one that iteraties through
#the different country groupings defined in regionnames, and another
#to loop through and construct each country's independent plot.


####        FIGURE 1        ####

region_fig = [['Germany','United States','Japan','France']]
#The outermost loop here loops through the country lists
it = 0
for ccode in region_fig:
    countries = len(ccode)\
    #default to 2x2 grid
    ncols = 2
    nrows = 2
    ccode_fig,axs = plt.subplots(nrows,ncols, sharex = 'all', sharey = 'all', gridspec_kw = {'hspace': .4, 'wspace': 0})
    #Now we uniquely identify each plot. This is based on the number of subplots we have in our grid,
    #and each ax{x} corresponds to one figure within that grid. We mainly use this to identify each plot
    #and to more easily modify the legend later.
    if nrows == 2:
        (ax1, ax2), (ax3,ax4) = axs
    else:
        (ax1, ax2), (ax3,ax4), (ax5, ax6) = axs
    #Now here's where we loop through the countries to plot each one.
    #We loop through each country in our list, create a temp plotting_df
    #to contain only the relevant variables (i.e., columns) for that country,
    #and plot that.
    cn = 0
    for country in ccode:
        #here's where we get the relevant column names to plot
        plotcols = [col for col in ra_df.columns.values if country in col[1]]
        order = [1,0,2]
        plotcols = [plotcols[i] for i in order]
        #extract the years as a new variable so that we can use these to plot
        x = ra_oecd.index.get_level_values('year')
        #...and plot them!
        #sn and cn are the row/column numbers of the plot. note cn corresponds
        #to the number iteration of countries we're working with, so we can use
        #that to determine where we want to plot these values on the grid.
        #sn is the column number...
        sn = 0
        if cn%2 !=0:
            sn = 1
        #...fn is the row number.
        fn = cn//2
        #now, we can identify each plot by axs[fn,sn]. Then we iterate through
        #the columns plotting each of the desired variables at a time. We use
        #the var_colors dn to uniquely color each variable, and label using plotvars_dn.
        for meas in list(ra_df[plotcols].columns.values):
            axs[fn,sn].plot(x, ra_df[meas], color = var_colors[meas[0]], label =plotvars_dn[meas[0]])
            axs[fn,sn].axhline(y=0, color = 'black', linestyle = "-", linewidth = 1.25)
                #...add a legend...
            axs[fn,sn].set_title(country)
        #and iterate through the next country
        cn += 1
    #Add some gridlines on each plot.
    axs = np.array(axs)
    for ax in axs.reshape(-1):
        ax.grid(axis = 'y')
    plt.ylim((-2.5,12))
    plt.xticks(np.arange(min(x)+1, max(x)-5, 15.0))
    plt.yticks(np.arange(-2.5,12.5, step = 2.5))
    ccode_fig.text(0.5, 0.15, 'Year', ha='center')
    ccode_fig.text(0.01, 0.55, 'Yearly Growth Rate (%)', va='center', rotation='vertical')
    #plt.annotate('Source: PWT', (.65,0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top', fontsize = 8)
    #Main issue with specifying a legend for a multipane plot: it will have an entry
    #for every series plotted. This means something like 'TFP' will show up once
    #for every country we plot, so multiple times.
    #Solution: all plots have the same plotted variables. So let's only take
    #the legend information from the first plot ax1 for our overall legend.
    handles, labels = ax1.get_legend_handles_labels()
    #Add whitespace to the bottom for the legend.

    #edited bottom line for final figs
    ccode_fig.subplots_adjust(bottom = .25, wspace = .25)
    #and add the actual legend
    ccode_fig.legend(handles = handles,labels = labels, loc = "lower center", ncol = (2+len(plotvars)//2),
                     borderaxespad=1.3)
    plt.savefig(os.path.join(output_path,"fig1.png"), dpi = 300)
    #plt.show()
    plt.close()
    it += 1



################ Graphing Figure 2 #######################
#Initialize the plot
ax = plt.subplot(111)
#We only want the columns from ra_df that correspond to the calculated OECD average
#values. These are uniquely identified by having 'OECD Avg' as their country name, which
#is the second element of the multiindex. So, let's create a subset of the columns here.
plotcols = [col for col in ra_df_conv.columns.values if col[1] == 'OECD Avg']
order = [1,0,2]
plotcols = [plotcols[i] for i in order]
#extract the years as a new variable so that we can use these to plot
x = ra_df_conv.index
#...and plot them!
#We're going to loop through each of the variables (i.e., columns) and plot
#them individually. We'll create a list of the columns based on the plotcols
#above.
for meas in list(ra_df_conv[plotcols].columns.values):
    #Plot the current variable (i.e., column) in the loop,
    #labeling it with the dictionary variable label and variable color dictionary
    ax.plot(x, ra_df_conv[meas], color = var_colors[meas[0]], label = plotvars_dn[meas[0]])
ax.axhline(y=0, color = 'black', linestyle = "-", linewidth = 1.25)
ax.grid(axis = 'y')
plt.ylim((-1,7))
plt.yticks(np.arange(-1,7, step = 1))
plt.xticks(np.arange(1960,2020, step = 10))
plt.legend(loc=(0.2,0.9), ncol = (2+len(plotvars)//2),
                 borderaxespad=-4)
plt.xlabel("Year")
plt.ylabel("Yearly Growth Rate (%)")
plt.tight_layout()
plt.savefig(os.path.join(output_path,"fig2.png"), dpi = 300)
#plt.show()
plt.close()


##### Generating  FIGURES 3        ####
df = df_pwt.copy()
names = [x for x in df['country'].unique()]
abrev_dn = dict(zip(df.country,df.countrycode))
val_tabl = pd.DataFrame(columns = ['Source', 'Sample', 'Period', 'Slope', 'R-squared', 'Standard Error'])
#

df = df[df['country'].isin(names)]
lst = df.country.unique()

number_of_colors = len(lst)
colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
          for i in range(number_of_colors)]
df.loc[:,'pop'] *= 1000000
df.loc[:,'emp'] *= 1000000
df.loc[:,'rnna'] *= 1000000
df.loc[:,'rgdpna'] *= 1000000

tbl = df[['year', 'country', 'rgdpna', 'pop', 'avh', 'emp']].copy()
tbl.loc[:,'h'] = tbl.loc[:,'avh'] * tbl.loc[:,'emp']
tbl['gdppercap'] = tbl['rgdpna']/tbl['pop']
tbl['gdpperhr'] = tbl['rgdpna']/tbl['h']
tbl['hrperemp'] = tbl['h']/tbl['emp']
tbl['empperpop'] = tbl['emp']/tbl['pop']

country_subset = [x for x in tbl['country'].unique()]
tbl_list = list(tbl)
tbl_list.remove('country')
tbl_list.remove('year')
tbl = tbl.pivot(index = 'year', columns = 'country', values = tbl_list)


#So, begin by finding the first index in the data set, i.e. the earliest
#observation of data available over the entire set.
minidx = int(tbl.index[0])
#Now, since each column is a variable measure for a certain country,
#we just need to iterate through the columns to calculate each growth rate
cols = tbl.columns.values
for col in cols:
    tbl[col] = np.log(tbl[col])
tbl = tbl[sorted(tbl.columns, key = lambda x: (x[1], x[0]))]
tfp_df = pd.read_excel(io=os.path.join(output_path,'pwt_tfp.xlsx'), sheet_name = 'Data', header = [0,1])
tfp_df = tfp_df[tfp_df.index > 0]
tfp_df = tfp_df.set_index(tfp_df.columns[0])
tbl = pd.concat([tbl,tfp_df], axis = 1, sort = False)
cols = tbl.columns.values
maxidx = int(tbl.index[-1])
#Modified ranges to just include desired final figures
ranges = ['1960-%s' %(maxidx,), '2000-%s' %(maxidx,)]


graph_df = pd.DataFrame(index = lst, columns = ranges, dtype = 'object')
for it in ranges:
    yrs = it.split('-')
    minyr = int(yrs[0])
    maxyr = int(yrs[1])
    yrdiff = maxyr-minyr
    out = tbl.loc[[minyr,maxyr]].copy()
    for col in out.columns.values:
        out[(col[0]+"_out", col[1])] = np.nan
        if col[0] != "rtfpna_cumgrowth":
            out[(col[0]+"_out", col[1])][maxyr] = ((out[col][maxyr]-out[col][minyr])/yrdiff)*100
        else:
            out[(col[0]+"_out", col[1])][maxyr] = (out[col][maxyr]-out[col][minyr])/yrdiff
    for cn in lst:
        graph_df[it][cn] = (out[('gdppercap',cn)][minyr],out[('gdppercap_out',cn)][maxyr])
    outcols = [c for c in out.columns.values if '_out' in c[0]]
    out = out[outcols]
    out = out[sorted(out.columns, key = lambda x: (x[1], x[0]))]
    outvars = ['rgdpna_out', 'gdppercap_out', 'pop_out', 'gdpperhr_out', 'hrperemp_out', 'empperpop_out', "rtfpna_cumgrowth_out"]
    displaycols = [c for c in out.columns.values if c[0] in outvars]
    out = out[displaycols]
    out = out[out.index == maxyr]
    out = out.unstack().unstack(level = 1).reset_index(level=1, drop = True).T


graph_df['filt'] = [0  if any(isinstance(n, float) and np.isnan(n) for n in x) else 1 for x in graph_df['1960-2019']]
graph_df = graph_df[graph_df['filt'] == 1]
graph_df.drop(columns = 'filt', inplace = True)


tmp_df = graph_df.copy()


tblcols = ['gdppercap']
tcols = [col for col in tbl.columns.values if col[0] in tblcols]
tbl = tbl[tcols]
for col in tcols:
    tbl.loc[:,col] = np.log(tbl.loc[:,col])
tbl = tbl[sorted(tbl.columns, key = lambda x: (x[1], x[0]))]
maxidx = int(tbl.index[-1])

#Modified ranges to just include desired final figures
ranges = ['1960-%s' %(maxidx,), '2000-%s' %(maxidx,)]


graph_df = pd.DataFrame(index = names, columns = ranges, dtype = 'object')

for it in ranges:
    yrs = it.split('-')
    minyr = int(yrs[0])
    maxyr = int(yrs[1])
    yrdiff = maxyr-minyr
    out = tbl.loc[[minyr,maxyr]].copy()
    #graph_df[it] = np.nan
    for col in out.columns.values:
        out[(col[0]+"_out", col[1])] = np.nan
        if col[0] != "TFPG_cumgrowth":
            out[(col[0]+"_out", col[1])][maxyr] = ((out[col][maxyr]-out[col][minyr])/yrdiff)*100
        else:
            out[(col[0]+"_out", col[1])][maxyr] = (out[col][maxyr]-out[col][minyr])/yrdiff
    for cn in names:
        graph_df[it][cn] = (out[('gdppercap',cn)][minyr],out[('gdppercap_out',cn)][maxyr])
    outcols = [c for c in out.columns.values if '_out' in c[0]]
    out = out[outcols]
    out = out[sorted(out.columns, key = lambda x: (x[1], x[0]))]
    outvars = ['GDP_eks_out', 'gdppercap_out', "TFPG_cumgrowth_out"]
    displaycols = [c for c in out.columns.values if c[0] in outvars]
    out = out[displaycols]
    out = out[out.index == maxyr]
    out = out.unstack().unstack(level = 1).reset_index(level=1, drop = True).T



graph_df['filt'] = [0  if any(isinstance(n, float) and np.isnan(n) for n in x) else 1 for x in graph_df['1960-2019']]
graph_df = graph_df[graph_df['filt'] == 1]
graph_df.drop(columns = 'filt', inplace = True)


countries = list(set(graph_df.index.values) | set(tmp_df.index.values))
intc = list(set(graph_df.index.values) & set(tmp_df.index.values))
intc.sort()
graph_df = graph_df[graph_df.index.isin(set(countries))]
tmp_df = tmp_df[tmp_df.index.isin(set(countries))]

graph_df.sort_index(inplace = True)
tmp_df.sort_index(inplace = True)

number_of_colors = len(countries)
colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
          for i in range(number_of_colors)]

marker_dn = dict(zip(countries,colors))

graph_df1 = tmp_df
tmp_df = graph_df
graph_df = graph_df1

colors = [marker_dn[x] for x in graph_df.index.values]



###Unlocking loops fig 3
fig = plt.figure()
minyr = 1960
markers = graph_df.index.tolist()
xs = [x[0] for x in graph_df['1960-2019']]
ys = [x[1] for x in graph_df['1960-2019']]

ax = fig.add_subplot(111)
ax.scatter(xs,ys, marker = '+', color = colors)
i = 0
for x,y in zip(xs,ys):
    plt.annotate(abrev_dn[markers[i]], (x,y), textcoords = "offset points",
                 xytext=(0,4), ha='center')
    i +=1
slope, intercept, r_value, p_value, std_err = stats.linregress(xs,ys)
slope = round(slope,3)
r_value = round(r_value**2,3)
std_err = round(std_err,3)
t_lst = [ 'PWT', 'Full', '1960-2019', slope,std_err, r_value]
val_tabl.loc[len(val_tabl)] = t_lst
plt.plot(np.unique(xs), intercept+slope*np.unique(xs), color = 'black', linestyle = '--')
plt.text(.62,.86,"Slope = %s (%s) \nR-squared = %s" %(slope,std_err,r_value,),
         bbox = dict(edgecolor = 'black', facecolor = 'none'), transform=ax.transAxes)
ax.set(ylabel = 'Average annual growth of GDP per capita 1960-2019', xlabel = 'Log GDP per capita 1960')
axes = plt.gca()
tmp = axes.get_ylim()
plt.ylim(tmp[0],tmp[1]+.25)
plt.savefig(os.path.join(output_path,'fig3.png'), dpi = 300)



#########       Table 1        #######
out_df = hj_df[['country','Y/L','K/Y**a','H/L','A']]
#Now, write the two tables to latex
#out_df.dropna(subset = [x for x in out_df.columns.values if x != "country"], inplace = True)
out_df = out_df.sort_values(by = 'Y/L', ascending = False, inplace = False)

out_df = out_df.rename(columns = {'country':'Country'}, inplace = False)
table1_countries = ["United States", "Switzerland", "France", "Germany", "Sweden", "Japan", "Republic of Korea", "Russian Federation", "Mexico", "Argentina", "Brazil", "China", "India", "Nigeria", "Kenya", "Zimbabwe"]

#
#
#Introducing Table 1, Table 1 and A1 were constructed in similar fashion
for country in table1_countries :
    if country == "United States" :
        out = out_df[out_df['Country'] == country]
    else :
        outy = out_df[out_df['Country'] == country]
        out = out.append(outy)
ltx = out.to_latex(index = False, float_format="%.3f", longtable = True)
ltx_lst = ltx.splitlines()
ltx_lst.insert(3, '\hline')
out_ltx = '\n'.join(ltx_lst)
with open(os.path.join(output_path,"table1.tex"),'w') as tf:
    tf.write(out_ltx)
df_10 = hj_df[['country','Y','Y/L','K/Y**a','PK/PY**a','pl_n','A','pl_i/pl_gdpo','pl_n/pl_gdpo','K','A_nom', 'K/Y', 'PK/PY', 'pl_i', 'pl_c','Y_c/L','pl_i/pl_c','pl_i/pl_con']]
df_10 = df_10.dropna(subset = [x for x in df_10.columns.values if x != "country"], inplace = False)

##################### Figure 4 ############################

###Create the nominal scatterplot###
#Now we can actually make the plot
fig = plt.figure()
ax = fig.add_subplot(111)
#Since we're also running a regression on the desired values,
#I'm going to define the X and Y values here for ease of reference
xs = np.log(hj_df['Y/L']) - np.mean(np.log(hj_df['Y/L']))
ys = np.log(hj_df['A']) - np.mean(np.log(hj_df['A']))
#Create a scatter plot of xs and ys, set marker, and use list of colors
#created above
ax.scatter(xs,ys, marker = '+', color = 'black')
#Run the regression #
#First, define a numpy mask so that we can subset the xs and ys to include
#ONLY observations that do not have either a missing x or y value. The regression
#code sometimes does not handle missing values well.
mask = ~np.isnan(xs) & ~np.isnan(ys)
#Use the mask when passing in the values to the regression itself. This is
#a simple OLS that estimates y ~ x + e.
model = sm.OLS(ys[mask],xs[mask])
#save the regression results stored in model
results = model.fit()
#Plot the line of best fit. Above, we simply included a model with no y intercept/cons.
#So, we just want to plot y = slope*x. We can do this by plotting the unique X values
#and estimating the y values from the model above. results.params[0] gives us the slope
#coefficient, which we can multiply our array of unique x values by for the fitted value.
plt.plot(np.unique(xs), results.params[0]*np.unique(xs), color = 'black', linestyle = "--")
#         round(results.bse[0],3),round(results.rsquared,3)),
#         bbox = dict(edgecolor = 'black', facecolor = 'none'), transform=ax.transAxes)
plt.text(0.62,0.40, "Slope = %s" %(round(results.params[0],3)),
         bbox = dict(edgecolor = 'black', facecolor = 'none'), transform=ax.transAxes)
plt.text(0.13,0.05,"45 degree line", color='grey', bbox = dict(edgecolor = 'grey', facecolor = 'none'),
         transform=ax.transAxes)
ax.set(ylabel = 'log TFP', xlabel = 'log output per worker')
#Plot a diagonal line here. We use the transform=ax.transAxes so that our value ranges
#for x and y are intepreted as fractions of each axis values; i.e. [0,1] becomes
#Plot the line for x values from the first to last x point (0-100%).
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])+.1]
ax.plot(lims,lims, color = 'grey', linewidth = 1, linestyle = ":")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal')
#Save a version of the figure before adding labels
plt.savefig(os.path.join(output_path,'fig4.png'), dpi = 300)
#plt.show()
plt.close()

##################### Figure 5 ############################

fig = plt.figure()
ax = fig.add_subplot(111)
#Since we're also running a regression on the desired values,
#I'm going to define the X and Y values here for ease of reference
xs = np.log(df_10['Y_c/L']) - np.mean(np.log(df_10['Y_c/L']))
ys = np.log(df_10['A_nom']) - np.mean(np.log(df_10['A_nom']))
#Create a scatter plot of xs and ys, set marker, and use list of colors
#created above
ax.scatter(xs,ys, marker = '+', color = 'black')

#Run the regression #
#First, define a numpy mask so that we can subset the xs and ys to include
#ONLY observations that do not have either a missing x or y value. The regression
mask = ~np.isnan(xs) & ~np.isnan(ys)
#Use the mask when passing in the values to the regression itself. This is
#a simple OLS that estimates y ~ x + e.
model = sm.OLS(ys[mask],xs[mask])
#save the regression results stored in model
results = model.fit()
#Plot the line of best fit. Above, we simply included a model with no y intercept/cons.
#So, we just want to plot y = slope*x. We can do this by plotting the unique X values
#and estimating the y values from the model above. results.params[0] gives us the slope
#coefficient, which we can multiply our array of unique x values by for the fitted value.
plt.plot(np.unique(xs), results.params[0]*np.unique(xs), color = 'black', linestyle = "--")
#Add in a text box with some regression information
plt.text(.62,.4,"Slope = %s " %(round(results.params[0],3)),
         bbox = dict(edgecolor = 'black', facecolor = 'none'), transform=ax.transAxes)
plt.text(0.13,0.05,"45 degree line", color='grey', bbox = dict(edgecolor = 'grey', facecolor = 'none'),
         transform=ax.transAxes)
ax.set(ylabel = 'log TFP', xlabel = 'log output per worker')
#Plot a diagonal line here. We use the transform=ax.transAxes so that our value ranges
#for x and y are intepreted as fractions of each axis values; i.e. [0,1] becomes
#Plot the line for x values from the first to last x point (0-100%).
lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()])+.1]
ax.plot(lims,lims, color = 'grey', linewidth = 1, linestyle = ":")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal')
plt.savefig(os.path.join(output_path,'fig5.png'), dpi = 300)
#plt.show()
plt.close()




################## Graphing Figure 6 ######################

#specified desired vars
df = ann_tfp.copy()
variables_avg = ['dLP', 'dLQ', 'dtfp', 'cap_dep']
for v in variables_avg:
        df[v+'_avg'] = df[v].rolling(window = 5).mean().sort_index(level = 1).values
plotvars = ['dLP_avg', 'dLQ_avg', 'dtfp_avg', 'cap_dep_avg']
#Identifying colors
var_colors_tfp = {}
legend_colors = ['purple','blue','red','green','red','green','green','b']
i = 0
for var in ['dLP','dLQ','dtfp','cap_dep']:
    var_colors_tfp[var] = legend_colors[i]
    i+=1
#Identifying labels
plotvars_tfp = {}
plotvars_tfp['dLP'] = 'LP'
plotvars_tfp['dtfp'] = 'TFP'
plotvars_tfp['dLQ'] = 'LQ'
plotvars_tfp['cap_dep'] = 'capital deepening'
ax = plt.subplot(111)
plt_df = df[['date','dLP_avg', 'dLQ_avg', 'dtfp_avg', 'cap_dep_avg']]
x = plt_df.date
plotcols = [col for col in plt_df.columns.values if col != 'date']
order = [1,0,2,3]
plotcols = [plotcols[i] for i in order]
for var in variables_avg:
    #Plot the current variable (i.e., column) in the loop,
    #labeling it with the dictionary variable label and variable color dictionary
    ax.plot(x, plt_df[var+'_avg'], color = var_colors_tfp[var], label = plotvars_tfp[var])

ax.axhline(y=0, color = 'black', linestyle = "-", linewidth = 1.25)
ax.grid(axis = 'y')
plt.ylim((-1,7))
plt.yticks(np.arange(-1,7, step = 1))
plt.xticks(np.arange(1960,2020, step = 10))
plt.legend(loc=(0.05,0.9), ncol = (2+len(plotvars)//2),
                 borderaxespad=-4)
plt.xlabel("Year")
plt.ylabel("Yearly Growth Rate (%)")
plt.tight_layout()
plt.savefig(os.path.join(output_path,"fig6.png"), dpi = 300)
#plt.show()
plt.close()



#######          Generating Table2        ########
#Creating TFP table
#Fernald, Li and Ochse (2021) replication
#https://www.frbsf.org/economic-research/indicators-data/total-factor-productivity-tfp/

#Now select desired variables
qtr_tfp = qtr_tfp[['date','dLP','dtfp','capital deepening', 'dLQ']]


#Now let's try to establish an appropriate temporal index
qtr_tfp[['year','quarter']] = qtr_tfp.date.str.split(":Q",expand=True)
qtr_tfp['year'] = qtr_tfp['year'].astype(int)
qtr_tfp['quarter'] = qtr_tfp['quarter'].astype(int)

index = pd.PeriodIndex(year=qtr_tfp.year, quarter=qtr_tfp.quarter, freq='Q-DEC')
qtr_tfp = qtr_tfp.set_index(index)
#index is now set and of the form 1947Q2 == 1947:Q2

#Now, establish set date frames for which the two-way table should be specified
ranges= ['1996Q1-2004Q4','2005Q1-2019Q4']
vars_in_table = ['dLP','dtfp','capital deepening', 'dLQ']



#Now let's make a dataframe of averages for the time periods specified in ranges
#Notice how the for loop below loops through the ranges specified above to calculate the aggregate statistics
index = []
var_names = [x for x in vars_in_table]
averages = []
for quarter in ranges:
    period = quarter.split('-')
    minqtr = pd.Period(period[0],freq='Q')
    maxqtr = pd.Period(period[1],freq='Q')
    subset_df = qtr_tfp.loc[minqtr:maxqtr]
    #This list comprehension makes a series of lists inside of the list started before the for loop. The lists have the calculated
    #statistics in order of the variables specified in vars_in_table
    averages.append([statistics.mean(subset_df[x]) for x in vars_in_table])
    index.append(quarter)


#Bulk of the data is created. Now we need to append rows for subsequent desired calculations.
#The title of the row added.
index.append("Change")
#Calculating the row itself. Notice that for generalization purposes, the for loop goes through the amount of
#variables pre-specified. Recall that the first element in the list is indexed by 0 and the last by n-1.
change = []
for i in range(len(vars_in_table)):
    x = averages[0]
    y = averages[1]
    z = y[i] - x[i]
    change.append(z)

averages.append(change)

#Now we need to add the final row that has nothing in the index portion
#Specifying the title of the last row.
index.append("")

#Now we need to calculate the missing percentages
percentages = []
for i in range(len(vars_in_table)):
    if i == 0:
        percentages.append("")
    else:
        changes = averages[-1]
        z = changes[i]/changes[0]
        z = 100*z
        z = str(int(round(z,0)))+"%"
        percentages.append(z)
#Print to verify that the row has the desired calculations in the desired order
print(percentages)
averages.append(percentages)


#Now let's convert everything into a dataframe, the order of inputs seems critical here.
#1. Body of data, 2. index (Row titles), 3. column titles
df_tfp = pd.DataFrame(averages, index, var_names)
#Print to verify output is as expected.
print(df_tfp)

#Now let's export the dataframe as a LaTeX table. Check with to_latex() options for additional formatting.
with open(os.path.join(output_path,'table2.tex'),'w') as tf:
    tf.write(df_tfp.to_latex())






# Appendix Figures


#Plotting Figure A1
#Initialize the plot
ax = plt.subplot(111)
#We only want the columns from ra_df that correspond to the calculated OECD average
#values. These are uniquely identified by having 'nonOECD Avg' as their country name, which
#is the second element of the multiindex. So, let's create a subset of the columns here.
plotcols = [col for col in ra_df_conv.columns.values if col[1] == 'nonOECD Avg']
order = [1,0,2]
plotcols = [plotcols[i] for i in order]
#extract the years as a new variable so that we can use these to plot
x = ra_df_conv.index
#...and plot them!
#We're going to loop through each of the variables (i.e., columns) and plot
#them individually. We'll create a list of the columns based on the plotcols
#above.
for meas in list(ra_df_conv[plotcols].columns.values):
    #Plot the current variable (i.e., column) in the loop,
    #labeling it with the dictionary variable label and variable color dictionary
    ax.plot(x, ra_df_conv[meas], color = var_colors[meas[0]], label = plotvars_dn[meas[0]])
ax.axhline(y=0, color = 'black', linestyle = "-", linewidth = 1.25)
ax.grid(axis = 'y')
plt.ylim((-3,7))
plt.yticks(np.arange(-3,7, step = 1))
plt.xticks(np.arange(1960,2020, step = 10))
plt.legend(loc=(0.2,0.9), ncol = (2+len(plotvars)//2),
                 borderaxespad=-4)
plt.ylabel("Yearly Growth Rate (%)")
plt.xlabel("Year")
plt.tight_layout()
plt.savefig(os.path.join(output_path,"figA1.png"), dpi = 300)
#plt.show()
plt.close()




###Plotting Figure A2
fig = plt.figure()
minyr = 2000
markers = graph_df.index.tolist()
xs = [x[0] for x in graph_df['2000-2019']]
ys = [x[1] for x in graph_df['2000-2019']]

ax = fig.add_subplot(111)
ax.scatter(xs,ys, marker = '+', color = colors)
i = 0
for x,y in zip(xs,ys):
    plt.annotate(abrev_dn[markers[i]], (x,y), textcoords = "offset points",
                 xytext=(0,4), ha='center')
    i +=1
slope, intercept, r_value, p_value, std_err = stats.linregress(xs,ys)
slope = round(slope,3)
r_value = round(r_value**2,3)
std_err = round(std_err,3)
t_lst = [ 'PWT', 'Full', '2000-2019', slope,std_err, r_value]
val_tabl.loc[len(val_tabl)] = t_lst
plt.plot(np.unique(xs), intercept+slope*np.unique(xs), color = 'black', linestyle = '--')
plt.text(.62,.86,"Slope = %s (%s) \nR-squared = %s" %(slope,std_err,r_value,),
         bbox = dict(edgecolor = 'black', facecolor = 'none'), transform=ax.transAxes)
ax.set(ylabel = 'Average annual growth of GDP per capita 2000-2019', xlabel = 'Log GDP per capita 2000')
axes = plt.gca()
tmp = axes.get_ylim()
plt.ylim(tmp[0],tmp[1]+.25)
plt.savefig(os.path.join(output_path,'figA2.png'), dpi = 300)


#### Nominal HJ table A1
out_df = hj_df[['country','Y_c/L','PK/PY**a','H/L','A_nom']]
#out_df.dropna(subset = [x for x in out_df.columns.values if x != "country"], inplace = True)
out_df = out_df.sort_values(by = 'Y_c/L', ascending = False, inplace = False)

out_df = out_df.rename(columns = {'country':'Country', 'A_nom' :'A'}, inplace = False)
ltx = out_df.to_latex(index = False, float_format="%.3f", longtable = True)
ltx_lst = ltx.splitlines()
ltx_lst.insert(3, '\hline')
out_ltx = '\n'.join(ltx_lst)

##Adding Table A1
###############################################################################
#
for country in table1_countries :
    if country == "United States" :
        out = out_df[out_df['Country'] == country]
    else :
        outy = out_df[out_df['Country'] == country]
        out = out.append(outy)
ltx = out.to_latex(index = False, float_format="%.3f", longtable = True)
ltx_lst = ltx.splitlines()
ltx_lst.insert(3, '\hline')
out_ltx = '\n'.join(ltx_lst)
with open(os.path.join(output_path,"tableA1.tex"),'w') as tf:
    tf.write(out_ltx)
