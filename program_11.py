#!/bin/env python

""" Program to generate various plots
	in accordance with AGU presentation guidelines
	http://www.projectionnet.com/Styleguide/presentationstyleguide.aspx
    
    Author: Varun Aggarwal
    Username: aggarw82
    Github: https://github.com/Environmental-Informatics/11-presentation-graphics-aggarw82
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import decimal


def ReadData( fileName ):
	"""This function takes a filename as input, and returns a dataframe with
	raw data read from that file in a Pandas DataFrame.  The DataFrame index
	should be the year, month and day of the observation.  DataFrame headers
	should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
	"Date" column should be used as the DataFrame index. The pandas read_csv
	function will automatically replace missing values with np.NaN, but needs
	help identifying other flags used by the USGS to indicate no data is 
	availabiel.  Function returns the completed DataFrame, and a dictionary 
	designed to contain all missing value counts that is initialized with
	days missing between the first and last date of the file."""
	
	# define column names
	colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

	# open and read the file
	DataDF = pd.read_csv(fileName, 
						 names=colNames,  
						 header=1, 
						 delimiter=r"\s+",parse_dates=[2], comment='#',
						 na_values=['Eqp'])
	DataDF = DataDF.set_index('Date')
	
	# To remove negative streamline values 
	for i in DataDF["Discharge"]:
		if i < 0: 
			DataDF["Discharge"][i] = np.NaN
	
	# quantify the number of missing values
	MissingValues = DataDF["Discharge"].isna().sum()
	
	# filtering out NoDate values 
	DataDF = DataDF.dropna()
	
	return( DataDF, MissingValues )

def ClipData( DataDF, startDate, endDate ):
	"""This function clips the given time series dataframe to a given range 
	of dates. Function returns the clipped dataframe and and the number of 
	missing values."""
	   
	# Clips the data to date range: startDate to endDate 
	DataDF = DataDF.loc[startDate:endDate]
	
	# quantify the number of missing values
	MissingValues = DataDF["Discharge"].isna().sum()
	
	return( DataDF, MissingValues )

def ReadMetrics( fileName ):
	"""This function takes a filename as input, and returns a dataframe with
	the metrics from the assignment on descriptive statistics and 
	environmental metrics.  Works for both annual and monthly metrics. 
	Date column should be used as the index for the new dataframe.  Function 
	returns the completed DataFrame."""
	
	# read matrices
	DataDF = pd.read_csv( fileName, 
						  header=0,
						  delimiter=',',
						  parse_dates=['Date'],
						  index_col=['Date'])

	return( DataDF )


def GetMonthlyAverages(MoDataDF):
    """This function calculates annual average monthly values for all 
    Mean Flow. The routine returns an array of mean values 
    in the original dataframe."""
    
    # defining months and column names 
    month = [3,4,5,6,7,8,9,10,11,0,1,2]
    columns = ['Mean Flow']

    # creating dataframe to store monthly statistics 
    MonthlyAverages = pd.DataFrame( 0, index=range(1, 13), 
                                       columns = columns)
    

    for i in range(12):
        MonthlyAverages.iloc[i,0]=MoDataDF['Mean Flow'][month[i]::12].mean()

    return( MonthlyAverages )

def annualPeakFlow(MoDataDF):
    """This function calculates annual peak flow ."""
    
    # sort values
    # MoDataDF = MoDataDF.sort_values(ascending=False)

    # rank values 
    ranks = sc.rankdata(MoDataDF, method='average')
    ranks = ranks[::-1]

    # length of MoDataDF
    len_DF = len(MoDataDF)

    # plotting position
    prob = [(ranks[i] / (len_DF+1) ) for i in range(0, len_DF)]

    return( prob )


# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

	# define full river names as a dictionary so that abbreviations are not used in figures
	riverName = { "Wildcat": "Wildcat Creek",
				  "Tippe": "Tippecanoe River" }
	
	# read raw data
	fileName = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
				 "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }

	# define blank dictionaries (these will use the same keys as fileName)
	DataDF = {}
	MissingValues = {}
	
	# figure font
	font = {'family' : 'normal',
	        'weight' : 'normal',
	        'size'   : 12}

	# process input datasets
	for file in fileName.keys():
		
		print( "\n", "="*50, "\n  Working on {} \n".format(file), "="*50, "\n" )
		
		DataDF[file], MissingValues[file] = ReadData(fileName[file])
		print( "-"*50, "\n\nRaw data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
		
		# clip to last 5 years
		DataDF[file], MissingValues[file] = ClipData( DataDF[file], '2014-10-01', '2019-09-30' )
		print( "-"*50, "\n\nSelected period data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))

		# plotting discharge for both rivers
		plt.plot( DataDF[file]['Discharge'], 
				  label=riverName[file])

	plt.title('Daily Flow in Tippecanoe and Wildcat Streams')
	plt.xlabel('Date')
	plt.ylabel('Discharge (cfs)')
	plt.legend()
	plt.rc('font', **font)
	plt.savefig('Daily_flow_both.png', dpi=96) 
	plt.close()


	# dictionary for matrices
	met_filename = { "Ann": "Annual_Metrics.csv", "Mon": "Monthly_Metrics.csv" }

	# read annual csv files
	Annual_DF = ReadMetrics(met_filename["Ann"])

	# seperate stream data
	tippe_an = Annual_DF.loc[Annual_DF['Station'] == 'Tippe']
	wildcat_an = Annual_DF.loc[Annual_DF['Station'] == 'Wildcat']

	# plotting Annual coefficient of variation - Annual_metrics.csv
	plt.plot(wildcat_an['Coeff Var'],'*--')  
	plt.plot(tippe_an['Coeff Var'],'*--')
	plt.title('Coefficient of Variation of Flow')
	plt.xlabel('Date')
	plt.ylabel('Coefficient of Variation')
	plt.grid()
	plt.legend([riverName['Wildcat'],riverName['Tippe']])
	plt.rc('font', **font)
	plt.savefig('Coeff_Var_both.png', dpi = 96)
	plt.close() 

	# plotting Annual TQ mean - Annual_metrics.csv
	plt.plot(wildcat_an['Tqmean'],'*--')  
	plt.plot(tippe_an['Tqmean'],'*--')
	plt.title('T-Q Mean')
	plt.xlabel('Date')
	plt.ylabel('T-Q Mean')
	plt.grid()
	plt.legend([riverName['Wildcat'],riverName['Tippe']])
	plt.rc('font', **font)
	plt.savefig('Tqmean_both.png', dpi = 96)
	plt.close() 

	# plotting Annual RB index - Annual_metrics.csv
	plt.plot(wildcat_an['R-B Index'],'*--')  
	plt.plot(tippe_an['R-B Index'],'*--')
	plt.title('Richards-Baker Flashiness Index')
	plt.xlabel('Date')
	plt.ylabel('R-B Index')
	plt.grid()
	plt.legend([riverName['Wildcat'],riverName['Tippe']])
	plt.rc('font', **font)
	plt.savefig('R-B_Index_both.png', dpi = 96)
	plt.close() 

	# read monthly csv files
	Montly_DF = ReadMetrics(met_filename["Mon"])

	# seperate stream data
	tippe_mon = Montly_DF.loc[Montly_DF['Station'] == 'Tippe']
	wildcat_mon = Montly_DF.loc[Montly_DF['Station'] == 'Wildcat']

	# plot monthly averages
	plt.plot(GetMonthlyAverages(wildcat_mon),'*--')  
	plt.plot(GetMonthlyAverages(tippe_mon),'*--')
	plt.title('Average Annual Monthly Flow')
	plt.xlabel('Month')
	plt.xticks(range(1,13))
	plt.ylabel('Flow (cfs)')
	plt.grid()
	plt.legend([riverName['Wildcat'],riverName['Tippe']])
	plt.rc('font', **font)
	plt.savefig('ann_mon_flow_both.png', dpi = 96)
	plt.close() 

	# plot annual peak probability

	# sort values
	wildcat_an = wildcat_an['Peak Flow'].sort_values(ascending=False)
	tippe_an = tippe_an['Peak Flow'].sort_values(ascending=False)

	plt.plot(annualPeakFlow(wildcat_an), wildcat_an,'*--')  
	plt.plot(annualPeakFlow(tippe_an), tippe_an,'*--')
	plt.title('Return period of annual peak flow events')
	plt.xlabel('Exceedence Probability')
	plt.xlim(1,0)
	plt.ylabel('peak discharge (cfs)')
	plt.grid()
	plt.legend([riverName['Wildcat'],riverName['Tippe']])
	plt.tight_layout()
	plt.savefig('Exceedence_both.png', dpi = 96)
	plt.close() 