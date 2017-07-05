import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#%matplotlib inline

################loading the data and reshaping for analysis####################
#nhrm = pd.read_csv('C:/Users/Cat/Desktop/Main_NRHM123.csv', names = ["Block Name","Type of Facility","Name of the facility","Indicators","S.No","Parameters","01-04-2015","01-05-2015","01-06-2015","01-07-2015","01-08-2015","01-09-2015","01-10-2015","01-11-2015","01-12-2015","01-01-2016","01-02-2016","01-03-2016","Total Report"],encoding='utf8')
nhrm = pd.read_csv('C:/Users/Cat/Desktop/Main_NRHM1234.csv', encoding='utf8')

#nhrm.groupby('Taluk Name').describe()

#print( nhrm.head())
nhrm.dtypes

df1 = nhrm[["Name of District","Name of Health Block","Health Facility","Type of Facility","Name of the facility","Technical Components","Health Service","S.No","Indicators","Total Report"]]
#df2 = nhrm[["Block Name","Type of Facility","Name of the facility","Indicators","Parameters","01-04-2015","01-05-2015","01-06-2015","01-07-2015","01-08-2015","01-09-2015","01-10-2015","01-11-2015","01-12-2015","01-01-2016","01-02-2016","01-03-2016"]]
df2 = nhrm[["Name of District","Name of Health Block","Health Facility","Type of Facility","Name of the facility","Technical Components","Health Service","S.No","Indicators","Apr-15","May-15","Jun-15","Jul-15","Aug-15","Sep-15","Oct-15","Nov-15","Dec-15","Jan-16","Feb-16","Mar-16"]]
df1 = df1.rename_axis({"S.No":"SNo"}, axis="columns")
df1.dtypes
df3 = df2.rename_axis({"S.No":"SNo","Apr-15":"Apr-2015","May-15":"May-2015","Jun-15":"Jun-2015","Jul-15":"Jul-2015","Aug-15":"Aug-2015","Sep-15":"Sep-2015","Oct-15":"Oct-2015","Nov-15":"Nov-2015","Dec-15":"Dec-2015","Jan-16":"Jan-2016","Feb-16":"Feb-2016","Mar-16":"Mar-2016"}, axis="columns")
df3.dtypes
#print(df1.head())
#print(df2.head())

melted = pd.melt(df3, id_vars=["Name of District","Name of Health Block","Health Facility","Type of Facility","Name of the facility","Technical Components","Health Service","SNo","Indicators"],var_name=["Month"],value_name="Monthly Reports")        
#total_report = df1.stack()
#wideframe = df1.pivot("Block Name","Name of the facility","Parameters","Total Report")
#df1 = pd.DataFrame(df1)

melted.head(5)
melted.tail(5)
melted.shape
melted.describe()
melted.dtypes
#visualise time series data
from matplotlib import pyplot
melted.plot()
pyplot.show()
#### Grouping or clustering the data##########################
df1.dtypes
df1 = df1.convert_objects(convert_numeric=True)
Parameter_reports =  df1['Total Report'].groupby( df1['SNo'])
Parameter_agg = Parameter_reports.sum()
Parameter_agg

Parameter_reports1 =  df1['Total Report'].groupby( df1['Indicators'])
Parameter_agg1 = Parameter_reports1.sum()
Parameter_agg1
Parameter_agg1.to_csv("C:/Users/Cat/Desktop/Indicator.csv", sep=',', encoding='utf-8')

Block_Report =  df1['Total Report'].groupby( df1['Name of Health Block'])
Block_sum = Block_Report.sum()
Block_sum


tfecility_Report =  df1['Total Report'].groupby(df1['Type of Facility'])
tfecility_sum = tfecility_Report.sum()
tfecility_sum

service_Report =  df1['Total Report'].groupby(df1['Health Service'])
service_Report = service_Report.sum()

Program_Report1 =  melted['Monthly Reports'].groupby([melted['Technical Components'],melted['Month']])
program_sum1 = Program_Report1.sum()
program_sum1

Program_Report =  df1['Total Report'].groupby( df1['Technical Components'])
program_sum = Program_Report.sum()
program_sum.head(10)

Facility_report =  df1['Total Report'].groupby( df1['Health Facility'])
Facility_agg = Facility_report.sum()
Facility_agg
agg = Facility_report.agg(['mean','std'])
#######task1###########
import matplotlib.pyplot as plt
import seaborn as sns
sns.clustermap(agg)
########################
Facility_report1 =  df1['Total Report'].groupby( [df1['Health Facility'],df1['Type of Facility'],df1['Name of the facility']])
Facility_agg2 = Facility_report1.sum()
Facility_agg2

Indicator_report =  melted['Monthly Reports'].groupby([melted['Health Facility'],melted['Technical Components'],melted['Health Service'],melted['SNo'],melted['Indicators'], melted['Month']])
Indicator_agg1 = Indicator_report.sum()
#Indicator_agg1.unstack()

#df.to_csv("new2.csv", sep='\t', encoding='utf-8')
##hierarchy clustering
#agg_mean = ggroup.agg(['mean','std',range])

ggroup2 = df1['Total Report'].groupby([df1["Name of Health Block"],df1["Health Facility"],df1["Type of Facility"],df1["Name of the facility"]])
cluster = ggroup2.groups
sum1 = ggroup2.sum()
sum1.to_csv("C:/Users/Cat/Desktop/health.csv", sep=',', encoding='utf-8')


ggroup1 = melted['Monthly Reports'].groupby([melted["Name of District"],melted["SNo"],melted['Month']])
sum2 = ggroup1.sum()

ggroup3 = melted['Monthly Reports'].groupby([melted["Name of District"],melted["SNo"],melted["Indicators"],melted['Month']])
sum3 = ggroup3.sum()
#group_name.size().unstack()
 #########Making Dataframes from aggregated data and visualisation###########
Block_df = pd.DataFrame(Block_sum)
Block_df = Block_df.reset_index(level=['Name of Health Block'])
#visualisation blocks report
#hist
Block_df.set_index(["Name of Health Block"],inplace=True)
Block_df.plot(kind='bar',alpha=0.75)
plt.xlabel("")
#pie chart
colors = ['#F0F8FF', '#FAEBD7','#00FFFF','#FFA500','#000080','#800000','#FFFFE0','#87CEFA','#FFB6C1','#7CFC00','#FFFFF0','#FFFACD','#808080']
plt.pie(
    Block_df['Total Report'],
    labels=Block_df['Name of Health Block'],
    shadow=False,
    colors=colors,
    # with one slide exploded out
    explode=(0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0.1, 0, 0, 0.15),
    # with the start angle at 90%
    startangle=90,
    # with the percent listed as a fraction
    autopct='%1.1f%%',
    )
# View the plot drop above
plt.axis('equal')
# View the plot
plt.tight_layout()
plt.show()

fecility_df = pd.DataFrame(Facility_agg)
fecility_df = fecility_df.reset_index(level=['Health Facility'])
fecility_df.set_index(["Health Facility"],inplace=True)
#histogram
fecility_df.plot(kind='bar',alpha=0.75)
plt.xlabel("")
#pie chart
colors = ['#008000','#808080','#FFD700']
plt.pie(
    fecility_df['Total Report'],
    labels=fecility_df['Health Facility'],
    shadow=False,
    colors=colors,
    # with one slide exploded out
    explode=(0, 0, 0.15),
    # with the start angle at 90%
    startangle=90,
    # with the percent listed as a fraction
    autopct='%1.1f%%',
    )
# View the plot drop above
plt.axis('equal')
# View the plot
plt.tight_layout()
plt.show()

#visualisation of technical components
program_sum.plot(kind='bar',alpha=0.75)
plt.xlabel("")


#health service visualization
service = pd.DataFrame(service_Report)
service.plot(kind='bar',alpha=0.75)
plt.xlabel("")

#total reports of type of fecility
tfecility = pd.DataFrame(tfecility_sum)
tfecility.plot(kind='bar',alpha=0.75)
plt.xlabel("")


bK_feci = pd.DataFrame(sum1)
bK_feci = bK_feci.reset_index(level=["Name of Health Block",'Health Facility',"Type of Facility","Name of the facility"])


IndicatorNo_df = pd.DataFrame(Indicator_agg1)
IndicatorNo_df = IndicatorNo_df.reset_index(level=['Health Facility','Technical Components','Health Service','SNo', 'Month'])

IndicatorNo_df1 = pd.DataFrame(sum3)
IndicatorNo_df1 = IndicatorNo_df1.reset_index(level=['Name of District','SNo','Indicators', 'Month'])

Indicator = pd.DataFrame(sum2)
Indicator = Indicator.reset_index(level=["Name of District","SNo", 'Month'])
Indicator.to_csv("C:/Users/Cat/Desktop/Indicator.csv", sep=',', encoding='utf-8')

#################visualisation#############################
import matplotlib.pyplot as plt
import seaborn

date1 = pd.to_datetime(Indicator['Month'], format="%b-%Y")
date12 = pd.DataFrame(date1, columns = ['Month'])
df_data1 = Indicator[["Name of District","SNo"]]
df_data12 = Indicator['Monthly Reports']
IndicatorNo = pd.concat([df_data1, date12, df_data12], axis=1)
IndicatorNo = IndicatorNo.rename_axis({"Name of District":"Name_of_District"}, axis="columns")
IndicatorNo.dtypes
IndicatorNo.shape

all_names_index = IndicatorNo.set_index(['Name_of_District','SNo','Month']).sort_index()

plt.figure(figsize = (14, 4))
def name_plot(Name_of_District, SNo):
    data = all_names_index.loc[Name_of_District, SNo]
    plt.plot(data.index, data.values)

name_plot('Nagpur', '10.1.02')

#visualise whole data for observing trends
names = ['1.1', '1.1.1','1.2','1.3','1.4.1','1.4.2','1.5','1.6.1','1.6.2','1.7.1','1.7.2','1.8','2.1.1.a','2.1.1.b','2.1.1.c','2.1.2','2.1.3','2.2','2.2.1','2.2.2.a','2.2.2.b','2.2.2.c','2.3','2.3.1.a','2.3.1.b','2.3.1.c','3.1.1','3.1.2','3.1.3','3.1.4','3.1.5','3.2','4.1.1.a','4.1.1.b','4.1.1.c','4.1.2','4.1.3','4.2.1','4.2.2','4.3','5.1.1','5.1.2','5.1.3','5.1.4','5.1.5','5.2','5.3.1','5.3.2','5.3.3','5.3.4','6.1','6.2','6.3','7.1.1','7.1.2','7.1.3','7.2','8.1.a','8.1.b','8.1.c','8.2','9.1.1.a','9.1.1.b','9.1.1.c','9.1.1.d','9.1.1.e','9.1.2','9.2.1.a','9.2.1.b','9.2.1.c','9.2.1.d','9.2.1.e','9.2.2','9.3.1.a','9.3.1.b','9.3.1.c','9.3.1.d','9.3.1.e','9.3.2','9.4.1.a','9.4.1.b','9.4.1.c','9.4.1.d','9.4.1.e','9.4.2','9.5.1.a','9.5.1.b','9.5.1.c','9.5.1.d','9.5.1.e','9.5.1.f','9.5.1A','9.5.2','9.06','9.07','9.08','9.09','9.1','9.11.1.a','9.11.1.b','9.11.2.a','9.11.2.b','9.11.3.a','9.11.3.b','9.12','10.1.01','10.1.02','10.1.03','10.1.04','10.1.04A','10.1.04B','10.1.04C','10.1.05','10.1.06','10.1.07','10.1.08','10.1.09','10.1.10','10.1.11','10.1.12','10.1.13','10.1.14','10.1.13.a','10.1.13.b ','10.1.13.c','10.1A','10.2.1','10.2.2','10.2.3','10.3.1.a','10.3.1.b','10.3.1.c','10.3.2','10.3.3','10.3.4','10.3.5.a','10.3.5.b','10.3.5.c','10.4.1','10.4.2','10.4.3','11.1.1','11.1.2','11.1.3','12.1','12.2','12.3','12.4','12.5','12.6','12.7','12.8','12.9','13.1','13.2','13.3','13.4','13.5','13.6','14.01','14.02','14.03','14.04','14.05','14.06','14.07','14.08','14.09','14.10.1.a','14.10.1.b','14.10.1.c','14.10.2.a','14.10.2.b','14.10.2.c','14.11','14.12.1','14.13.1','14.13.1A','14.13.2','14.14.a','14.14.b','14.14.c','15.1.1.a','15.1.1.b','15.1.2.a','15.1.2.b','15.1.2.c','15.1.2.d','15.2','15.3.a','15.3.b','15.3.c','15.3.d','15.4.1','15.4.2','15.4.3','17.1','17.2.1','17.2.2','17.2.3','17.2.4','17.3.1','17.3.2','17.3.2','17.3.3','17.3.4','17.3.5','17.4.1','17.4.2','17.4.3','17.4.4','17.4.5','17.4.6','17.4.7','17.4.8','17.4.9(a)','17.4.9(b)','17.4.9(c)','17.4.9(d)','17.4.9(e)','17.4.9(f)','17.4.10','17.4.11','17.4.12','17.4.13(a)','17.4.13(b)','17.4.13(c)']

plt.figure(figsize = (18,8))
for name in names:
    name_plot('Nagpur', name)
plt.legend(names)
           
########################################################           
   
date = pd.to_datetime(IndicatorNo_df1['Month'], format="%b-%Y")
date = pd.DataFrame(date, columns = ['Month'])
df_data = IndicatorNo_df1[['SNo','Indicators']]
df_data1 = IndicatorNo_df1['Monthly Reports']
IndicatorNo_df2 = pd.concat([df_data, date, df_data1], axis=1)
IndicatorNo_df2.dtypes
IndicatorNo_df2.shape
#IndicatorNo_df1.loc[IndicatorNo_df1['SNo'] == '1.1']
all_names_index = IndicatorNo_df2.set_index(['SNo','Indicators','Month']).sort_index()
       
def name_plot(SNo, Indicators):
    data = all_names_index.loc[SNo, Indicators]
    plt.plot(data.index, data.values)

plt.figure(figsize = (14, 4))
plt.title('Total number of pregnant women Registered for ANC')
name_plot('1.1','Total number of pregnant women Registered for ANC')

plt.figure(figsize = (14, 4))
plt.title('Total Number of NSV or Conventional Vasectomy conducted at Public facilities i.e. PHC, CHC, SDH, DH and other State owned public institutions (sum of items from 9.1.1.a to 9.1.1.d)')
name_plot('9.1.1.e','Total Number of NSV or Conventional Vasectomy conducted at Public facilities i.e. PHC, CHC, SDH, DH and other State owned public institutions (sum of items from 9.1.1.a to 9.1.1.d)')


#plt.figure(figsize = (14, 4))
#plt.title('Number of Infants (0 to 11 months old) received Pentavalent1 immunisation')
#name_plot('10.1.04A','Number of Infants (0 to 11 months old) received Pentavalent1 immunisation')

plt.figure(figsize = (14, 4))
plt.title('Number of Infants (0 to 11 months old) received Pentavalent2 immunisation')
name_plot('10.1.04B','Number of Infants (0 to 11 months old) received Pentavalent2 immunisation')

plt.figure(figsize = (14, 4))
plt.title('Number of Infants (0 to 11 months old) received Pentavalent3 immunisation')
name_plot('10.1.04C','Number of Infants (0 to 11 months old) received Pentavalent3 immunisation')

plt.figure(figsize = (14, 4))
plt.title('Number of Mini-lap sterilizations conducted at PHCs')
name_plot('9.3.1.a','Number of Mini-lap sterilizations conducted at PHCs')

plt.figure(figsize = (14, 4))
plt.title('Total Number of Mini-lap sterilizations conducted at Public facilities i.e. PHC, CHC, SDH, DH and other State owned public institutions (sum of items from 9.3.1.a to 9.3.1.d)')
name_plot('9.3.1.e','Total Number of Mini-lap sterilizations conducted at Public facilities i.e. PHC, CHC, SDH, DH and other State owned public institutions (sum of items from 9.3.1.a to 9.3.1.d)')

plt.figure(figsize = (14, 4))
plt.title('Number of facilities having a Rogi Kalyan Samiti (RKS)')
name_plot('14.04','Number of facilities having a Rogi Kalyan Samiti (RKS)')

plt.figure(figsize = (14, 4))
plt.title('Number of Infants (more than 16 months old) received DPT Booster dose')
name_plot('10.2.1','Number of Infants (more than 16 months old) received DPT Booster dose')


plt.figure(figsize = (14, 4))
plt.title('Number of Infants (more than 16 months old) received OPV Booster dose')
name_plot('10.2.2','Number of Infants (more than 16 months old) received OPV Booster dose')

plt.figure(figsize = (14, 4))
plt.title('Number of children (more than 16 years old) given TT16')
name_plot('10.3.4','Number of children (more than 16 years old) given TT16')


############using 
date = pd.to_datetime(IndicatorNo_df1['Month'], format="%b-%Y")
date = pd.DataFrame(date, columns = ['Month'])
df_data4 = IndicatorNo_df1[['Name of District','Indicators']]
df_data5 = IndicatorNo_df1['Monthly Reports']
IndicatorNo_df2 = pd.concat([df_data4, date, df_data5], axis=1)
IndicatorNo_df2.dtypes
IndicatorNo_df2 = IndicatorNo_df2.rename_axis({"Name of District":"Name_of_District"}, axis="columns") 
all_names_index1 = IndicatorNo_df2.set_index(['Name_of_District','Indicators','Month']).sort_index()

names1 = ['Number of NSV (No Scalpel Vasectomy) or Conventional Vasectomy conducted at PHCs',
         'Number of NSV or Conventional Vasectomy conducted at CHCs',
         'Number of NSV or Conventional Vasectomy conducted at SDHs or DHs',
         'Number of NSV or Conventional Vasectomy conducted at other State owned public institutions',
         'Total Number of NSV or Conventional Vasectomy conducted at Public facilities i.e. PHC, CHC, SDH, DH and other State owned public institutions (sum of items from 9.1.1.a to 9.1.1.d)'
        ]
names2 = ['Number of Infants (0 to 11 months old) received DPT1 immunisation',
         'Number of Infants (0 to 11 months old) received DPT2 immunisation',
         'Number of Infants (0 to 11 months old) received DPT3 immunisation']
  
names3 = ['Number of Infants (0 to 11 months old) received Pentavalent1 immunisation',
         'Number of Infants (0 to 11 months old) received Pentavalent2 immunisation',
         'Number of Infants (0 to 11 months old) received Pentavalent3 immunisation']

names4 = ['Number of Mini-lap sterilizations conducted at PHCs',
          'Number of Mini-lap sterilizations conducted at CHCs',
          'Number of Mini-lap sterilizations conducted at SDHs or DHs',
          'Number of Mini-lap sterilizations conducted at other State owned public institutions',
          'Total Number of Mini-lap sterilizations conducted at Public facilities i.e. PHC, CHC, SDH, DH and other State owned public institutions (sum of items from 9.3.1.a to 9.3.1.d)'     
          ]         

##Reshaping data     
#NSV1 = IndicatorNo_df1.loc[IndicatorNo_df1['SNo'] == '9.1.1.b']
def name_plot1(Name_of_District, Indicators):
    data = all_names_index1.loc[Name_of_District, Indicators]
    plt.plot(data.index, data.values)
    
plt.figure(figsize = (18,8))
for name1 in names1:
    name_plot1('Nagpur', name1)
plt.legend(names1)

plt.figure(figsize = (16,6))
for name2 in names2:
    name_plot1('Nagpur', name2)
plt.legend(names2)

plt.figure(figsize = (18,8))
for name3 in names3:
    name_plot1('Nagpur', name3)
plt.legend(names3)

plt.figure(figsize = (18,8))
for name4 in names4:
    name_plot1('Nagpur', name4)
plt.legend(names4)
#############Total number of pregnant women Registered for ANC data###################
#visualising important trends center wise
#Extracting Total number of pregnant women Registered for ANC data
ANC = melted.loc[melted['SNo'] == '1.1']
ANC1 = ANC['Monthly Reports'].groupby([ANC["Name of Health Block"],ANC["Health Facility"],ANC["Indicators"],ANC['Month']])
ANC_sum = ANC1.sum()
ANC2 = pd.DataFrame(ANC_sum)
ANC2 = ANC2.reset_index(level=["Name of Health Block",'Health Facility',"Indicators", 'Month'])
ANC3 = ANC2['Monthly Reports'].groupby(ANC2["Health Facility"])
ANC3 = ANC3.sum()
ANC3 = pd.DataFrame(ANC3)
ANC3 = ANC3.reset_index(level=['Health Facility'])
colors = ['#F0F8FF', '#FAEBD7','#00FFFF']
plt.pie(
    ANC3['Monthly Reports'],
    labels=ANC3['Health Facility'],
    shadow=False,
    colors=colors,
    explode=(0, 0, 0.15),
    startangle=90,
    autopct='%1.1f%%',
    )
plt.axis('equal')
plt.tight_layout()
plt.show()
#visualising ANC report in block level
ANC4 = ANC2['Monthly Reports'].groupby([ANC2["Name of Health Block"],ANC2["Month"]])
ANC4 = ANC4.sum()
ANC4 = pd.DataFrame(ANC4)
ANC4 = ANC4['Monthly Reports'].groupby(ANC4["Name of Health Block"])
ANC4 = ANC4.sum()
ANC4 = pd.DataFrame(ANC4)
ANC4 = ANC4.reset_index(level=['Name of Health Block'])
colors = ['#F0F8FF', '#FAEBD7','#00FFFF','#FFA500','#000080','#800000','#FFFFE0','#87CEFA','#FFB6C1','#7CFC00','#FFFFF0','#FFFACD','#808080']
plt.pie(
    ANC4['Monthly Reports'],
    labels=ANC4['Name of Health Block'],
    shadow=False,
    colors=colors,
    explode=(0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0.1, 0, 0, 0.15),
    startangle=90,
    autopct='%1.1f%%',
    )
plt.axis('equal')
plt.tight_layout()
plt.show()


###Total Number of NSV or Conventional Vasectomy conducted at Public facilities i.e. PHC, CHC, SDH, DH and other State owned public institutions (sum of items from 9.1.1.a to 9.1.1.d)############
#Estimating & Eliminating Trend
#reshaping
NSV1 = IndicatorNo_df1.loc[IndicatorNo_df1['SNo'] == '9.1.1.e']
date = pd.to_datetime(NSV1['Month'], format="%b-%Y")
date = pd.DataFrame(date, columns = ['Month'])
Monthly = NSV1['Monthly Reports']
NSV1 = pd.concat([date, Monthly], axis=1)
NSV1.set_index(["Month"],inplace=True)
NSV1.index
NSV = NSV1['Monthly Reports']
#ARIMA model
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
NSV.plot()
pyplot.show()
autocorrelation_plot(NSV)
pyplot.show()
# fit model  
model = ARIMA(NSV, order=(2,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())
#Rolling Forecast ARIMA Model
X = NSV.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(2,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='green')
pyplot.show()

#block wise visualisation NSV oror Conventional Vasectomy conducted at Public facilities i.e.  CHC, SDH, DH and other State owned public institutions very low reports recorded.######
NSV2 = df1.loc[df1['Indicators'] == 'Total Number of NSV or Conventional Vasectomy conducted at Public facilities i.e. PHC, CHC, SDH, DH and other State owned public institutions (sum of items from 9.1.1.a to 9.1.1.d)']
NSV3 = NSV2['Total Report'].groupby(NSV2["Name of Health Block"])
NSV3 = NSV2.sum()
NSV3 = pd.DataFrame(NSV3)
NSV3.plot(kind='bar',alpha=0.75)
plt.xlabel("")
###center wise
NSV4 = NSV2['Total Report'].groupby(NSV2["Health Facility"])
NSV4 = NSV4.sum()
NSV4 = NSV4.reset_index(level=['Health Facility'])
colors = ['#F0F8FF','#008000', '#FAEBD7']
plt.pie(
    NSV4['Total Report'],
    labels=NSV4['Health Facility'],
    shadow=False,
    colors=colors,
    explode=(0, 0, 0.15),
    startangle=90,
    autopct='%1.1f%%',
    )
plt.axis('equal')
plt.tight_layout()
plt.show()

###############pentavalent1, 2 &3 immunisation recieved by infants##############
Penta1 = melted.loc[melted['SNo'] == '10.1.04A']
Penta2 = melted.loc[melted['SNo'] == '10.1.04B']
Penta3 = melted.loc[melted['SNo'] == '10.1.04C']
Penta1 = Penta1['Monthly Reports'].groupby([Penta1["Name of Health Block"],Penta1["Health Facility"],Penta1["Indicators"],Penta1['Month']])
Penta2 = Penta2['Monthly Reports'].groupby([Penta2["Name of Health Block"],Penta2["Health Facility"],Penta2["Indicators"],Penta2['Month']])
Penta3 = Penta3['Monthly Reports'].groupby([Penta3["Name of Health Block"],Penta3["Health Facility"],Penta3["Indicators"],Penta3['Month']])
Penta1_s = Penta1.sum()
Penta2_s = Penta2.sum()
Penta3_s = Penta3.sum()
Penta1_s = pd.DataFrame(Penta1_s)
Penta2_s = pd.DataFrame(Penta2_s)
Penta3_s = pd.DataFrame(Penta3_s)
Penta1_s = Penta1_s.reset_index(level=["Name of Health Block",'Health Facility',"Indicators", 'Month'])
Penta2_s = Penta2_s.reset_index(level=["Name of Health Block",'Health Facility',"Indicators", 'Month'])
Penta3_s = Penta3_s.reset_index(level=["Name of Health Block",'Health Facility',"Indicators", 'Month'])

Penta11 = Penta1_s['Monthly Reports'].groupby(Penta1_s["Health Facility"])
Penta11 = Penta11.sum()
Penta11 = pd.DataFrame(Penta11)
Penta11 = Penta11.reset_index(level=['Health Facility'])
Penta22 = Penta2_s['Monthly Reports'].groupby(Penta2_s["Health Facility"])
Penta22 = Penta22.sum()
Penta22 = pd.DataFrame(Penta22)
Penta22 = Penta22.reset_index(level=['Health Facility'])
Penta33 = Penta3_s['Monthly Reports'].groupby(Penta3_s["Health Facility"])
Penta33 = Penta33.sum()
Penta33 = pd.DataFrame(Penta33)
Penta33 = Penta33.reset_index(level=['Health Facility'])

colors = ['#F0F8FF', '#FAEBD7','#00FFFF']
plt.pie(
    Penta11['Monthly Reports'],
    labels=Penta11['Health Facility'],
    shadow=False,
    colors=colors,
    explode=(0, 0, 0.15),
    startangle=90,
    autopct='%1.1f%%',
    )
plt.axis('equal')
plt.tight_layout()
plt.show()

colors = ['#F0F8FF', '#FF0000','#00FFFF']
plt.pie(
    Penta22['Monthly Reports'],
    labels=Penta22['Health Facility'],
    shadow=False,
    colors=colors,
    explode=(0, 0, 0.15),
    startangle=90,
    autopct='%1.1f%%',
    )
plt.axis('equal')
plt.tight_layout()
plt.show()

colors = ['#F0F8FF', '#FAEBD7','#FF0000']
plt.pie(
    Penta33['Monthly Reports'],
    labels=Penta33['Health Facility'],
    shadow=False,
    colors=colors,
    explode=(0, 0, 0.15),
    startangle=90,
    autopct='%1.1f%%',
    )
plt.axis('equal')
plt.tight_layout()
plt.show()
#visualising blocks
df12 = df1.loc[df1['Indicators'] == 'Number of Infants (0 to 11 months old) received Pentavalent1 immunisation']
df13 = df1.loc[df1['Indicators'] == 'Number of Infants (0 to 11 months old) received Pentavalent2 immunisation']
df14 = df1.loc[df1['Indicators'] == 'Number of Infants (0 to 11 months old) received Pentavalent3 immunisation']
Penta4 = df12['Total Report'].groupby(df12["Name of Health Block"])
Penta5 = df13['Total Report'].groupby(df13["Name of Health Block"])
Penta6 = df14['Total Report'].groupby(df14["Name of Health Block"])
Penta4 = Penta4.sum()
Penta5 = Penta5.sum()
Penta6 = Penta6.sum()
Penta4 = pd.DataFrame(Penta4)
Penta5 = pd.DataFrame(Penta5)
Penta6 = pd.DataFrame(Penta6)
Penta4.plot(kind='bar',alpha=0.75)
plt.xlabel("")
Penta5.plot(kind='bar',alpha=0.75)
plt.xlabel("")
Penta6.plot(kind='bar',alpha=0.75)
plt.xlabel("")

#########Minilap sterilisation#######################
mini1 = melted.loc[melted['SNo'] == '9.3.1.a']
mini2 = melted.loc[melted['SNo'] == '9.3.1.e']

mini1 = mini1['Monthly Reports'].groupby([mini1["Name of Health Block"],mini1["Health Facility"],mini1["Indicators"],mini1['Month']])
mini2 = mini2['Monthly Reports'].groupby([mini2["Name of Health Block"],mini2["Health Facility"],mini2["Indicators"],mini2['Month']])
mini1_s = mini1.sum()
mini2_s = mini2.sum()

mini1_s = pd.DataFrame(mini1_s)
mini2_s = pd.DataFrame(mini2_s)

mini1_s = mini1_s.reset_index(level=["Name of Health Block",'Health Facility',"Indicators", 'Month'])
mini2_s = mini2_s.reset_index(level=["Name of Health Block",'Health Facility',"Indicators", 'Month'])

mini11 = mini1_s['Monthly Reports'].groupby(mini1_s["Health Facility"])
mini11 = mini11.sum()
mini11 = pd.DataFrame(mini11)
mini11 = mini11.reset_index(level=['Health Facility'])
mini12 = mini2_s['Monthly Reports'].groupby(mini2_s["Health Facility"])
mini12 = mini12.sum()
mini12 = pd.DataFrame(mini12)
mini12 = mini12.reset_index(level=['Health Facility'])

colors = ['#F0F8FF', '#FAEBD7','#00FFFF']
plt.pie(
    mini11['Monthly Reports'],
    labels=mini11['Health Facility'],
    shadow=False,
    colors=colors,
    explode=(0, 0, 0.15),
    startangle=90,
    autopct='%1.1f%%',
    )
plt.axis('equal')
plt.tight_layout()
plt.show()

colors = ['#F0F8FF', '#FF0000','#00FFFF']
plt.pie(
    mini12['Monthly Reports'],
    labels=mini12['Health Facility'],
    shadow=False,
    colors=colors,
    explode=(0, 0, 0.15),
    startangle=90,
    autopct='%1.1f%%',
    )
plt.axis('equal')
plt.tight_layout()
plt.show()
#######################################################