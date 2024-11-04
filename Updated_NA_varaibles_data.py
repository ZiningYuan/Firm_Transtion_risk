import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

## dependent variables - monthly returns 
file_path = '/Users/ziningyuan/Library/CloudStorage/Dropbox/CVTARClimateFinance/Data/full_range_data/full_range_monthly_stock_NA.csv'
#file_path = '/Users/zinin/Dropbox/CVTARClimateFinance/Data/full_range_data/full_range_monthly_stock_NA.csv'
monthly_stock =  pd.read_csv(file_path)
monthly_stock['datadate'] = pd.to_datetime(monthly_stock['datadate'], format='%d/%m/%Y')
monthly_stock['year-month'] = monthly_stock['datadate'].dt.to_period('M')
monthly_filtered = monthly_stock[monthly_stock["secstat"] == "A"]
monthly_filtered = monthly_filtered[['gvkey', 'year-month', 'prccm', 'trt1m', 'cshom']]
##### I noticed some mis-calculation of the data
monthly_filtered['prccm_lag'] = monthly_filtered.groupby('gvkey')['prccm'].shift(1)
def fill_trt1m(row, previous_prccm):
    if pd.isna(row['trt1m']) and not pd.isna(row['prccm']):
        if previous_prccm != 0:
            return ((row['prccm'] - previous_prccm) / previous_prccm)* 100
    return row['trt1m']
monthly_filtered['trt1m'] = monthly_filtered.apply(
    lambda row: fill_trt1m(row, row['prccm_lag']), axis=1
)



## control variables - annual
file_path = '/Users/ziningyuan/Library/CloudStorage/Dropbox/CVTARClimateFinance/Data/full_range_data/full_range_controls_NA_tic.csv'
#file_path = '/Users/zinin/Dropbox/CVTARClimateFinance/Data/full_range_data/full_range_controls_NA_tic.csv'
annual_controls = pd.read_csv(file_path)
#### there are some duplicates for some firms, mainly due to they report their financial statements in both INDUSTRY standards and FS standard
#### I will keep the ones with INDUSTRY standards
annual_controls = annual_controls[annual_controls["indfmt"] == "INDL"]
annual_controls['datadate'] = pd.to_datetime(annual_controls['datadate'])
group_annual_controls = annual_controls.groupby(["gvkey", "fyear"])
group_annual_controls_df =group_annual_controls.first().reset_index()
#### Note that, there are firms whose financial year ends at mid-year, and those end at year-end.
#### I will treat them by calednar year - as it later, when merging with the emissions, the fiscal year will need to be aligned again

### Adding monthly stock to the annual_controls
group_annual_controls_df['fyear'] = group_annual_controls_df['fyear'].astype(int)
group_annual_controls_expanded = group_annual_controls_df.loc[group_annual_controls_df.index.repeat(12)].copy()
group_annual_controls_expanded['month'] = group_annual_controls_expanded.groupby(['gvkey', 'fyear']).cumcount() % 12 + 1

group_annual_controls_expanded['year-month'] = pd.to_datetime(group_annual_controls_expanded['fyear'].astype(str) + '-' + group_annual_controls_expanded['month'].astype(str), format='%Y-%m').dt.to_period('M')
merged_data = pd.merge(group_annual_controls_expanded, monthly_filtered, on=['gvkey', 'year-month'], how='outer')


## B&K independent - emissions
file_path = '/Users/ziningyuan/Library/CloudStorage/Dropbox/CVTARClimateFinance/Data/full_range_data/full_range_emission_NA.csv'
#file_path = '/Users/zinin/Dropbox/CVTARClimateFinance/Data/full_range_data/full_range_emission_NA.csv'
annual_emissions = pd.read_csv(file_path)
annual_emissions['fiscalyear'] = annual_emissions['fiscalyear'].astype(int)
annual_emissions.rename(columns = {
    'di_376883':'GHG_ABS_AR_1',
    'di_319413':'GHG_ABS_1',
    'di_376884':'GHG_ABS_AR_2_loc',
    'di_376886':'GHG_ABS_AR_2_mkt',
    'di_319414':'GHG_ABS_2_loc',
    'di_367750':'GHG_ABS_2_mkt',
    'di_319415':'GHG_ABS_3_up_tot',
    'di_326737':'GHG_ABS_3_down_tot',

    'di_319407':'GHG_INT_1',
    'di_319408':'GHG_INT_2_loc',
    'di_368314':'GHG_INT_2_mkt',
    'di_319409':'GHG_INT_3_up',
    'di_326738':'GHG_INT_3_down_tot',

    'di_319404':'GHG_INT_DIRECT',
    'di_319406':'GHG_INT_INDIRECT',
    'di_319438':'GHG_DIRECT_IR',
    'di_319440':'GHG_INDIRECT_IR',

}, inplace = True)

emission_columns = [
    'GHG_ABS_AR_1', 'GHG_ABS_1', 'GHG_ABS_AR_2_loc', 'GHG_ABS_AR_2_mkt',
    'GHG_ABS_2_loc', 'GHG_ABS_2_mkt', 'GHG_ABS_3_up_tot', 'GHG_ABS_3_down_tot'
]
annual_emissions_all = annual_emissions.sort_values(by=['gvkey', 'fiscalyear']).reset_index(drop=True)

for col in emission_columns:
    annual_emissions_all[f'GR_{col}'] = annual_emissions_all.groupby('gvkey')[col].apply(lambda x: x.pct_change()).reset_index(drop=True)
    annual_emissions_all[f'GR_{col}'].replace([np.inf, -np.inf], np.nan, inplace=True)


merged_data_2 = pd.merge(merged_data, annual_emissions_all, left_on=['gvkey', 'fyear'], right_on=['gvkey', 'fiscalyear'], how='left')
merged_data_2.head(30)


## control variable - HHI
#### Tidy into panel
file_path = '/Users/ziningyuan/Library/CloudStorage/Dropbox/CVTARClimateFinance/Data/full_range_data/full_range_segment_NA.csv'
#file_path = '/Users/zinin/Dropbox/CVTARClimateFinance/Data/full_range_data/full_range_segment_NA.csv'
segment = pd.read_csv(file_path)
segment['datadate'] = pd.to_datetime(segment['datadate'])
segment['srcdate'] = pd.to_datetime(segment['srcdate'])
segment.shape
segment = segment.sort_values(by=['gvkey', 'sid', 'datadate', 'srcdate'], ascending=[True, True, True, False])
segment_data = segment.drop_duplicates(subset=['gvkey', 'sid', 'datadate'], keep='first')
segment_data.shape
segment_data.head(20)
#### Calculation 
segment_bs = segment_data[segment_data['stype'] == "BUSSEG"]
segment_bs['revts'] = pd.to_numeric(segment_bs['revts'], errors='coerce')

df_firm_total_revenue = segment_bs.groupby(['gvkey', 'datadate'])['revts'].sum().reset_index()
df_firm_total_revenue.columns = ['gvkey', 'datadate', 'total_revenue']
segment_bs = pd.merge(segment_bs, df_firm_total_revenue, on=['gvkey', 'datadate'], how='left')
segment_bs['market_share'] = segment_bs['revts'] / segment_bs['total_revenue']
segment_bs['market_share_squared'] = segment_bs['market_share'] ** 2
df_hhi = segment_bs.groupby(['gvkey', 'datadate'])['market_share_squared'].sum().reset_index()
df_hhi.columns = ['gvkey', 'datadate', 'HHI']

df_hhi.head(20)
print(df_hhi["HHI"].describe())
##### THere are some really bizzare values, but I decide to treat them when the data is finalized.

### exact matching
merged_data_3 = pd.merge(merged_data_2, df_hhi[['gvkey', 'datadate', 'HHI']], 
                                on=['gvkey', 'datadate'], how='left')

#manually checking the data if merging was correct
#merged_data_3.to_csv('/Users/ziningyuan/Library/CloudStorage/Dropbox/CVTARClimateFinance/Data/full_range_data/merged_data_raw.csv', index=False)



## control variables - exisiting ratios [for double checking]
print(merged_data_3.head(30))
merged_data_3.columns





## Essential calculations
merged_data_3['cyear'] =  merged_data_3['year-month'].dt.year
merged_data_3['year_end_prccm'] = merged_data_3.groupby(['gvkey', 'cyear'])['prccm'].transform('last')
#merged_data_3['year_end_cshom'] = merged_data_3.groupby(['gvkey', 'cyear'])['cshom'].transform('last')
merged_data_3['market_cap'] = merged_data_3['year_end_prccm'] * merged_data_3['cshpri']



### annual firm controls
##### Calculate Log Market Cap
merged_data_3['LOGSIZE'] =np.where(merged_data_3['market_cap'] > 0, np.log(merged_data_3['market_cap']), np.nan)
merged_data_3['LOGSIZE'].mean()
##### Calculate B/M
merged_data_3['BM'] = merged_data_3['seq'] / merged_data_3['market_cap']
##### Calculate Leverage
####### (dlc / ebitda)
merged_data_3['LEVERAGE_c'] = merged_data_3['dlc'] / merged_data_3['ebitda']
####### (book value of debt = long-term debt + notes payable + current portion of long-term debt) / book value of asset = at
merged_data_3['at'].replace(0, np.nan, inplace=True)
merged_data_3['LEVERAGE'] = (merged_data_3['dt']+ merged_data_3['np']) / merged_data_3['at']
merged_data_3['LEVERAGE'].replace([np.inf, -np.inf], np.nan, inplace=True)
##### Calculate ROE (Yearly Net INcome / Shareholders' Equity) (in %)

merged_data_3['seq'].replace(0, np.nan, inplace=True)
merged_data_3['ROE'] = merged_data_3['ni'] / merged_data_3['seq']
merged_data_3['ROE'].replace([np.inf, -np.inf], np.nan, inplace=True)
merged_data_3['ROE'] = merged_data_3['ROE']*100
##### Calculate INVEST/A = capitl ependitures / the book value of its asset
merged_data_3['INVESTA'] = merged_data_3['capx'] / merged_data_3['at']
##### Calculate LOG PPE
merged_data_3['LOGPPE'] = np.where(merged_data_3['ppent'] > 0, np.log(merged_data_3['ppent']), np.nan)





### monthly firm controls 
##### Calculate MOM : the average of most recnet 12 months' returns
merged_data_3['trt1m_abs'] = merged_data_3['trt1m']/100
merged_data_3['MOM_MA'] = merged_data_3.groupby('gvkey')['trt1m_abs'].apply(
    lambda x: x.shift(1).rolling(window=12, min_periods=12).mean()).reset_index(level=0, drop=True)
    ##### might need to * 100 later
##### Calculate MOM : the cumulative returns of the  most recnet 12 months' returns
merged_data_3['MOM_CUM'] = merged_data_3.groupby('gvkey')['trt1m_abs'].apply(
    lambda x: (1 + x.shift(1)).rolling(window=12, min_periods=12).apply(lambda y: y.prod() - 1)
).reset_index(level=0, drop=True)


##### Calculate VOLATI: the standard deviation of past 12 months' returns
merged_data_3['VOLAT_abs'] = merged_data_3.groupby('gvkey')['trt1m_abs'].apply(
    lambda x: x.shift(1).rolling(window=12, min_periods=12).std()).reset_index(level=0, drop=True)
merged_data_3['VOLAT'] = merged_data_3.groupby('gvkey')['trt1m'].apply(
    lambda x: x.shift(1).rolling(window=12, min_periods=12).std()).reset_index(level=0, drop=True)

#merged_data_3['MOM_MA'].mean()
#merged_data_3['MOM_CUM'].mean()
#merged_data_3['trt1m_abs'].std()
#merged_data_3['VOLAT_abs'].mean()
#merged_data_3['VOLAT'].mean()


### annual and monthly combined 
merged_data_3 = merged_data_3.sort_values(by=['gvkey', 'year-month']).reset_index(drop=True)
##### Calculate SALESGR
####### step 1: monthly market cap
###### I spotted some missing values of the monthly oustanding shares
###### To fill in, I would use the shares outstanding of that year if it's not NA
def fill_cshom(row):
    if pd.isna(row['cshom']) and not pd.isna(row['cshpri']):
        return row['cshpri']
    return row['cshom']
merged_data_3['cshom'] = merged_data_3.apply(fill_cshom, axis=1)

merged_data_3['monthly_market_cap'] = merged_data_3['prccm'] * merged_data_3['cshom']
####### step 2: shift 1 -> lat month's market cap
merged_data_3['last_month_market_cap'] = merged_data_3.groupby('gvkey')['monthly_market_cap'].shift(1)
####### step 3: dollar change in annual firm revenues 
merged_data_3['lag_revenue'] = merged_data_3.groupby('gvkey')['revt'].shift(12)  # Shift revenue by 12 months
merged_data_3['revenue_change'] = merged_data_3['revt'] - merged_data_3['lag_revenue']
####### step 4: normalization
merged_data_3['SALESGR'] = merged_data_3['revenue_change'] / merged_data_3['last_month_market_cap']
merged_data_3['SALESGR'].replace([np.inf, -np.inf], np.nan, inplace=True)

##### Calculate EPSGR
merged_data_3['EPS'] = merged_data_3['ni'] / merged_data_3['cshpri']
merged_data_3['lag_EPS'] = merged_data_3.groupby('gvkey')['EPS'].shift(12)
merged_data_3['EPS_change'] = merged_data_3['EPS'] - merged_data_3['lag_EPS']
merged_data_3['EPSGR'] = merged_data_3['EPS_change'] / merged_data_3['year_end_prccm']
merged_data_3['EPSGR'].replace([np.inf, -np.inf], np.nan, inplace=True)


### daily - beta

#file_path = '/Users/ziningyuan/Library/CloudStorage/Dropbox/CVTARClimateFinance/Data/full_range_data/ctvar_index_daily_price.xlsx'
file_path = '/Users/zinin/Dropbox/CVTARClimateFinance/Data/full_range_data/ctvar_index_daily_price.xlsx'
xls = pd.ExcelFile(file_path)
index_price =  pd.read_excel(xls, sheet_name='Sheet1', skiprows=5)
# Rename the columns for better readability
index_price.columns = ['Date', 'SPX', 'DJI', 'CCMP', 'RAY', 'MXWO']
index_price['Date'] = pd.to_datetime(index_price['Date'], errors='coerce')
index_price= index_price.dropna(subset=['Date'])
index_price.head(20)

file_path = '/Users/ziningyuan/Library/CloudStorage/Dropbox/CVTARClimateFinance/Data/full_range_data/full_range_daily_price_NA.csv'
#file_path = '/Users/zinin/Dropbox/CVTARClimateFinance/Data/full_range_data/full_range_daily_price_NA.csv'
daily_price = pd.read_csv(file_path)
daily_price['datadate'] = pd.to_datetime(daily_price['datadate'], format='%Y-%m-%d')

gvkey_tic_counts = daily_price.groupby('gvkey')['tic'].nunique()
gvkeys_with_multiple_tics = gvkey_tic_counts[gvkey_tic_counts > 1]
#### Since there are some with multiple tickers, to ease the confusion later on when merging to the final dataset, we would do it by ticker
daily_price_pivot_tic = daily_price.pivot(index='datadate', columns='tic', values='prccd')
merged_data_beta = pd.merge(index_price, daily_price_pivot_tic, left_on='Date', right_index=True, how='left')
merged_data_beta.head(10)

## Calculation
daily_returns = merged_data_beta.set_index('Date').pct_change()
####### To avoid errors, we get rid of any return that is higher than 1
daily_returns = daily_returns.applymap(lambda x: np.nan if abs(x) > 1 else x)
daily_returns['Year'] = daily_returns.index.year

def calculate_beta(firm_returns, market_returns):
    # Add constant to market returns for intercept
    X = sm.add_constant(market_returns)
    model = sm.OLS(firm_returns, X, missing='drop').fit()
    return model.params[1]  

tickers_in_use = merged_data_3['tic'].unique()
tickers_to_use = [col for col in merged_data_beta.columns if col in tickers_in_use]
columns_to_keep = ['Date', 'SPX', 'DJI', 'CCMP', 'RAY', 'MXWO'] + list(tickers_to_use)
filtered_merged_data_beta = merged_data_beta[columns_to_keep]


market_indices = ['SPX', 'DJI', 'CCMP', 'RAY', 'MXWO']
yearly_betas = pd.DataFrame(columns=['Ticker', 'Year', 'Market Index', 'Beta'])
# for ticker in tickers_to_use:
#     for year in daily_returns['Year'].unique():
#         for market_index in market_indices:
#             # Filter data for the specific year and ticker
#             yearly_data = daily_returns[daily_returns['Year'] == year]
#             firm_returns = yearly_data[ticker]
#             market_returns = yearly_data[market_index]
            
#             if firm_returns.dropna().shape[0] > 0 and market_returns.dropna().shape[0] > 0:
#                 # Step 4: Calculate beta using OLS regression
#                 beta = calculate_beta(firm_returns, market_returns)
#                 # Append the result to the DataFrame
#                 yearly_betas = yearly_betas.append({
#                     'Ticker': ticker,
#                     'Year': year,
#                     'Market Index': market_index,
#                     'Beta': beta
#                 }, ignore_index=True)




beta_results = []

for ticker in tickers_to_use:
    for year in daily_returns['Year'].unique():
        for market_index in market_indices:
            # Filter data for the specific year and ticker
            yearly_data = daily_returns[daily_returns['Year'] == year]
            firm_returns = yearly_data[ticker]
            market_returns = yearly_data[market_index]
            
            if firm_returns.dropna().shape[0] > 0 and market_returns.dropna().shape[0] > 0:
                # Step 4: Calculate beta using OLS regression
                beta = calculate_beta(firm_returns, market_returns)
                
                # Store the result as a dictionary
                beta_results.append({
                    'Ticker': ticker,
                    'Year': year,
                    'Market Index': market_index,
                    'Beta': beta
                })

# Convert the list of dictionaries to a DataFrame
yearly_betas = pd.DataFrame(beta_results)
yearly_betas.head(20)
#yearly_betas.to_csv('/Users/zinin/Dropbox/CVTARClimateFinance/Data/full_range_data/market_beta.csv', index=False)

### append
#yearly_betas = pd.read_csv('/Users/zinin/Dropbox/CVTARClimateFinance/Data/full_range_data/market_beta.csv')
yearly_betas = pd.read_csv('/Users/ziningyuan/Library/CloudStorage/Dropbox/CVTARClimateFinance/Data/full_range_data/market_beta.csv')
yearly_betas.rename(columns={'Ticker': 'tic', 'Year': 'cyear'}, inplace=True)
merged_data_4 = pd.merge(merged_data_3, yearly_betas, on=['tic', 'cyear'], how='left')
merged_data_4.head(20)
merged_data_4_SPX = merged_data_4[merged_data_4["Market Index"] == "SPX"]
merged_data_4_DJI = merged_data_4[merged_data_4["Market Index"] == "DJI"]
merged_data_4_CCMP = merged_data_4[merged_data_4["Market Index"] == "CCMP"]
merged_data_4_RAY = merged_data_4[merged_data_4["Market Index"] == "RAY"]
merged_data_4_MXWO = merged_data_4[merged_data_4["Market Index"] == "MXWO"]


print(merged_data_4_SPX.columns)


### Create lagged columns for controls 

###### lagged by one year
merged_data_4_SPX = merged_data_4_SPX.sort_values(by=['gvkey', 'cyear']).reset_index(drop=True)
lag_columns_1 = ['LOGSIZE', 'LEVERAGE_c', 'BM', 'LEVERAGE', 'ROE', 'INVESTA', 'LOGPPE', 'Beta' ]
for col in lag_columns_1:
    merged_data_4_SPX[f'{col}_lag'] = merged_data_4_SPX.groupby('gvkey')[col].shift(12)

##### lagged by one month
lag_columns_2 = ['MOM_CUM', 'MOM_MA', 'VOLAT', 'VOLAT_abs', "SALESGR", "EPSGR"]
for col in lag_columns_2:
    merged_data_4_SPX[f'{col}_lag'] = merged_data_4_SPX.groupby('gvkey')[col].shift(1)

####merged_data_4.to_csv('/Users/ziningyuan/Library/CloudStorage/Dropbox/CVTARClimateFinance/Data/full_range_data/raw_with_lag.csv', index=False)

merged_data_4_0517_SPX = merged_data_4_SPX[(merged_data_4_SPX['cyear'] >= 2005) & (merged_data_4_SPX['cyear'] <= 2017)]
merged_data_4_2224_SPX = merged_data_4_SPX[merged_data_4_SPX['cyear'] >= 2022]
merged_data_4_0517_SPX.to_csv('/Users/ziningyuan/Library/CloudStorage/Dropbox/CVTARClimateFinance/Data/full_range_data/0517_SPX_recal.csv', index=False)
merged_data_4_2224_SPX.to_csv('/Users/ziningyuan/Library/CloudStorage/Dropbox/CVTARClimateFinance/Data/full_range_data/2224_SPX_recal.csv', index=False)


#### WINSORIZE and final formatting
#BK_data = pd.read_csv('/Users/ziningyuan/Library/CloudStorage/Dropbox/CVTARClimateFinance/Data/full_range_data/0517_SPX_recal.csv')
BK_data = pd.read_csv('/Users/zinin/Dropbox/CVTARClimateFinance/Data/full_range_data/0517_SPX_recal.csv')
#JP_data = pd.read_csv('/Users/ziningyuan/Library/CloudStorage/Dropbox/CVTARClimateFinance/Data/full_range_data/2224_SPX_recal.csv')

BK_data.columns
BK_data = BK_data[['gvkey', 'tic','conm', 'curcd', 'year-month','trt1m',
       'GHG_INT_DIRECT', 'GHG_INT_INDIRECT', 'GHG_INT_1',
       'GHG_INT_2_loc', 'GHG_INT_3_up', 'GHG_ABS_1', 'GHG_ABS_2_loc',
       'GHG_ABS_3_up_tot', 'GHG_DIRECT_IR', 'GHG_INDIRECT_IR',
       'GHG_ABS_3_down_tot', 'GHG_INT_3_down_tot', 'GHG_ABS_AR_1',
       'GHG_ABS_AR_2_loc', 'GHG_ABS_2_mkt',
       'GHG_INT_2_mkt', 'GHG_ABS_AR_2_mkt',
       'country', 'yearfounded', 'GR_GHG_ABS_AR_1',
       'GR_GHG_ABS_1', 'GR_GHG_ABS_AR_2_loc', 'GR_GHG_ABS_AR_2_mkt',
       'GR_GHG_ABS_2_loc', 'GR_GHG_ABS_2_mkt', 'GR_GHG_ABS_3_up_tot',
       'GR_GHG_ABS_3_down_tot', 'HHI', 'cyear',
       'LOGSIZE','LEVERAGE_c', 'BM', 'LEVERAGE', 'ROE',
       'INVESTA', 'LOGPPE', 'MOM_CUM','MOM_MA', 'VOLAT','VOLAT_abs','SALESGR', 'EPSGR', 'Beta','Market Index', 
       'LOGSIZE_lag','LEVERAGE_c_lag', 'BM_lag', 'LEVERAGE_lag', 'ROE_lag',
       'INVESTA_lag', 'LOGPPE_lag', 'MOM_CUM_lag','MOM_MA_lag', 'VOLAT_lag','VOLAT_abs_lag', 'Beta_lag',
       'gind', 'gsector']]

BK_data.shape

##### OPTIONS
##### 1  - use country defined in Compustat  ====== found out this is not reliable as there are many NAs which results in lossing data points
####BK_data = BK_data[BK_data['country'] =='United States'] 

##### 2  -  get a list of tickers in CTVaR then select based on the list
#CTVaR = pd.read_csv('/Users/ziningyuan/Library/CloudStorage/Dropbox/CVTARClimateFinance/Data/CTVaR_FIRM_CHA.csv')
CTVaR = pd.read_csv('/Users/zinin/Dropbox/CVTARClimateFinance/Data/CTVaR_FIRM_CHA.csv')

us_tickers = CTVaR[CTVaR['country_of_risk'] == 'US']['Ticker'].unique()
us_gvkeys = BK_data[BK_data['tic'].isin(us_tickers)]['gvkey'].unique()
BK_data = BK_data[BK_data['gvkey'].isin(us_gvkeys)] 
BK_data.shape

BK_data["gvkey"].nunique()
BK_data["tic"].nunique()

####### To get corresponding ISIN
BK_data_with_isin = pd.merge(BK_data, CTVaR[['Ticker', 'isin']], left_on='tic', right_on='Ticker', how='left')
BK_data_with_isin['isin'].nunique()
isin_ticker_counts = BK_data_with_isin.groupby('tic')['isin'].nunique()

row_mismatch = BK_data_with_isin[BK_data_with_isin['tic'] !=BK_data_with_isin['Ticker']]

BK_data_with_isin_cleaned = BK_data_with_isin.dropna(subset=['isin'])
BK_data_with_isin_cleaned.shape
BK_data_with_isin_cleaned["gvkey"].nunique()

###### TO get the tickers where I can only get from GLOBAL dataset
with open('/Users/zinin/Dropbox/CVTARClimateFinance/Data/CTVaR_Ticker.txt', 'r') as file:
    ctvar_tickers = {line.strip() for line in file.readlines()}
bk_tickers = set(BK_data_with_isin_cleaned['tic'].unique())  
missing_tickers = ctvar_tickers - bk_tickers
with open('/Users/zinin/Dropbox/CVTARClimateFinance/Data/Non_NA_Ticker.txt', 'w') as output_file:
    for ticker in missing_tickers:
        output_file.write(f"{ticker}\n")


###### Similarly tO get the ISIN where I can only get from GLOBAL dataset
with open('/Users/zinin/Dropbox/CVTARClimateFinance/Data/CTVaR_ISIN.txt', 'r') as file:
    ctvar_isins = {line.strip() for line in file.readlines()}
bk_isins = set(BK_data_with_isin_cleaned['isin'].unique())  
missing_isins = ctvar_isins - bk_isins
with open('/Users/zinin/Dropbox/CVTARClimateFinance/Data/Non_NA_ISIN.txt', 'w') as output_file:
    for isin in missing_isins:
        output_file.write(f"{isin}\n")



##### Step 1: Get rid of outliers returns
BK_data_p = BK_data[abs(BK_data['trt1m']) <= 100]
BK_data_p.shape
##### Step 2: normalize (/100) (LOG) if needed
#### special treatment for Log -- ignore negtaice values 
BK_data_p['LOG_GHG_ABS_1'] = np.where(BK_data_p['GHG_ABS_1'] > 0, np.log(BK_data_p['GHG_ABS_1']), np.nan)
BK_data_p['LOG_GHG_ABS_2_loc'] = np.where(BK_data_p['GHG_ABS_2_loc'] > 0, np.log(BK_data_p['GHG_ABS_2_loc']), np.nan)
BK_data_p['LOG_GHG_ABS_2_mkt'] = np.where(BK_data_p['GHG_ABS_2_mkt'] > 0, np.log(BK_data_p['GHG_ABS_2_mkt']), np.nan)
BK_data_p['LOG_GHG_ABS_3_up_tot'] = np.where(BK_data_p['GHG_ABS_3_up_tot'] > 0, np.log(BK_data_p['GHG_ABS_3_up_tot']), np.nan)
BK_data_p['LOG_GHG_ABS_3_down_tot'] = np.where(BK_data_p['GHG_ABS_3_down_tot'] > 0, np.log(BK_data_p['GHG_ABS_3_down_tot']), np.nan)
BK_data_p['LOG_GHG_ABS_AR_1'] = np.where(BK_data_p['GHG_ABS_AR_1'] > 0, np.log(BK_data_p['GHG_ABS_AR_1']), np.nan)
BK_data_p['LOG_GHG_ABS_AR_2_loc'] =np.where(BK_data_p['GHG_ABS_AR_2_loc'] > 0, np.log(BK_data_p['GHG_ABS_AR_2_loc']), np.nan)
BK_data_p['LOG_GHG_ABS_AR_2_mkt'] = np.where(BK_data_p['GHG_ABS_AR_2_mkt'] > 0, np.log(BK_data_p['GHG_ABS_AR_2_mkt']), np.nan)

BK_data_p['GHG_INT_1'] = BK_data_p['GHG_INT_1']/100
BK_data_p['GHG_INT_2_loc'] = BK_data_p['GHG_INT_2_loc'] /100
BK_data_p['GHG_INT_2_mkt'] = BK_data_p['GHG_INT_2_mkt']/100
BK_data_p[ 'GHG_INT_3_up'] = BK_data_p[ 'GHG_INT_3_up'] /100
BK_data_p['GHG_INT_3_down_tot'] = BK_data_p['GHG_INT_3_down_tot']/100
BK_data_p['GHG_INT_2_mkt'] = BK_data_p['GHG_INT_2_mkt']/100
BK_data_p[ 'GHG_INT_DIRECT'] = BK_data_p[ 'GHG_INT_DIRECT']/100
BK_data_p['GHG_INT_INDIRECT'] = BK_data_p['GHG_INT_INDIRECT']/100




##### Step 3: Winsorize
from scipy.stats.mstats import winsorize
def winsorize_series(series, lower_percent, upper_percent):
    return winsorize(series, limits=(lower_percent, upper_percent))
winsorization_levels = {
    'GHG_INT_1': 0.025,  # Winsorize at 2.5%
    'GR_GHG_ABS_AR_1': 0.025,
    'GR_GHG_ABS_1': 0.025, 
    'GR_GHG_ABS_AR_2_loc': 0.025,
    'GR_GHG_ABS_AR_2_mkt': 0.025,
    'GR_GHG_ABS_2_loc': 0.025, 
    'GR_GHG_ABS_2_mkt': 0.025, 
    'GR_GHG_ABS_3_up_tot': 0.025,
    'GR_GHG_ABS_3_down_tot': 0.025,
    'GHG_INT_1': 0.025,
    'GHG_INT_2_loc': 0.025,
    'GHG_INT_2_mkt': 0.025,
    'GHG_INT_3_up': 0.025,
    'GHG_INT_3_down_tot': 0.025,
    'GHG_INT_2_mkt': 0.025,
    'GHG_INT_DIRECT': 0.025,
    'GHG_INT_INDIRECT': 0.025,
    'GHG_DIRECT_IR': 0.025, 
    'GHG_INDIRECT_IR': 0.025,

    'BM': 0.025, 
    'LEVERAGE': 0.025, 
    'LEVERAGE_c':0.025,
    'ROE': 0.025,
    'INVESTA': 0.025,

    'MOM_CUM': 0.005,  # Winsorize at 0.5%
    'MOM_MA': 0.005,
    'VOLAT': 0.005,
    'VOLAT_abs': 0.005,
    'SALESGR': 0.005, 
    'EPSGR': 0.005
}

# Apply winsorization to each column
for column, level in winsorization_levels.items():
    # Winsorize both lower and upper percent of the data
    BK_data_p[column] = winsorize_series(BK_data_p[column], lower_percent=level, upper_percent=level)

##### for different market index
#BK_data_p_SPX = BK_data_p[BK_data_p["Market Index"] == "SPX"]
#BK_data_p_DJI = BK_data_p[BK_data_p["Market Index"] == "DJI"]
##BK_data_p_CCMP = BK_data_p[BK_data_p["Market Index"] == "CCMP"]
#BK_data_p_RAY = BK_data_p[BK_data_p["Market Index"] == "RAY"]
#BK_data_p_MXWO = BK_data_p[BK_data_p["Market Index"] == "MXWO"]


#### Try for one 
BK_data_p.to_csv('/Users/ziningyuan/Library/CloudStorage/Dropbox/CVTARClimateFinance/Data/full_range_data/BK_data_p_SPX_country2.csv')



#### Repeat for the JP year
JP_data.columns
JP_data = JP_data[['gvkey', 'tic','conm', 'curcd', 'year-month','trt1m',
       'GHG_INT_DIRECT', 'GHG_INT_INDIRECT', 'GHG_INT_1',
       'GHG_INT_2_loc', 'GHG_INT_3_up', 'GHG_ABS_1', 'GHG_ABS_2_loc',
       'GHG_ABS_3_up_tot', 'GHG_DIRECT_IR', 'GHG_INDIRECT_IR',
       'GHG_ABS_3_down_tot', 'GHG_INT_3_down_tot', 'GHG_ABS_AR_1',
       'GHG_ABS_AR_2_loc', 'GHG_ABS_2_mkt',
       'GHG_INT_2_mkt', 'GHG_ABS_AR_2_mkt',
       'country', 'yearfounded', 'GR_GHG_ABS_AR_1',
       'GR_GHG_ABS_1', 'GR_GHG_ABS_AR_2_loc', 'GR_GHG_ABS_AR_2_mkt',
       'GR_GHG_ABS_2_loc', 'GR_GHG_ABS_2_mkt', 'GR_GHG_ABS_3_up_tot',
       'GR_GHG_ABS_3_down_tot', 'HHI', 'cyear',
       'LOGSIZE','LEVERAGE_c', 'BM', 'LEVERAGE', 'ROE',
       'INVESTA', 'LOGPPE', 'MOM_CUM','MOM_MA', 'VOLAT','VOLAT_abs','SALESGR', 'EPSGR', 'Beta','Market Index', 
       'LOGSIZE_lag','LEVERAGE_c_lag', 'BM_lag', 'LEVERAGE_lag', 'ROE_lag',
       'INVESTA_lag', 'LOGPPE_lag', 'MOM_CUM_lag','MOM_MA_lag', 'VOLAT_lag','VOLAT_abs_lag', 'Beta_lag',
       'gind', 'gsector']]

JP_data['country'].unique()
#### option 1:
# JP_data = JP_data[JP_data['country'] =='United States'] #### OPTION: NEED TO IDENTIFY WHICH TICKERS ARE FROM US
#### option 2:
JP_data = JP_data[JP_data['tic'].isin(us_tickers)] 
JP_data.shape
##### Step 1: Get rid of outliers returns
JP_data_p = JP_data[abs(JP_data['trt1m']) <= 100]
JP_data_p.shape
##### Step 2: normalize (/100) (LOG) if needed
#### special treatment for Log -- ignore negtaice values 
JP_data_p['LOG_GHG_ABS_1'] = np.where(JP_data_p['GHG_ABS_1'] > 0, np.log(JP_data_p['GHG_ABS_1']), np.nan)
JP_data_p['LOG_GHG_ABS_2_loc'] = np.where(JP_data_p['GHG_ABS_2_loc'] > 0, np.log(JP_data_p['GHG_ABS_2_loc']), np.nan)
JP_data_p['LOG_GHG_ABS_2_mkt'] = np.where(JP_data_p['GHG_ABS_2_mkt'] > 0, np.log(JP_data_p['GHG_ABS_2_mkt']), np.nan)
JP_data_p['LOG_GHG_ABS_3_up_tot'] = np.where(JP_data_p['GHG_ABS_3_up_tot'] > 0, np.log(JP_data_p['GHG_ABS_3_up_tot']), np.nan)
JP_data_p['LOG_GHG_ABS_3_down_tot'] = np.where(JP_data_p['GHG_ABS_3_down_tot'] > 0, np.log(JP_data_p['GHG_ABS_3_down_tot']), np.nan)
JP_data_p['LOG_GHG_ABS_AR_1'] = np.where(JP_data_p['GHG_ABS_AR_1'] > 0, np.log(JP_data_p['GHG_ABS_AR_1']), np.nan)
JP_data_p['LOG_GHG_ABS_AR_2_loc'] =np.where(JP_data_p['GHG_ABS_AR_2_loc'] > 0, np.log(JP_data_p['GHG_ABS_AR_2_loc']), np.nan)
JP_data_p['LOG_GHG_ABS_AR_2_mkt'] = np.where(JP_data_p['GHG_ABS_AR_2_mkt'] > 0, np.log(JP_data_p['GHG_ABS_AR_2_mkt']), np.nan)

JP_data_p['GHG_INT_1'] = JP_data_p['GHG_INT_1']/100
JP_data_p['GHG_INT_2_loc'] = JP_data_p['GHG_INT_2_loc'] /100
JP_data_p['GHG_INT_2_mkt'] = JP_data_p['GHG_INT_2_mkt']/100
JP_data_p[ 'GHG_INT_3_up'] = JP_data_p[ 'GHG_INT_3_up'] /100
JP_data_p['GHG_INT_3_down_tot'] = JP_data_p['GHG_INT_3_down_tot']/100
JP_data_p['GHG_INT_2_mkt'] = JP_data_p['GHG_INT_2_mkt']/100
JP_data_p[ 'GHG_INT_DIRECT'] = JP_data_p[ 'GHG_INT_DIRECT']/100
JP_data_p['GHG_INT_INDIRECT'] = JP_data_p['GHG_INT_INDIRECT']/100

##### Step 3: Winsorize
from scipy.stats.mstats import winsorize
def winsorize_series(series, lower_percent, upper_percent):
    return winsorize(series, limits=(lower_percent, upper_percent))
winsorization_levels = {
    'GHG_INT_1': 0.025,  # Winsorize at 2.5%
    'GR_GHG_ABS_AR_1': 0.025,
    'GR_GHG_ABS_1': 0.025, 
    'GR_GHG_ABS_AR_2_loc': 0.025,
    'GR_GHG_ABS_AR_2_mkt': 0.025,
    'GR_GHG_ABS_2_loc': 0.025, 
    'GR_GHG_ABS_2_mkt': 0.025, 
    'GR_GHG_ABS_3_up_tot': 0.025,
    'GR_GHG_ABS_3_down_tot': 0.025,
    'GHG_INT_1': 0.025,
    'GHG_INT_2_loc': 0.025,
    'GHG_INT_2_mkt': 0.025,
    'GHG_INT_3_up': 0.025,
    'GHG_INT_3_down_tot': 0.025,
    'GHG_INT_2_mkt': 0.025,
    'GHG_INT_DIRECT': 0.025,
    'GHG_INT_INDIRECT': 0.025,
    'GHG_DIRECT_IR': 0.025, 
    'GHG_INDIRECT_IR': 0.025,

    'BM': 0.025, 
    'LEVERAGE': 0.025, 
    'LEVERAGE_c':0.025,
    'ROE': 0.025,
    'INVESTA': 0.025,

    'MOM_CUM': 0.005,  # Winsorize at 0.5%
    'MOM_MA': 0.005,
    'VOLAT': 0.005,
    'VOLAT_abs': 0.005,
    'SALESGR': 0.005, 
    'EPSGR': 0.005
}

# Apply winsorization to each column
for column, level in winsorization_levels.items():
    # Winsorize both lower and upper percent of the data
    JP_data_p[column] = winsorize_series(JP_data_p[column], lower_percent=level, upper_percent=level)

##### for different market index
#JP_data_p_SPX = JP_data_p[JP_data_p["Market Index"] == "SPX"]
#JP_data_p_DJI = JP_data_p[JP_data_p["Market Index"] == "DJI"]
##JP_data_p_CCMP = JP_data_p[JP_data_p["Market Index"] == "CCMP"]
#JP_data_p_RAY = JP_data_p[JP_data_p["Market Index"] == "RAY"]
#JP_data_p_MXWO = JP_data_p[JP_data_p["Market Index"] == "MXWO"]


#### Try for one 
JP_data_p.to_csv('/Users/ziningyuan/Library/CloudStorage/Dropbox/CVTARClimateFinance/Data/full_range_data/JP_data_p_SPX.csv')
