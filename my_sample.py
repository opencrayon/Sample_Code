#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:38:28 2024

@author: ari
"""

import quandl
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf, ShrunkCovariance

# Set your Quandl API key
quandl.ApiConfig.api_key = ''

months = 12
end_date = pd.to_datetime('2024-02-15').date()  # need date objects
start_date = end_date - pd.DateOffset(months=months)

# We need to define some filterable space
# We want to filter by marketcap (in this experiment), set it up how you'd like
main = quandl.get_table('SHARADAR/DAILY', paginate=True, date=start_date)

main2 = main[(main['marketcap'] >= 100000) & (main['marketcap'] <= 50000000)].ticker.unique()
# And we fix our set of tickers

log_returns = pd.DataFrame()

def set_returns(tickers, start_date, end_date):
    return_frame = pd.DataFrame()
    # Loop through our tickers
    for ticker in tqdm(tickers, desc='pulling data and processing'):
        # Fetch the daily adjusted close prices for the ticker over the 12-month period
        data = quandl.get_table('SHARADAR/SEP', ticker=ticker,
                                qopts={'columns': ['date', 'open', 'closeadj']},
                                date={'gte': start_date, 'lte': end_date},
                                paginate=True)
        
        # Ensure the data is sorted by date
        data.sort_values(by='date', inplace=True)
    
        # Calculate log returns
        data['log_return'] = data['closeadj'] - data['open']
        
        # data['log_return'] = (data['log_return'] - data['log_return'].mean()) / data['log_return'].std()
        # data['log_return'] = data['log_return'].rank(method='average')
        # min_rank = data['log_return'].min()
        # max_rank = data['log_return'].max()
        # data['log_return'] = (data['log_return'] - min_rank) / (max_rank - min_rank)
        
        # Rename the log_return column to the ticker name for clarity
        data.rename(columns={'log_return': f'{ticker}'}, inplace=True)
        
        # Drop all other columns except for the date and the log return
        data = data[['date', f'{ticker}']]
        # data.rename({f"{ticker}_log_return": f'{ticker}'})
        # Merge the log returns into the consolidated DataFrame
        if return_frame.empty:
            return_frame = data
        else:
            return_frame = pd.merge(return_frame, data, on='date', how='outer')
    return return_frame

def return_maker(start_date, end_date, tickers):
    returns = set_returns(tickers, start_date, end_date)
    returns.drop('date', axis=1, inplace=True)
    returns /= returns.std()
    returns.fillna(method='ffill', inplace=True)
    returns.dropna(inplace=True)
    return returns

print(start_date)
print(end_date)
print(main2)
returns = set_returns(main2, start_date, end_date)
returns.drop('date', axis=1, inplace=True)
# Let's normalize this
returns /= returns.std()

# Forward-fill any missing values
returns.fillna(method='ffill', inplace=True)

# Drop any rows with NaN values remaining (like the first row where there's no previous day's price)
returns.dropna(inplace=True)

tickers = returns.columns.tolist()

def sector_based_partial(tickers, alpha, returns):
    '''
    Function iterates a defined sector-based ticker interaction
    In this case, classify all tickers according to sectors and industries
    Then based on their own sector-based spaces, define partial correlations where significant
    using Glasso (Tibshirani, Hastie, Friedman)

    Parameters
    ----------
    tickers : list of tickers
    alpha : The shrinkage parameter (lasso)

    Returns
    -------
    partial_correlations : dictionary of partial correlation matrices
    ind_sect_dict : dictionary of sectors/industries with ticker lists attached
    '''
    
    answers = quandl.get_table('SHARADAR/TICKERS',
                               ticker=list(tickers),
                               qopts={'columns': ['ticker', 'sicsector', 'sicindustry']})
    
    ind_sect_dict = {}
    
    # We seek to define clustered tickers
    # So that we can define partial correlations without overfitting
    # Serious multicollinearity
    drop_list = []
    
    for sector in answers['sicsector'].unique():
        
        sector_data = answers[answers['sicsector'] == sector]
        
        # If sector size is small enough for consideration
        if len(sector_data) < 40:
            ind_sect_dict[sector] = list(set(sector_data['ticker']))
        else:
            drop_list = []
            # For larger sectors, iterate through each industry
            for industry in sector_data['sicindustry'].unique():
                print(industry)
                industry_data = sector_data[sector_data['sicindustry'] == industry]
                
                if len(list(set(industry_data.ticker))) > 2: 
                    
                    # Add tickers by industry to the dictionary, using a set to avoid duplicates
                    if industry not in ind_sect_dict:
                        ind_sect_dict[industry] = list(set(industry_data['ticker']))
                    else:
                        # If the industry is already in the dictionary, update it by adding new tickers
                        existing_tickers = set(ind_sect_dict[industry])
                        new_tickers = set(industry_data['ticker'])
                        ind_sect_dict[industry] = list(existing_tickers.union(new_tickers))
                else:
                    print('no')
                    drop_list += list(set(industry_data['ticker']))
                                                 
            ind_sect_dict[f"{sector}_miscellaneous"] = drop_list
    
    partial_correlations = {}
    print(ind_sect_dict, 'this is my dictionary')
    for keys in ind_sect_dict.keys():
    
        stocks = ind_sect_dict[keys]
        if len(stocks) > 1:
            rets = returns[stocks]
            print(rets)
            model = GraphicalLasso(alpha=alpha)
            model.fit(rets) 
            
            # Extract the estimated precision matrix
            precision_matrix = model.precision_
            
            diagonal = np.sqrt(np.outer(np.diag(precision_matrix), np.diag(precision_matrix)))
            
            partial_corr = -precision_matrix / diagonal
            np.fill_diagonal(partial_corr, 1)
            # Store the partial correlation matrix in the dictionary
            partial_correlations[keys] = partial_corr
        else:
            pass
    return partial_correlations, ind_sect_dict

def ledoit_wolf_correlation(returns):
    '''
    We use the Ledoit-Wolf covariance matrix (Honey, I shrunk the covariance matrix),
    to create a sparse correlation matrix, and select the top 10% of edges
    
    This is returned in the form of an adjacency matrix which will be defined over the graph

    Parameters
    ----------
    returns : DataFrame of returns
    
    Returns
    -------
    adjacency : Adjacency matrix determining graph links
    count_list : counts of nodes
    cov : covariance estimator
    '''
    
    # Use the Ledoit-Wolf estimator, and fit it to returns
    cov = LedoitWolf(assume_centered=True)
    covar = cov.fit(returns).covariance_
    std_devs = np.sqrt(np.diag(covar))

    # Step 2: Create a diagonal matrix with the inverse of the standard deviations
    D_inv = np.diag(1 / std_devs)
    
    # Use this inverse diagonal to compute the correlation matrix
    correlation_matrix = D_inv @ covar @ D_inv

    # Step 1: Set the diagonal elements to NaN so they are not included in the distribution
    np.fill_diagonal(correlation_matrix, np.nan)

    # Flatten the matrix to a 1D array and remove NaN values, we sort the values to get the
    # find the decile we want to restrict our adjacency matrix to
    off_diagonal_elements = correlation_matrix.flatten()
    off_diagonal_elements = off_diagonal_elements[~np.isnan(off_diagonal_elements)]
    off_diagonal_elements.sort()
    
    # Step 3: Plot the distribution of off-diagonal elements
    plt.figure(figsize=(10, 6))
    plt.hist(off_diagonal_elements, bins=50, edgecolor='k', alpha=0.7)
    plt.title('Distribution of Correlation Coefficients (Excluding Diagonals)')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    top10_threshold = off_diagonal_elements[int(off_diagonal_elements.shape[0] * 0.9)]
    # Create a copy of the correlation matrix to apply the threshold
    top10_matrix = correlation_matrix.copy()
    
    # Apply the condition: retain only the top decile correlations
    top10_matrix[top10_matrix <= top10_threshold] = 0
    adjacency = top10_matrix.copy()
    
    # Adjacency matrix occurs for non-zeros
    adjacency[adjacency != 0] = 1
    
    # Gather counts, these will be node degrees
    count_list = [np.sum(adjacency[x, :]) for x in range(adjacency.shape[0])]
    
    return adjacency, count_list, cov