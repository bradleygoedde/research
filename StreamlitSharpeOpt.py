# %% Importing Big Functions
import SharpeOptFunctions as func
#from iexfinance.stocks import get_historical_data
import torch
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.write("""
# Sharpe Ratio Maximization App
developed by Chanwook Park, Stefan Knapik, and Renee Zhuang

Portfolio diversification is a well-known strategy for risk management. However, finding the best way to allocate money when diversifying is not straighrtforward. 
Generally, investors seek to maximize returns while minimizing risk. The Sharpe ratio quantifies the expected return per unit risk of an investment by using volatility as a proxy for risk.

This app lets you choose a group of assets and a time period of data to consider. Then this application will return the optimal distribution of your assets in a portfolio such that the Sharpe ratio is maximized.

### Reading Historical Price Data
""")

def ConvertToTorch(fetches): 

    prices_tensor = torch.zeros(len(fetches), len(fetches[0])-1,  dtype=torch.float32)
    returns_tensor = torch.zeros(len(fetches), (len(fetches[0])-1),  dtype=torch.float32)
    
    for idx in range(len(fetches)):
        
        fetch = fetches[idx]
        price_torch = torch.tensor(fetch.iloc[1:,0], dtype=torch.float32)
        return_torch = torch.tensor(fetch.iloc[1:,1], dtype=torch.float32)
        
        p_torch = torch.reshape(price_torch, (1, len(price_torch)))
        r_torch = torch.reshape(return_torch, (1, len(return_torch)))
        
        prices_tensor[idx,:] = p_torch
        returns_tensor[idx,:] = r_torch

    input_tensor = torch.cat((prices_tensor, returns_tensor), 1)

    return input_tensor, returns_tensor

def read_prices_returns(tickers):
    file_name = 'All_1800_data.xlsx'
    xls = pd.ExcelFile(file_name)

    fetches = []
    for ticker in tickers:
        fetch = pd.read_excel(xls, ticker, index_col = 0)
        fetches.append(fetch.loc[start_date:end_date])
        
    return fetches


# %% Sharpe Optimization to Get Asset Ratios and Portfolio Returns

##################### User Settings #########################
name = st.selectbox(
    "What group of assets would you like to consider for a portfolio?",
    ("Tech Stocks", "Vanguard Sector ETFs" , "Choose My Own"))

# types of tickers
if name =='Tech Stocks':
    tickers = ['AAPL', 'TSLA', 'MSFT', 'GOOG','AMZN'] # tech
elif name == 'Vanguard Sector ETFs':
    tickers = ['VOX', 'VCR', 'VDC', 'VDE', 'VFH', 'VHT', 'VIS', 'VGT', 'VAW', 'VNQ', 'VPU'] # Vanguard ETF for each of the 11 sectors in the stock market 
elif name == 'Choose My Own':
    tickers = st.multiselect(
        'Assets I will consider',
        ['AAPL', 'AMZN', 'BABA', 'COST', 'DIS', 'F', 'FB', 'GOOG', 'MSFT', 'NFLX', 'NKE', 'PEP', 'SPGI', 'TSLA', 'VAW', 'VCR', 'VDC', 'VDE', 'VFH', 'VGT', 'VHT', 'VIS', 'VNQ', 'VOX', 'VPU'])

start_date = st.date_input('Start Date', dt.date(2021,3,24), dt.date(2017,7,4), dt.date(2021,9,17))
end_date = st.date_input('End Date', dt.date(2021,9,24), start_date+dt.timedelta(weeks=1), dt.date(2021,9,24))

st.sidebar.write("""
# Model Parameters
""")

hidden_size = st.sidebar.slider(
    "Size of the hidden layer",
    1,128,64)

num_iter = st.sidebar.slider(
    "Iterations of gradient ascent",
    1,200,100)

learning_rate = st.sidebar.slider(
    "Learning rate for gradient ascent",
    0.000,0.100,.050,0.001,'%f')

startGD = st.sidebar.button("Start Gradient Ascent")

st.sidebar.write("""
#
# Monte Carlo Parameters
""")

num_portfolios = st.sidebar.select_slider(
     'Number of random portfolios to generate',
     [10,100,1000,10000,100000,1000000], value=1000)

startMC = st.sidebar.button("Start Monte Carlo")

print_interval = 40
print_threshhold = 5

############# Load data ##############
fetches = read_prices_returns(tickers)

#Gradient Ascent
if startGD or startMC:
    st.write("""
        ### Performing Gradient Ascent
        """)
    input_tensor, returns_tensor = ConvertToTorch(fetches)
    portfolio_returns, asset_ratios, returns_tensor, sharpe_progression = func.SharpeOpt(input_tensor, returns_tensor, tickers, num_iter, learning_rate, hidden_size, print_interval, print_threshhold)

#%% Sharpe Plots
    fig = plt.figure(1)

    ax = fig.add_subplot()
    ax.plot(sharpe_progression)
    ax.set_xlabel('Iterations of Gradient Ascent', fontsize=16)
    ax.set_ylabel('Objective Function (~ Sharpe Ratio)', fontsize=16)
    plt.tight_layout()
    plt.savefig(name+'_Performance'+'.jpg', dpi=300)

    st.pyplot(plt.figure(1))
#%% Performance Plots
    fig = plt.figure(2)

    portfolio_returns_optim = portfolio_returns
    sharpe = (torch.mean(portfolio_returns_optim)-1)/torch.std(portfolio_returns_optim)
    time = fetches[0].index[1:]
    num_trading_days = len(time)
    num_assets = len(tickers)

    port_value = torch.empty(num_trading_days)
    asset_value = torch.empty(num_assets, num_trading_days)
    for day in range(num_trading_days):
        port_value[day] = (100 * portfolio_returns_optim.detach()[0:day].prod())
        for asset in range(num_assets):
            asset_value[asset, day] = (100 * returns_tensor.detach()[asset, 0:day].prod())


    ax = fig.add_subplot()
    ax.plot(time, port_value, label='Portfolio', color='black', linewidth=5)
    for asset in range(num_assets):
        ax.plot(time, asset_value[asset, :], label=tickers[asset])
    ax.set_xlabel('Trading days', fontsize=16)
    ax.set_ylabel('Price ratio compared with \n the first day (%)', fontsize=16)
    plt.xticks(rotation=45)
    ax.text(0.74,0.05, 'Asset\nTickers:\n'+'\n'.join(tickers), transform=ax.transAxes, ha='left',  va='bottom').set_bbox(dict(fc='whitesmoke',ec='black'))
    ax.text(0.87,0.05, 'Asset\nRatios:\n'+'\n'.join([f'{i:.3f}' for i in asset_ratios.detach().reshape(num_assets).tolist()]), transform=ax.transAxes, ha='left',  va='bottom', ).set_bbox(dict(fc='whitesmoke',ec='black'))

    ax.legend(loc='upper left', facecolor='whitesmoke', edgecolor='black', fontsize=10) 
    plt.tight_layout()
    plt.savefig(name+'_Performance'+'.jpg', dpi=300)
    print(f'Sharpe ratio of optimal portfolio: {sharpe:.2f}')

    st.pyplot(plt.figure(2))
    
    
    fig = plt.figure(3)
    ax = fig.add_subplot()
    plt.title('Ideal Asset Distribution')#,loc='left')
    sizes = []
    for i in asset_ratios.detach().reshape(num_assets).tolist():
        sizes.append(i)
    #explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
    ax.pie(sizes, labels=tickers, autopct='%1.1f%%',
                   shadow=False, startangle=0)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(plt.figure(3))

# %% Monte Carlo Simulation and Plots

if startMC:
    st.write("""
        ### Performing Monte Carlo
        """)
    port_return_means, port_return_stds = func.MonteCarlo(tickers, num_portfolios, returns_tensor)


    # Plot Random Portfolio Data
    fig = plt.figure(7)
    sharpe = (portfolio_returns_optim.mean().item()-1) / portfolio_returns_optim.std().item()

    # plt.rcParams['font.size'] = '10'
    ax = fig.add_subplot()
    y = np.linspace(port_return_means.min()-1,max((port_return_means.max()-1),1.1*(portfolio_returns_optim.detach().mean()-1)),10)
    x = y / sharpe
    ax.scatter(portfolio_returns_optim.detach().std().item(), portfolio_returns_optim.detach().mean().item()-1, color='red', label='Optimized Portfolio', zorder=3)
    ax.scatter(port_return_stds, port_return_means-1, s=1, label=str(num_portfolios) +' Random Portfolios', zorder=2)
    ax.plot(x,y, color='black', zorder=1)
    ax.text(0.74,0.05, 'Asset\nTickers:\n\n'+'\n'.join(tickers), transform=ax.transAxes, ha='left',  va='bottom').set_bbox(dict(fc='whitesmoke',ec='black'))
    ax.text(0.87,0.05, 'Ideal \nRatios:\n\n'+'\n'.join([f'{i:.3f}' for i in asset_ratios.detach().reshape(num_assets).tolist()]), transform=ax.transAxes, ha='left',  va='bottom').set_bbox(dict(fc='whitesmoke',ec='black'))
    ax.set_ylabel('Mean of Daily Returns', fontsize=16)
    ax.set_xlabel('Standard Deviation of Daily Returns', fontsize=16)
    ax.legend(loc='best', facecolor='whitesmoke',edgecolor='black') 
    fig.tight_layout()
    plt.savefig(name+'_MC_' + str(num_portfolios)+'.jpg', dpi=300)
    
    st.pyplot(plt.figure(7))
