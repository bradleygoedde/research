from iexfinance.stocks import get_historical_data
import datetime as dt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def SharpeOpt(input_tensor, returns_tensor, tickers, num_iter, learning_rate, hidden_size, print_interval, print_threshhold):
    
    input_size = returns_tensor.shape[1] # Number of trading days

    # Model
    # Network Architecture
    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(NeuralNet, self).__init__()
            self.l1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.LeakyReLU()
            self.l2 = nn.Linear(hidden_size, 1)
            self.l3 = nn.Linear(hidden_size, hidden_size)
            self.softmax = nn.Softmax(dim=0)
        
        def forward(self, x):
            out = self.l1(x)
            out = self.relu(out)
            # out = self.l3(out)      # option for second layer
            # out = self.relu(out)    # option for second layer
            raw_asset_ratios = self.l2(out)
            asset_ratios = self.softmax(raw_asset_ratios)
            return asset_ratios
    
    class DeepNeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(DeepNeuralNet, self).__init__()
            self.l1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.LeakyReLU()
            self.l2 = nn.Linear(hidden_size, 1)
            self.l3 = nn.Linear(hidden_size, hidden_size)
            self.softmax = nn.Softmax(dim=0)
        
        def forward(self, x):
            out = self.l1(x)
            out = self.relu(out)
            out = self.l3(out)      # option for second layer
            out = self.relu(out)    # option for second layer
            out = self.l3(out)      # option for second layer
            out = self.relu(out)    # option for second layer
            raw_asset_ratios = self.l2(out)
            asset_ratios = self.softmax(raw_asset_ratios)
            return asset_ratios
    
    model = NeuralNet(input_size, hidden_size)
    #model = DeepNeuralNet(input_size, hidden_size)

    # Objective Function and Optimizer
    #Rp = torch.matmul(x,weights))
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # Gradient Ascent
    sharpe_progression = []
    for iter in range(num_iter):
        # Hidden Forward Pass
        asset_ratios = model(returns_tensor) # or could use input_tensor
            
        # Output Layers
        portfolio_returns = torch.matmul(torch.transpose(returns_tensor,0,1), asset_ratios)
        sharpe = (torch.mean(portfolio_returns)-1)/torch.std(portfolio_returns)
        
        # Loss Functoin
        loss = -sharpe
        
        # Computation of Gradients : Backward Pass
        loss.backward()
        
        # Update Model Parameters
        optimizer.step()
        
        # Reset Gradients (Computational Graph)
        optimizer.zero_grad()
        
        # Printing Progress
        sharpe_progression.append(sharpe.detach().numpy())
        pretty_ratios = ''
        pretty_returns = ''
        pretty_returns_annual = ''
        for asset in range(len(asset_ratios)):
            pretty_ratios += f'{asset_ratios[asset].item():3.3f}  '
            pretty_returns += f'{returns_tensor[asset,:].mean():3.5f}  '
            pretty_returns_annual += f'{torch.pow(returns_tensor[asset,:].mean(),252):3.5f}  '
        
        if (iter+1) % print_interval == 0 or iter<print_threshhold:
            print(f'{iter+1:5d}  Sharpe: {sharpe:8.3f}        Asset Ratios:  {pretty_ratios}')
            # print(weights)
            # print(Rp)

    print(f'\nAvg. Daily  Port. Return: {portfolio_returns.mean():.5f}     STD: {portfolio_returns.std():.5f}')
    print(f'Avg. Annual Port. Return: {torch.pow(portfolio_returns.mean(),252):.5f}')
    print(f'Avg. Daily  Asset Return: {pretty_returns}')
    print(f'Avg. Annual Asset Return: {pretty_returns_annual}')

    return portfolio_returns, asset_ratios, returns_tensor, sharpe_progression



#%%
def MonteCarlo(tickers, num_portfolios, returns_tensor):
    returns_tensor.detach()
    
    num_assets = len(tickers)

    # Loop for Random Portfolio Data
    port_return_means = torch.empty(num_portfolios)
    port_return_stds = torch.empty(num_portfolios)
    for portfolio in range(num_portfolios):
        x = torch.rand(num_assets)
        #temp_asset_ratios = x.softmax(0)
        temp_asset_ratios = x/sum(x)
        temp_portfolio_returns = torch.matmul(torch.transpose(returns_tensor,0,1), temp_asset_ratios)
        port_return_means[portfolio] = temp_portfolio_returns.mean()
        port_return_stds[portfolio]  = temp_portfolio_returns.std()
        
    return port_return_means, port_return_stds
    

    # fig = plt.figure(2)
    # ax = fig.add_subplot()
    # ax.set_xlim(0, 1.1*(port_return_stds.max()))
    # ax.set_ylim(0, 1.1* max((port_return_means.max()-1),portfolio_returns.mean()-1))
    # x = np.linspace(0,port_return_stds.max(),10)
    # y = x * sharpe
    # ax.scatter(portfolio_returns.std().item(), portfolio_returns.mean().item()-1, color='red', label='Optimized Portfolio', zorder=3)
    # ax.scatter(port_return_stds, port_return_means-1, s=1, label=str(num_portfolios) +' Random Portfolios', zorder=2)
    # ax.plot(x,y,color='black', zorder=1)
    # ax.text(0.74,0.05, 'Asset\nTickers:\n\n'+'\n'.join(tickers), transform=ax.transAxes, ha='left',  va='bottom').set_bbox(dict(fc='whitesmoke',ec='black'))
    # ax.text(0.87,0.05, 'Ideal \nRatios:\n\n'+'\n'.join([f'{i:.3f}' for i in asset_ratios.tolist()[0]]), transform=ax.transAxes, ha='left',  va='bottom').set_bbox(dict(fc='whitesmoke',ec='black'))
    # ax.set_ylabel('Mean of Daily Returns', fontsize=16)
    # ax.set_xlabel('Standard Deviation of Daily Returns', fontsize=16)
    # ax.legend(loc='lower left', facecolor='whitesmoke',edgecolor='black') 
    # #plt.savefig(name+'_'+str(num_days)+'_MC'+'.svg')
    # fig.tight_layout()
    # plt.savefig(name+'_'+str(num_days)+'_MC'+'.jpg', dpi=300)
    # # plt.show()


def PricePlots(fetches, num_days, tickers, portfolio_returns, returns_tensor, name):
    num_assets = len(tickers)
    # time = np.linspace(1, num_days, num_days)
    time = fetches[0].index
    port_value = torch.empty(num_days)
    asset_value = torch.empty(num_assets, num_days)
    for day in range(num_days):
        port_value[day] = (100 * portfolio_returns.detach()[0:day].prod())
        for asset in range(num_assets):
            asset_value[asset, day] = (100 * returns_tensor.detach()[asset, 0:day].prod())

    fig = plt.figure(3)
    ax = fig.add_subplot()
    ax.plot(time, port_value, label='Portfolio', color='black', linewidth=5)
    for asset in range(num_assets):
        ax.plot(time, asset_value[asset, :], label=tickers[asset])
    ax.set(title='Portfolio and Individual Asset Performance',
           ylabel='Percentage of Initial Value',
           xlabel='Trading Days')
    ax.legend(loc='best', facecolor='whitesmoke', edgecolor='black') 
    #plt.savefig(name+'_'+str(num_days)+'_Performance'+'.svg')
    plt.savefig(name+'_'+str(num_days)+'_Performance'+'.jpg')
    # plt.show()