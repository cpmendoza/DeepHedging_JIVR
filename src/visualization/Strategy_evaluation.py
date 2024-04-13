"""
Usage:
    1. cd src
    2. python models/matchmaking/advisor_to_company_match.py
"""


import os, sys
from pathlib import Path

main_folder = str(Path.cwd().parent)
sys.path.append(main_folder)

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy import stats


class strategy_hedging_valuation(object):
    
    """ Class to compute hedging error of self-financing strategies
    Inputs:
    transacion_cost : Proportional transaction cost [0,1]
    r               : Annualized continuous risk-free rate
    q               : Annualized continuous dividend yield
    isput           : Put or Call: {True,False}
    """
    
    def __init__(self, transaction_cost = 0, isput=False, r = 0.009713920000000001, q = 0.01543706, hedging_strategy = "one_instrument",):

        # Parameters
        self.r = r                                    
        self.q = q                                  
        self.transaction_cost = transaction_cost
        self.hedging_strategy = hedging_strategy
        self.isput = isput
        
    #Function to compute option prices 
    def option_price(self, S, K, TT, sigma):

        """Function that provides the initial price of a portfolio of options

            Parameters
            ----------
            Price_mat : Underlying asset price
            K         : Strike of the option
            TT        : Time-to-maturity
            sigma     : volatility
            isput     : Parameter to determine Put or Call
            r         : annualized risk-free rate
            q         : annualized dividend yield

            Returns
            -------
            price     : Option price of all paths at time zero
        """
        d1 = (np.log(S/K)+(self.r-self.q+(sigma**2)/2)*TT)/(sigma*np.sqrt(TT))
        d2 = d1-sigma*np.sqrt(TT)
        if self.isput==False:
            price = S*np.exp(-self.q*TT)*norm.cdf(d1)-K*np.exp(-self.r*TT)*norm.cdf(d2)
        else:
            price = K*np.exp(-self.r*TT)*norm.cdf(-d2)-S*np.exp(-self.q*TT)*norm.cdf(-d1)
        
        return price
    
    def hedging_error_vector(self, K, Stock_paths, implied_volatility_simulation_1, position_underlying_asset, Hedge_option_price_path = None,  position_hedging_option = None, close_limit_days = 5):

        """
        Input:
        K                                   : Strike of the option
        Stock_paths                         : Underlying asset price
        implied_volatility_simulation_1     : Implied volatility of the underlying
        position_underlying_asset           : Vector of the positions in the underlying
        Hedge_option_price_path             : Hedging option price path 
        position_hedging_option             : Vector of the positions in another hedging instruments
        close_limit_days                    : Number of days before maturity to close the position

        Output:
        hedging_portfolio                   : Portfolio value per path
        hedging_error                       : Hedging error per time-step for all paths
        hedging_error_limit                 : Vector with hedging error at time we liquidate the position
        cost_limit                          : Final transaction cost of the strategy
        option_price                        : Option price matrix

        """
        
        #General values
        time_steps = Stock_paths.shape[1]-1
        number_simulations = Stock_paths.shape[0]
        limit = time_steps-close_limit_days

        ############# CHANGE FOR GENERAL CASE RIHT HERE ##############
        ##############################################################

        TT = time_steps/252 #for general daily hedging (63/252 or 252/252)
        h  = TT / time_steps
        time_to_maturity = np.array(sorted((np.arange(time_steps)+1),reverse=True))*h
        option_price = self.option_price(Stock_paths[:,:-1], K, time_to_maturity, implied_volatility_simulation_1)
        
        #Hedging portfolio
        hedging_portfolio = np.zeros([number_simulations,(time_steps+1)])
        hedging_error = np.zeros([number_simulations,(time_steps+1)])
        cost_matrix = np.zeros([number_simulations,(time_steps)])
        hedging_portfolio[:,0] = option_price[:,0]
        
        if self.isput==False:
            payoff = np.maximum(Stock_paths[:,time_steps]-K,0)
        else:
            payoff = np.maximum(K-Stock_paths[:,time_steps],0)
        
        #Compute P&L for the hedging startegies
        if self.hedging_strategy == "one_instrument":
            
            #Computing hedging errors with only one hedging instrument
            for time_step in range(time_steps):
                #Transacition cost computation
                if time_step == 0:
                    cost = np.abs(Stock_paths[:,time_step]*position_underlying_asset[:,time_step]*self.transaction_cost)
                    cost_matrix[:,0] = cost
                else:
                    cost = np.abs(self.transaction_cost*Stock_paths[:,time_step]*(position_underlying_asset[:,time_step]-position_underlying_asset[:,time_step-1]))
                    cost_matrix[:,time_step] = cost_matrix[:,time_step-1] + cost #*np.exp((-1)*self.r*h)

                phi_0 = hedging_portfolio[:,time_step] - Stock_paths[:,time_step]*position_underlying_asset[:,time_step] - cost
                hedging_portfolio[:,(time_step+1)] = phi_0*np.exp(self.r*h) + position_underlying_asset[:,time_step]*Stock_paths[:,(time_step+1)]*np.exp(self.q*h)
            
            #P&L Computation
            hedging_error[:,:-1] = hedging_portfolio[:,:-1]-option_price
            hedging_error[:,time_steps] = hedging_portfolio[:,time_steps] - payoff
            hedging_error_limit = hedging_error[:,limit]
            cost_limit = cost_matrix[:,limit-1]
            
        elif self.hedging_strategy == "two_instruments":
            
            #Computing hedging errors with two hedging instruments
            for time_step in range(time_steps):
            
                 #Transacition cost computation
                if time_step == 0:
                    cost = np.abs(Stock_paths[:,time_step]*position_underlying_asset[:,time_step]*self.transaction_cost) + np.abs(self.transaction_cost*Hedge_option_price_path[:,(time_step)]*position_hedging_option[:,time_step])
                    cost_matrix[:,0] = cost
                else:
                    cost = np.abs(self.transaction_cost*Stock_paths[:,time_step]*(position_underlying_asset[:,time_step] - position_underlying_asset[:,time_step-1])) + np.abs(self.transaction_cost*Hedge_option_price_path[:,(time_step)]*(position_hedging_option[:,time_step]-position_hedging_option[:,time_step-1]))
                    cost_matrix[:,time_step] = cost_matrix[:,time_step-1] + cost*np.exp((-1)*self.q*(time_step/252))

                phi_0 = hedging_portfolio[:,time_step] - Stock_paths[:,time_step]*position_underlying_asset[:,time_step] - Hedge_option_price_path[:,(time_step)]*position_hedging_option[:,time_step] - cost
                hedging_portfolio[:,(time_step+1)] = phi_0*np.exp(self.r*h) + position_underlying_asset[:,time_step]*Stock_paths[:,(time_step+1)]*np.exp(self.q*h) + position_hedging_option[:,time_step]*Hedge_option_price_path[:,(time_step+1)]
            
            #P&L Computation
            hedging_error[:,:-1] = hedging_portfolio[:,:-1]-option_price
            hedging_error[:,time_steps] = hedging_portfolio[:,time_steps] - payoff
            hedging_error_limit = hedging_error[:,limit]
            cost_limit = cost_matrix[:,limit-1]
 
        return hedging_portfolio, hedging_error, hedging_error_limit, cost_limit, option_price
    

def loss_functions(hedging_err):
    """
    Input:
    hedging_err   : Vector of hedging errors

    Output:
    loss          : Loss function values

    """
    "Mean"
    loss = np.mean(hedging_err)
    "CVaR - 95"
    loss = np.append(loss,np.mean(np.sort(hedging_err)[int(0.95*hedging_err.shape[0]):]))
    "CVaR - 99"
    loss = np.append(loss,np.mean(np.sort(hedging_err)[int(0.99*hedging_err.shape[0]):]))
    "MSE"
    loss = np.append(loss,np.mean(np.square(hedging_err)))
    "SMSE"
    loss = np.append(loss,np.mean(np.square(np.where(hedging_err>0,hedging_err,0))))
        
    return(loss)
    
def statistics(hedging_error):
    """
    Input:
    hedging_err   : Vector of hedging errors

    Output:
    statistics    : Distributional statistics

    """
    percentiles = np.percentile(hedging_error, [10,20,30,40,50,60,70,80,90])
    x = stats.describe(hedging_error)
    statistics = np.insert(percentiles, 0, [x[2],np.sqrt(x[3])], axis=None)
    return(statistics)

def load_standard_datasets(maturity, sigma):
    
    """Function that loads the sets to create the training set

        Parameters
        ----------
        maturity     : time to maturity of the options

        Returns
        -------
        Price_mat    : Matrix of underlying asset prices
        S            : Transposed matrix of the underlying asset price
        betas        : Coefficients of the IV surface 
        h_simulation : Volatility of the underlying asset

      """
    if sigma is None:
        # 1) Load datasets to train and test deep hedging algorithm (Simulated paths, Betas IV, volatilities)
        # 1.1) matrix of simulated stock prices
        Price_mat    = np.load(os.path.join(f"Stock_paths__random_f_{maturity}.npy"))
        S            = Price_mat
        # 1.2) IV coefficients
        betas        = np.load(os.path.join(f"Betas_simulation__random_f_{maturity}.npy"))
        # 1.3) Volatility
        h_simulation = np.load(os.path.join(f"H_simulation__random_f_{maturity}.npy"))
        returns      = np.load(os.path.join(f"Returns_random__random_f_{maturity}.npy"))
    else:
        # 1) Load datasets to train and test deep hedging algorithm (Simulated paths, Betas IV, volatilities)
        # 1.1) matrix of simulated stock prices
        Price_mat    = np.load(os.path.join(f"Stock_paths__random_b_{maturity}.npy"))
        S            = Price_mat
        # 1.2) IV coefficients
        betas        = None
        # 1.3) Volatility
        h_simulation = None
        returns = np.log(Price_mat[:,1:]/Price_mat[:,:-1])
        
    return Price_mat, S, betas, h_simulation, returns

def volatility(M,tau,beta,T_conv = 0.25,T_max = 5):

    """Function that provides the volatility surface model

        Parameters
        ----------
        M          : sigle value - Moneyness
        tau        : single value - Time to maturity
        beta       : numpy.ndarray - parameters
        T_conv     : location of a fast convexity change in the IV term structure with respect to time to maturity
        T_max      : single value - Maximal maturity represented by the model

        Returns
        -------
        volatility : Implied volatility

    """

    Long_term_ATM_Level = beta[:,0]
    Time_to_maturity_slope = beta[:,1]*np.exp(-1*np.sqrt(tau/T_conv))
    M_plus = (M<0)*M
    Moneyness_slope = (beta[:,2]*M)*(M>=0)+ (beta[:,2]*((np.exp(2*M_plus)-1)/(np.exp(2*M_plus)+1)))*(M<0)
    Smile_attenuation = beta[:,3]*(1-np.exp(-1*(M**2)))*np.log(tau/T_max)
    Smirk = (beta[:,4]*(1-np.exp((3*M_plus)**3))*np.log(tau/T_max))*(M<0)

    volatility = np.maximum(Long_term_ATM_Level + Time_to_maturity_slope + Moneyness_slope + Smile_attenuation + Smirk,0.01)

    return volatility


def hedging_valuation_dataset(backtest, K, deltas, sigma, r = 0.009713920000000001, q = 0.01543706):

    """
    Input:
    transacion_cost                   : Proportional transaction cost [0,1]
    isput                             : Put or Call: {True,False}
    K                                 : Strike of the option
    Stock_paths                       : Underlying asset price
    implied_volatility_simulation_1   : Implied volatility
    deltas                            : Position in the underlying asset
    close_limit_days                  : Number of days before maturity to close the position

    Output:
    hedging_portfolio                 : Portfolio value per path
    hedging_error                     : Hedging error per time-step for all paths
    hedging_error_limit               : Vector with hedging error at time we liquidate the position
    cost_limit                        : Final transaction cost of the strategy
    option_price                      : Option price matrix
    df_statistic_TC                   : Transaction cost distributional statistics 
    df_cost_functions                 : Loss function values

    """

    
    if sigma is None:
        time_steps = deltas.shape[1]
    else: 
        time_steps = 252 # 63 or 252

    owd = os.getcwd()

    #first change dir to build_dir path
    if backtest==True:
        number_simulations = 296#10000 #296
        os.chdir(os.path.join(main_folder, f"data/processed/Backtest/"))
        _, S, betas, _, _ = load_standard_datasets(time_steps,sigma)
        Stock_paths = S
        betas_simulation = betas
    else:
        number_simulations = 100000
        os.chdir(os.path.join(main_folder, f"data/processed/Training/"))
        _, S, betas, _, _ = load_standard_datasets(time_steps,sigma)
        Stock_paths = S[400000:,:]
        betas_simulation = betas[400000:,:,:] if sigma is None else None

    os.chdir(owd)
    if sigma is None:
        implied_volatility_simulation = np.zeros([number_simulations,time_steps])

        #Compute forward prices for all simulations
        time_to_maturity_1 = np.array(sorted((np.arange(time_steps)+1),reverse=True))/252
        interest_rates_difference = r - q
        forward_price_1 = Stock_paths[:,:-1]*np.exp(time_to_maturity_1*interest_rates_difference)

        #Compute Moneyness for all simulations
        moneyness_1 = np.log(forward_price_1/K)*(1/np.sqrt(time_to_maturity_1))

            
        for time_step in range(time_steps+1):
            if time_step < time_steps:
                implied_volatility_simulation[:,time_step] = volatility(moneyness_1[:,time_step],time_to_maturity_1[time_step],betas_simulation[:,time_step,:])
    else: 
        implied_volatility_simulation = sigma

    return Stock_paths, implied_volatility_simulation

def hedging_valuation(backtest, strikes, deltas, transaction_cost, isput, close_limit_days, r, q, sigma = None):

    Stock_paths, implied_volatility_simulation = hedging_valuation_dataset(backtest, strikes, deltas, sigma, r, q)
    new_evaluation       = strategy_hedging_valuation(transaction_cost,isput,r,q)
    hedging_portfolio, hedging_error, hedging_error_limit, cost_limit, option_price = new_evaluation.hedging_error_vector(strikes, Stock_paths, implied_volatility_simulation, deltas, None, None, close_limit_days)
    df_statistic_TC   = statistics(cost_limit)
    df_cost_functions = loss_functions(-1*hedging_error_limit)

    df_cost_functions = pd.DataFrame(df_cost_functions)
    df_cost_functions.index = ["Mean-HE","CVaR_95%","CVaR_99%","MSE","SMSE"]
    df_cost_functions = df_cost_functions.T

    return hedging_portfolio, hedging_error, hedging_error_limit, cost_limit, option_price, df_statistic_TC, df_cost_functions


if __name__ == "__main__":
    main()