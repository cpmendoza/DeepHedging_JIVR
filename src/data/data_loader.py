"""
Usage:
    1. cd src
    2. python data/data_loader.py
"""

import os, sys
from pathlib import Path

main_folder = str(Path.cwd().parent)
sys.path.append(main_folder)

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import os, sys
from pathlib import Path
from scipy.stats import norm
import numpy as np

class portfolio_value():
    
    def Black_Scholes_price(self,S, K, TT, sigma, isput,r,q):
        
        """Function that provides the vanilla option price based on Black-Scholes formula

            Parameters
            ----------
            S     : sigle value - Moneyness
            K     : single value - Time to maturity
            TT    : numpy.ndarray - parameters
            sigma : location of a fast convexity change in the IV term structure with respect to time to maturity
            isput : single value - Maximal maturity represented by the model
            r     : annualized risk-free rate
            q     : annualized dividend yield

            Returns
            -------
            price : Option price

        """

        d1 = (np.log(S/K)+(r-q+(sigma**2)/2)*TT)/(sigma*np.sqrt(TT))
        d2 = d1-sigma*np.sqrt(TT)
        if isput == False:
            price = S*np.exp(-q*TT)*norm.cdf(d1)-K*np.exp(-r*TT)*norm.cdf(d2)
        else:
            price = K*np.exp(-r*TT)*norm.cdf(-d2)-S*np.exp(-q*TT)*norm.cdf(-d1)
        return price
    
    def volatility(self,M,tau,beta,T_conv = 0.25,T_max = 5):

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

    def portfolio(self, Price_mat, strike, T, isput, r, q, betas, sigma = None):
        
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
        forward_price_1 = Price_mat[0,:]*np.exp(T*(r-q))
        M               = np.log(forward_price_1/strike)*(1/np.sqrt(T))
        if sigma is None:
          sigma         = self.volatility(M,T,betas[:,0,:])
        
        price           = self.Black_Scholes_price(Price_mat[0,:], strike, T, sigma, isput,r,q)
        return price
    

def load_standard_datasets(maturity, sigma = None):
    
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
        Price_mat    = np.transpose(np.load(os.path.join(f"Stock_paths__random_f_{maturity}.npy")))
        S            = Price_mat
        # 1.2) IV coefficients
        betas        = np.load(os.path.join(f"Betas_simulation__random_f_{maturity}.npy"))
        # 1.3) Volatility
        h_simulation = np.load(os.path.join(f"H_simulation__random_f_{maturity}.npy"))
    else:
        # 1) Load datasets to train and test deep hedging algorithm (Simulated paths, Betas IV, volatilities)
        # 1.1) matrix of simulated stock prices
        Price_mat    = np.transpose(np.load(os.path.join(f"Stock_paths__random_b_{maturity}.npy")))
        S            = Price_mat
        # 1.2) IV coefficients
        betas        = None
        # 1.3) Volatility
        h_simulation = None
    
    return Price_mat, S, betas, h_simulation


def data_sets_preparation(moneyness,isput,prepro_stock, Price_mat, S, betas, h_simulation, trainphase = True, sigma = None, S_0 = 100, r = 0.009713920000000001, q = 0.01543706):
    
    """Function that loads the sets to create the training set

        Parameters
        ----------
        moneyness       : Moneyness option {"ATM","ITM","OTM"}
        isput           : Put or Call {True,False}
        prepro_stock    : Price preprocessing {"Log", "Log-moneyness", "Nothing"}
        Price_mat       : Matrix of underlying asset prices
        S               : Transposed matrix of the underlying asset price
        betas           : Coefficients of the IV surface 
        h_simulation    : Volatility of the underlying asset
        trainphase      : Parameter for Backtest procedure {True,False}

        Returns
        -------
        option          : Option name {"Call","Put"}
        n_timesteps     : Time steps
        train_input     : Training set (normalized stock price and features)
        test_input      : Validation set (normalized stock price and features)
        disc_batch      : Risk-free rate update factor exp(h*r)
        dividend_batch  : Dividend yield update factor exp(h*d)
        V_0_train       : Initial portfolio value for training set
        V_0_test        : Initial portfolio value for validation set
        strike          : Strike value of the option hedged 

      """

    # 1) Define environment in terms of the option
    # 1.1) Option type
    moneyness_list = ["ATM","ITM","OTM"]
    K_firsts    = [100,90,110]
    idx         = moneyness_list.index(moneyness)
    if isput == False:
      option = 'Call'
    else:
      option = 'Put'
    # 1.2) Moneyness and strike
    if isput == True:
      if moneyness_list[idx]=="ITM":
        strike   = K_firsts[idx+1]
      elif moneyness_list[idx]=="OTM":
        strike   = K_firsts[idx-1]
      else:
        strike   = K_firsts[idx]
    else:
      strike     = K_firsts[idx]

    n_timesteps  = Price_mat.shape[0]-1     # Daily hedging
    #T            = n_timesteps/252          # Time-to-maturity of the vanilla put option
    T = n_timesteps/252 if sigma is None else 252/252 # Time-to-maturity of the vanilla put option (63/252, 252/252)
    h            = T / n_timesteps # Daily size step (T / n_timesteps) #Modify size of the step

    # Apply a transformation to stock prices
    if(prepro_stock == "Log"):
        Price_mat = np.log(Price_mat)
    elif(prepro_stock == "Log-moneyness"):
        Price_mat = np.log(Price_mat/strike)

    if trainphase==True:
      # Construct the train and test sets
      # - The feature vector for now is [S_n, T-t_n]; the portfolio value V_{n} will be added further into the code at each time-step
      if sigma is None:
        train_input     = np.zeros((n_timesteps+1, 400000,13)) #8, 13
        test_input      = np.zeros((n_timesteps+1, 99000,13))
        time_to_mat     = np.zeros(n_timesteps+1)
        time_to_mat[1:] = T / (n_timesteps)      # [0,h,h,h,..,h]
        time_to_mat     = np.cumsum(time_to_mat) # [0,h,2h,...,Nh]
        time_to_mat     = time_to_mat[::-1]      # [Nh, (N-1)h,...,h,0]

        train_input[:,:,0] = Price_mat[:,0:400000]
        train_input[:,:,1] = np.reshape(np.repeat(time_to_mat, train_input.shape[1], axis=0), (n_timesteps+1, train_input.shape[1]))
        for i in range(5):
          train_input[:,:,2+i] = np.transpose(betas[:,:,i])[:,0:400000]
        #train_input[:,:,7] = np.transpose(h_simulation[:,:,5])[:,0:400000] if n_timesteps==63 else np.transpose(h_simulation[:,:])[:,0:400000] #np.transpose(h_simulation[:,:,5])[:,0:400000] 
        for i in range(6):
          train_input[:,:,7+i] = np.transpose(h_simulation[:,:,i])[:,0:400000]

        test_input[:,:,0]  = Price_mat[:,400000:499000]
        test_input[:,:,1]  = np.reshape(np.repeat(time_to_mat, test_input.shape[1], axis=0), (n_timesteps+1, test_input.shape[1]))
        for i in range(5):
          test_input[:,:,2+i] = np.transpose(betas[:,:,i])[:,400000:499000]
        #test_input[:,:,7] = np.transpose(h_simulation[:,:,5])[:,400000:499000] if n_timesteps==63 else np.transpose(h_simulation[:,:])[:,400000:499000] #np.transpose(h_simulation[:,:,5])[:,400000:499000]
        for i in range(6):
          test_input[:,:,7+i] = np.transpose(h_simulation[:,:,i])[:,400000:499000]
      else:
        train_input     = np.zeros((n_timesteps+1, 400000,2)) #8, 13
        test_input      = np.zeros((n_timesteps+1, 99000,2))
        time_to_mat     = np.zeros(n_timesteps+1)
        time_to_mat[1:] = T / (n_timesteps)      # [0,h,h,h,..,h]
        time_to_mat     = np.cumsum(time_to_mat) # [0,h,2h,...,Nh]
        time_to_mat     = time_to_mat[::-1]      # [Nh, (N-1)h,...,h,0]

        train_input[:,:,0] = Price_mat[:,0:400000]
        train_input[:,:,1] = np.reshape(np.repeat(time_to_mat, train_input.shape[1], axis=0), (n_timesteps+1, train_input.shape[1]))
        
        test_input[:,:,0]  = Price_mat[:,400000:499000]
        test_input[:,:,1]  = np.reshape(np.repeat(time_to_mat, test_input.shape[1], axis=0), (n_timesteps+1, test_input.shape[1]))
        #for i in range(6):
        #  test_input[:,:,7+i] = np.transpose(h_simulation[:,:,i])[:,400000:499000]       

      disc_batch            = np.exp(r*h)   # exp(rh)
      dividend_batch        = np.exp(q*h)   # exp(qh)

      #Initial portfolio values 
      price_function = portfolio_value()
      price          = price_function.portfolio(S, strike, T, isput, r, q, betas, sigma)
      V_0_train      = np.ones(train_input.shape[1])*price[0:400000]  
      V_0_test       = np.ones(test_input.shape[1])*price[400000:499000]

    else:
        # Construct the train and test sets
      # - The feature vector for now is [S_n, T-t_n]; the portfolio value V_{n} will be added further into the code at each time-step
      train_input     = None
      test_input      = np.zeros((n_timesteps+1, Price_mat.shape[1],8))
      time_to_mat     = np.zeros(n_timesteps+1)
      time_to_mat[1:] = T / (n_timesteps)      # [0,h,h,h,..,h]
      time_to_mat     = np.cumsum(time_to_mat) # [0,h,2h,...,Nh]
      time_to_mat     = time_to_mat[::-1]      # [Nh, (N-1)h,...,h,0]

      test_input[:,:,0]  = Price_mat
      test_input[:,:,1]  = np.reshape(np.repeat(time_to_mat, test_input.shape[1], axis=0), (n_timesteps+1, test_input.shape[1]))
      for i in range(5):
        test_input[:,:,2+i] = np.transpose(betas[:,:,i])
      test_input[:,:,7] = np.transpose(h_simulation[:,:,5])
      #for i in range(6):
      #  test_input[:,:,7+i] = np.transpose(h_simulation[:,:,i])[:,400000:499000]

      disc_batch            = np.exp(r*h)   # exp(rh)
      dividend_batch        = np.exp(q*h)   # exp(qh)

      #Initial portfolio values 
      price_function = portfolio_value()
      price          = price_function.portfolio(S, strike, T, isput, r, q, betas, sigma)
      V_0_train      = None
      V_0_test       = price

    return option, n_timesteps, train_input, test_input, disc_batch, dividend_batch, V_0_train, V_0_test, strike


def training_variables(maturity, moneyness, isput, prepro_stock, backtest, sigma = None):

    """Function that loads the sets to create the training set

        Parameters
        ----------
        maturity        : time to maturity of the options
        moneyness       : Moneyness option {"ATM","ITM","OTM"}
        isput           : Put or Call {True,False}
        prepro_stock    : Price preprocessing {"Log", "Log-moneyness", "Nothing"}
        backtest        : Parameter for Backtest procedure {True,False}

        Returns
        -------
        option          : Option name {"Call","Put"}
        n_timesteps     : Time steps
        train_input     : Training set (normalized stock price and features)
        test_input      : Validation set (normalized stock price and features)
        disc_batch      : Risk-free rate update factor exp(h*r)
        dividend_batch  : Dividend yield update factor exp(h*d)
        V_0_train       : Initial portfolio value for training set
        V_0_test        : Initial portfolio value for validation set
        strike          : Strike value of the option hedged 

      """

    owd = os.getcwd()
    try:
      #first change dir to build_dir path
      if backtest==True:
        trainphase = False
        os.chdir(os.path.join(main_folder, f"data/processed/Backtest/"))
      else:
        trainphase = True
        os.chdir(os.path.join(main_folder, f"data/processed/Training/"))

      Price_mat, S, betas, h_simulation = load_standard_datasets(maturity, sigma)
      option, n_timesteps, train_input, test_input, disc_batch, dividend_batch, V_0_train, V_0_test, strike = data_sets_preparation(moneyness,isput,prepro_stock, Price_mat, S, betas, h_simulation, trainphase, sigma)
    finally:
      #change dir back to original working directory (owd)
      os.chdir(owd)
      
    return option, n_timesteps, train_input, test_input, disc_batch, dividend_batch, V_0_train, V_0_test, strike
  
    
if __name__ == "__main__":
    main()

        
        




























