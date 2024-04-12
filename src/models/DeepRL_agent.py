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
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from tensorflow.python.framework import ops

class DeepAgent(object):
    """
    Inputs:
    network         : neural network architechture {LSTM,RNN-FNN,FFNN}
    nbs_point_traj  : if [S_0,...,S_N] ---> nbs_point_traj = N+1
    batch_size      : size of mini-batch
    nbs_input       : number of features (without considerint V_t)
    nbs_units       : number of neurons per layer
    nbs_assets      : dimension of the output layer (number of hedging instruments)
    cash_constraint : Lower bound of the output layer activation function
    constraint_max  : Lower bound of the output layer activation function
    loss_type       : loss function for the optimization procedure {CVaR,SMSE,MSE}
    lr              : learning rate hyperparameter of the Adam optimizer
    dropout_par:    : dropout regularization parameter [0,1]
    isput           : condition to determine the option type for the hedging error {True,False}
    prepro_stock    : {Log, Log-moneyness, Nothing} - what transformation was used for stock prices
    name            : name to store the trained model

    # Disclore     : Class adapted from https://github.com/alexandrecarbonneau/Deep-Equal-Risk-Pricing-of-Financial-Derivatives-with-Multiple-Hedging-Instruments/blob/main/Example%20of%20ERP%20with%20deep%20hedging%20multi%20hedge%20-%20Final.ipynb
    """
    def __init__(self, network, state_space, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, cash_constraint, constraint_max, loss_type, lr, dropout_par, isput, prepro_stock, name):
        tf.compat.v1.disable_eager_execution()
        ops.reset_default_graph()
        # 0) Network parameters
        self.network        = network
        self.state_space    = state_space
        self.nbs_point_traj = nbs_point_traj
        self.batch_size     = batch_size
        self.nbs_input      = nbs_input
        self.nbs_units      = nbs_units
        self.nbs_assets     = nbs_assets
        self.cash_constraint = cash_constraint
        self.constraint_max = -1*constraint_max
        self.loss_type      = loss_type
        self.lr             = lr
        self.dropout_par    = dropout_par
        self.isput          = isput
        self.prepro_stock   = prepro_stock

        # 1) Placeholder
        self.input            = tf.placeholder(tf.float32, [nbs_point_traj, batch_size, nbs_input])             # normalized prices and features
        self.strike           = tf.placeholder(tf.float32)                                                      # strike of the option (K)
        self.alpha            = tf.placeholder(tf.float32)                                                      # CVaR confidence level (alpha in (0,1))
        self.transacion_cost  = tf.placeholder(tf.float32)                                                      # transaction cost rate (tc in (0,1))
        self.disc_tensor      = tf.placeholder(tf.float32)                                                      # risk-free rate factor exp(delta*r)
        self.dividend_tensor  = tf.placeholder(tf.float32)                                                      # dividend rate factor exp(delta*d)
        self.deltas           = tf.zeros(shape = [nbs_point_traj-1, batch_size, nbs_assets], dtype=tf.float32)  # array to store position in the hedging instruments
        self.portfolio        = tf.zeros(shape = [batch_size], dtype=tf.float32)                                # initial portfolio value

        # 2) Discount prices computation:
        self.unorm_price = self.inverse_processing(self.input[:,:,0:self.nbs_assets], prepro_stock)             # Inverse normalization to compute payoff

        # 3) Payoff of each path depending on the option type (vanilla options)
        if (self.isput == False):
            self.payoff = tf.maximum(self.unorm_price[-1,:,0]-self.strike,0)
        else:
            self.payoff = tf.maximum(self.strike-self.unorm_price[-1,:,0],0)

        # 4) Network architechture for the deep hedging algorithm
        if (self.network == "LSTM"):
          # 4.1) Four LSTM cells (the dimension of the hidden state and the output is the same)
          layer_1 = tf.compat.v1.keras.layers.LSTM(units  = self.nbs_units) 
          layer_2 = tf.compat.v1.keras.layers.LSTM(units  = self.nbs_units)
          layer_3 = tf.compat.v1.keras.layers.LSTM(units  = self.nbs_units)
          layer_4 = tf.compat.v1.keras.layers.LSTM(units  = self.nbs_units)
          # 4.2) Output layer of dimension nbs_assets (outputs the position in the hedging instruments)
          layer_out = tf.layers.Dense(self.nbs_assets, None)
        elif (self.network == "RNNFNN"):
          # 4.1) Two LSTM cells (the dimension of the hidden state and the output is the same)
          #      Two regular layers with RELU activation function and dropout regularization per layer
          layer_1 = tf.compat.v1.keras.layers.LSTM(units  = self.nbs_units) 
          layer_2 = tf.compat.v1.keras.layers.LSTM(units  = self.nbs_units)
          layer_3 = tf.layers.Dense(self.nbs_units, tf.nn.relu)
          layer_drop_3 = tf.keras.layers.Dropout(self.dropout_par)
          layer_4 = tf.layers.Dense(self.nbs_units, tf.nn.relu)
          layer_drop_4 = tf.keras.layers.Dropout(self.dropout_par)
          # 4.2) Output layer of dimension nbs_assets (outputs the position in the hedging instruments)
          layer_out = tf.layers.Dense(self.nbs_assets, None)
        elif (self.network == "FFNN"):
          # 4.1) Four regular layers with RELU activation function and dropout regularization per layer
          layer_1 = tf.layers.Dense(self.nbs_units, tf.nn.relu)
          layer_drop_1 = tf.keras.layers.Dropout(self.dropout_par)
          layer_2 = tf.layers.Dense(self.nbs_units, tf.nn.relu)
          layer_drop_2 = tf.keras.layers.Dropout(self.dropout_par)
          layer_3 = tf.layers.Dense(self.nbs_units, tf.nn.relu)
          layer_drop_3 = tf.keras.layers.Dropout(self.dropout_par)
          layer_4 = tf.layers.Dense(self.nbs_units, tf.nn.relu)
          layer_drop_4 = tf.keras.layers.Dropout(self.dropout_par)
          # 4.2) Output layer of dimension one (outputs the position in the underlying)
          layer_out = tf.layers.Dense(self.nbs_assets, None)

        # 4.3) Compute hedging strategies for all time-steps
        V_t = self.portfolio
        self.layer_prev   = tf.zeros(shape = [batch_size,1], dtype=tf.float32)
        self.constraint_1 = tf.zeros(shape = [batch_size], dtype=tf.float32)
        self.constraint_2 = tf.zeros(shape = [batch_size], dtype=tf.float32)
        self.constraint_1 += tf.math.subtract(V_t,self.constraint_max)
        self.constraint_1 = tf.divide(self.constraint_1,self.unorm_price[0,:,0]*(1+self.transacion_cost))
        self.constraint_2 += self.constraint_1
        for t in range(self.nbs_point_traj-1):

            if self.state_space == "Full":
                input_t = tf.concat([self.input[t,:,:], tf.expand_dims(V_t, axis = 1)], axis=1)
                input_t = tf.concat([input_t, self.layer_prev], axis=1)
            elif self.state_space == "Reduced_2":
                input_t = tf.concat([self.input[t,:,:], self.layer_prev], axis=1)
            else:
                input_t = self.input[t,:,:]

            if (self.network == "LSTM"):
                # input of the LSTM cells at time 't': [S_t, T-t, V_t] with dimension [number of samples, time series dimension = 1 ,number of features]
                input_t = tf.expand_dims(input_t, axis = 1)
                # forward prop at time 't'
                layer = layer_1(input_t)
                layer = layer_2(tf.expand_dims(layer, axis = 1))
                layer = layer_3(tf.expand_dims(layer, axis = 1))
                layer = layer_4(tf.expand_dims(layer, axis = 1))
                layer = layer_out(layer)

            elif (self.network == "RNNFNN"):
                 # input of the LSTM cells at time 't': [S_t, T-t, V_t] with dimension [number of samples, time series dimension = 1 ,number of features]
                input_t = tf.expand_dims(input_t, axis = 1)
                # forward prop at time 't'
                layer = layer_1(input_t)
                layer = layer_2(tf.expand_dims(layer, axis = 1))
                layer = layer_3(layer)
                layer = layer_drop_3(layer)
                layer = layer_4(layer)
                layer = layer_drop_4(layer)
                layer = layer_out(layer)

            else:
                # forward prop at time 't'
                layer = layer_1(input_t)
                layer = layer_drop_1(layer)
                layer = layer_2(layer)
                layer = layer_drop_2(layer)
                layer = layer_3(layer)
                layer = layer_drop_3(layer)
                layer = layer_4(layer)
                layer = layer_drop_4(layer)
                layer = layer_out(layer)
            
            #Output layer with constraints
            if (t==0):
                self.upper_bound = tf.expand_dims(self.constraint_1,axis=1)
            else:
                self.upper_bound = tf.where(tf.math.greater_equal(layer,self.layer_prev),tf.expand_dims(self.constraint_1,axis=1),tf.expand_dims(self.constraint_2,axis=1))
            #Layer value
            layer = tf.math.minimum(layer,self.upper_bound) if self.cash_constraint == True else layer      

            # Compile trading strategies
            if (t==0):
                # At t = 0, need to expand the dimension to have [nbs_point_traj, batch_size, nbs_assets]
                self.deltas = tf.expand_dims(layer,axis=0)                      # [1, batch_size, nbs_assets]
                self.cost   = tf.zeros(shape = [batch_size],dtype=tf.float32)   # Vector to store the hedging strategy transaction cost
                #Compute transaction cost of all hedging instruments
                for a in range(self.nbs_assets):
                    self.cost += tf.math.abs(self.deltas[t,:,a]*self.unorm_price[t,:,a])*self.transacion_cost
            else:
                #Store the rest of the hedging positions
                self.deltas = tf.concat([self.deltas, tf.expand_dims(layer, axis = 0)], axis = 0)
                self.cost   = tf.zeros(shape = [batch_size], dtype=tf.float32)
                #Compute transaction cost of all hedging instruments
                for a in range(self.nbs_assets):
                    self.cost +=  tf.math.abs(self.unorm_price[t,:,a]*(self.deltas[t,:,a]-self.deltas[t-1,:,a]))*self.transacion_cost

            # Compute the portoflio value for the next period
            V_t_pre = V_t

            #Previous position
            self.aux_0 = tf.zeros(shape = [batch_size], dtype=tf.float32)
            for a in range(self.nbs_assets):
                 self.aux_0 += self.deltas[t,:,a]*self.unorm_price[t,:,a]
            phi_0   = V_t_pre - self.aux_0 - self.cost

            #New portfolio value
            self.aux = tf.zeros(shape = [batch_size], dtype=tf.float32)
            for a in range(self.nbs_assets):
                if a==0:
                  self.aux += self.deltas[t,:,a]*self.unorm_price[t+1,:,a]*self.dividend_tensor
                else:
                  self.aux += self.deltas[t,:,a]*self.unorm_price[t+1,:,a]

            V_t = phi_0*self.disc_tensor + self.aux

            self.layer_prev   = tf.zeros(shape = [batch_size,1], dtype=tf.float32)
            self.layer_prev  += layer

            #Constraints for next action 
            self.constraint_1 = tf.zeros(shape = [batch_size], dtype=tf.float32)
            self.constraint_2 = tf.zeros(shape = [batch_size], dtype=tf.float32)
            self.constraint_1 += tf.math.subtract(V_t,self.constraint_max)+tf.math.abs(self.deltas[t,:,0]*self.unorm_price[t+1,:,0])*self.transacion_cost
            self.constraint_1 = tf.divide(self.constraint_1,self.unorm_price[t+1,:,0]*(1+self.transacion_cost))
            self.constraint_2 += tf.math.subtract(V_t,self.constraint_max)-tf.math.abs(self.deltas[t,:,0]*self.unorm_price[t+1,:,0])*self.transacion_cost
            self.constraint_2 = tf.divide(self.constraint_2,self.unorm_price[t+1,:,0]*(1-self.transacion_cost))

        # 6) Compute hedging errors for each path      
        self.hedging_err = self.payoff - V_t

                                           
        # 7) Compute the loss function on the batch of hedging error
        # - This is the empirical cost functions estimated with a mini-batch
        if (self.loss_type == "CVaR"):
            self.loss = tf.reduce_mean(tf.sort(self.hedging_err)[tf.cast(self.alpha*self.batch_size,tf.int32):]) 
        elif (self.loss_type == "MSE"):
            self.loss = tf.reduce_mean(tf.square(self.hedging_err)) 
        elif (self.loss_type == "SMSE"):
            self.loss = tf.reduce_mean(tf.square(tf.nn.relu(self.hedging_err))) 
       
        # 8) SGD step with the adam optimizer
        optimizer  = tf.train.AdamOptimizer(learning_rate = self.lr)
        self.train = optimizer.minimize(self.loss)

        # 9) Save the model
        self.saver      = tf.train.Saver()
        self.model_name = name   # name of the neural network to save

    # Function to compute the CVaR_{alpha} outside the optimization, i.e. at the end of each epoch in this case
    def loss_out_optim(self, hedging_err, alpha, loss_type):
        if (loss_type == "CVaR"):
            loss = np.mean(np.sort(hedging_err)[int(alpha*hedging_err.shape[0]):])
        elif (loss_type == "MSE"):
            loss = np.mean(np.square(hedging_err))
        elif (loss_type == "SMSE"):
            loss = np.mean(np.square(np.where(hedging_err>0,hedging_err,0)))
        return loss

    # Given a type of preprocessing, reverse the processing of the stock price
    def inverse_processing(self, paths, prepro_stock):
        if (prepro_stock =="Log-moneyness"):
            paths = tf.multiply(self.strike, tf.exp(paths))
        elif (prepro_stock == "Log"):
            paths = tf.exp(paths)
        return paths


    # ---------------------------------------------------------------------------------------#
    # Function to call the deep hedging algorithm batch-wise
    """
    Input:
    paths           : Training set (normalized stock price and features)
    V_0             : Initial portfolio value for training set
    V_test          : Initial portfolio value for validation set
    strikes         : Strike value of the option hedged 
    disc_batch      : Risk-free rate update factor exp(h*r)
    dividend_batch  : Dividend yield update factor exp(h*d)
    transacion_cost : Proportional transaction cost [0,1]
    riskaversion    : CVaR confidence level (0,1)
    paths_valid     : Validation set (normalized stock price and features)
    epochs          : Number of epochs, training iterations 
    """
    def train_deephedging(self, paths, V_0, V_test, strikes, disc_batch, dividend_batch, transacion_cost, alpha, paths_valid, sess, epochs):
        sample_size       = paths.shape[1]               # total number of paths in the train set
        sample_size_valid = paths_valid.shape[1]
        batch_size        = self.batch_size
        idx               = np.arange(sample_size)       # [0,1,...,sample_size-1]
        idx_valid         = np.arange(sample_size_valid)
        start             = dt.datetime.now()            # Time-to-train
        self.loss_epochs  = 9999999*np.ones((epochs,2))      # Store the loss at the end of each epoch for the train
        valid_loss_best   = 999999999
        epoch             = 0

        # Loop for each epoch until the maximum number of epochs
        while (epoch < epochs):
            hedging_err_train = []  # Store hedging errors obtained for one complete epoch
            hedging_err_valid = []
            np.random.shuffle(idx)  # Randomize the dataset (not useful in this case since dataset is simulated iid)

            # loop over each batch size
            for i in range(int(sample_size/batch_size)):

                # Indexes of paths used for the mini-batch
                indices = idx[i*batch_size : (i+1)*batch_size]

                # SGD step
                _, hedging_err = sess.run([self.train, self.hedging_err],
                                               {self.input           : paths[:,indices,:],
                                                self.strike          : strikes,
                                                self.alpha           : alpha,
                                                self.disc_tensor     : disc_batch,
                                                self.dividend_tensor : dividend_batch,
                                                self.transacion_cost : transacion_cost,
                                                self.portfolio       : V_0[indices]})

                hedging_err_train.append(hedging_err)

            # 2) Evaluate performance on the valid set - we don't train
            for i in range(int(sample_size_valid/batch_size)):
                indices_valid = idx_valid[i*batch_size : (i+1)*batch_size]
                hedging_err_v = sess.run([self.hedging_err],
                                               {self.input           : paths_valid[:,indices_valid,:],
                                                self.strike          : strikes,
                                                self.alpha           : alpha,
                                                self.disc_tensor     : disc_batch,
                                                self.dividend_tensor : dividend_batch,
                                                self.transacion_cost : transacion_cost,
                                                self.portfolio       : V_test[indices_valid]})

                hedging_err_valid.append(hedging_err_v)

            # 3) Store the loss on the train and valid sets after each epoch
            self.loss_epochs[epoch,0] = self.loss_out_optim(np.concatenate(hedging_err_train), alpha, self.loss_type)
            self.loss_epochs[epoch,1] = self.loss_out_optim(np.reshape(np.concatenate(hedging_err_valid, axis=1), sample_size_valid), alpha, self.loss_type)

            # 4) Test if best epoch so far on valid set; if so, save model parameters.
            if((self.loss_epochs[epoch,1] < valid_loss_best)): #& (self.loss_epochs[epoch,1]>0)
                valid_loss_best = self.loss_epochs[epoch,1]
                self.saver.save(sess, self.model_name + '.ckpt')

                print("Saved")

            # Print the CVaR value at the end of each epoch
            if (epoch+1) % 1 == 0:
                print('Time elapsed:', dt.datetime.now()-start)
                print('Epoch %d, %s, Train: %.3f Valid: %.3f' % (epoch+1, self.loss_type,
                                                            self.loss_epochs[epoch,0], self.loss_epochs[epoch,1]))
            epoch+=1  # increment the epoch

        # End of training
        print("---Finished training results---")
        print('Time elapsed:', dt.datetime.now()-start)

        # Return the learning curve
        return self.loss_epochs


    # Function which will call the deep hedging optimization batchwise
    def training(self, paths, V_0, V_test, strikes, disc_batch, dividend_batch, transacion_cost, riskaversion, paths_valid, sess, epochs):
        sess.run(tf.global_variables_initializer())
        loss_train_epoch = self.train_deephedging(paths, V_0, V_test, strikes, disc_batch, dividend_batch, transacion_cost, riskaversion, paths_valid, sess, epochs)
        return loss_train_epoch

    # ---------------------------------------------------------------------- #
    # Function to compute the hedging strategies of a trained neural network
    # - Doesn't train the neural network, only outputs the hedging strategies
    def predict(self, paths, V_0, strikes, disc_paths, dividend_paths, transacion_cost, alpha, sess):
        sample_size = paths.shape[1]
        batch_size  = self.batch_size
        idx         = np.arange(sample_size)  # [0,1,...,sample_size-1]
        start       = dt.datetime.now()     # compute time
        strategy_pred = [] # hedging strategies

        # loop over sample size to do one complete epoch
        for i in range(int(sample_size/batch_size)):

            # mini-batch of paths (even if not training to not get memory issue)
            indices = idx[i*batch_size : (i+1)*batch_size]
            _, strategy = sess.run([self.hedging_err, self.deltas],
                                    {self.input           : paths[:,indices,:],
                                     self.strike          : strikes,
                                     self.alpha           : alpha,
                                     self.disc_tensor     : disc_paths,
                                     self.dividend_tensor : dividend_paths,
                                     self.transacion_cost : transacion_cost,
                                     self.portfolio       : V_0[indices]})

            # Append the batch of hedging strategies
            strategy_pred.append(strategy)
        return np.concatenate(strategy_pred,axis=1)
    
    def train_sage(self, paths, V_0, V_test, strikes, disc_batch, dividend_batch, transacion_cost, alpha, paths_valid, sess, epochs):
        sample_size       = paths.shape[1]               # total number of paths in the train set
        sample_size_valid = paths_valid.shape[1]
        batch_size        = self.batch_size
        idx               = np.arange(sample_size)       # [0,1,...,sample_size-1]
        idx_valid         = np.arange(sample_size_valid)
        start             = dt.datetime.now()            # Time-to-train
        self.loss_epochs  = 9999999*np.ones((epochs,2))      # Store the loss at the end of each epoch for the train
        valid_loss_best   = 999999999
        train_loss_best   = 999999999
        epoch             = 0

        # Loop for each epoch until the maximum number of epochs
        while (epoch < epochs):
            hedging_err_train = []  # Store hedging errors obtained for one complete epoch
            hedging_err_valid = []
            np.random.shuffle(idx)  # Randomize the dataset (not useful in this case since dataset is simulated iid)

            # loop over each batch size
            for i in range(int(sample_size/batch_size)):

                # Indexes of paths used for the mini-batch
                indices = idx[i*batch_size : (i+1)*batch_size]

                # SGD step
                _, hedging_err = sess.run([self.train, self.hedging_err],
                                               {self.input           : paths[:,indices,:],
                                                self.strike          : strikes,
                                                self.alpha           : alpha,
                                                self.disc_tensor     : disc_batch,
                                                self.dividend_tensor : dividend_batch,
                                                self.transacion_cost : transacion_cost,
                                                self.portfolio       : V_0[indices]})

                hedging_err_train.append(hedging_err)

            # 2) Evaluate performance on the valid set - we don't train
            for i in range(int(sample_size_valid/batch_size)):
                indices_valid = idx_valid[i*batch_size : (i+1)*batch_size]
                hedging_err_v = sess.run([self.hedging_err],
                                               {self.input           : paths_valid[:,indices_valid,:],
                                                self.strike          : strikes,
                                                self.alpha           : alpha,
                                                self.disc_tensor     : disc_batch,
                                                self.dividend_tensor : dividend_batch,
                                                self.transacion_cost : transacion_cost,
                                                self.portfolio       : V_test[indices_valid]})

                hedging_err_valid.append(hedging_err_v)

            # 3) Store the loss on the train and valid sets after each epoch
            self.loss_epochs[epoch,0] = self.loss_out_optim(np.concatenate(hedging_err_train), alpha, self.loss_type)
            self.loss_epochs[epoch,1] = self.loss_out_optim(np.reshape(np.concatenate(hedging_err_valid, axis=1), sample_size_valid), alpha, self.loss_type)

            # 4) Test if best epoch so far on valid set; if so, save model parameters.
            if((self.loss_epochs[epoch,1] < valid_loss_best)): #& (self.loss_epochs[epoch,1]>0)
                valid_loss_best = self.loss_epochs[epoch,1]
                #self.saver.save(sess, self.model_name + '.ckpt')
            if((self.loss_epochs[epoch,0] < train_loss_best)): #& (self.loss_epochs[epoch,1]>0)
                train_loss_best = self.loss_epochs[epoch,0]

            # Print the CVaR value at the end of each epoch
            if (epoch+1) % 1 == 0:
                print('Time elapsed:', dt.datetime.now()-start)
                print('Epoch %d, %s, Train: %.3f Valid: %.3f' % (epoch+1, self.loss_type,
                                                            self.loss_epochs[epoch,0], self.loss_epochs[epoch,1]))
            epoch+=1  # increment the epoch

        # End of training
        loss_stat = [train_loss_best,valid_loss_best]
        print("---Finished training results---")
        print('Time elapsed:', dt.datetime.now()-start)

        # Return the learning curve
        return loss_stat
    
    def sage(self, paths, V_0, V_test, strikes, disc_batch, dividend_batch, transacion_cost, riskaversion, paths_valid, sess, epochs):
        sess.run(tf.global_variables_initializer())
        loss = self.train_sage(paths, V_0, V_test, strikes, disc_batch, dividend_batch, transacion_cost, riskaversion, paths_valid, sess, epochs)
        return loss

    def restore(self, sess, checkpoint):
        self.saver.restore(sess, checkpoint)


def train_network(network, state_space, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, cash_constraint, constraint_max, loss_type, lr, dropout_par, paths, V_0, V_test, strikes, disc_batch,
dividend_batch, transacion_cost, riskaversion, paths_valid, epochs, display_plot, option, moneyness, isput, prepro_stock):
    
    # Function to train deep hedging algorithm
    """
    Input:
    # First layer of parameters 
    network          : Neural network architecture {"LSTM","RNNFNN","FFNN"}
    nbs_point_traj   : time steps 
    batch_size       : batch size {296,1000}
    nbs_input        : number of features
    nbs_units        : neurons per layer/cell
    nbs_assets       : number of hedging intruments
    cash_constraint   : lower bound for position in hedging instruments
    constraint_max   : upper bound for position in hedging instruments
    loss_type        : loss function {"CVaR","MSE","SMSE"}
    lr               : learning rate of the Adam optimizer
    dropout_par      : dropout regularization parameter 

    # Second layer of parameters
    paths            : Training set (normalized stock price and features)
    V_0              : Initial portfolio value for training set
    V_test           : Initial portfolio value for validation set
    strikes          : Strike value of the option hedged 
    disc_batch       : Risk-free rate update factor exp(h*r)
    dividend_batch   : Dividend yield update factor exp(h*d)
    transacion_cost  : Proportional transaction cost [0,1]
    riskaversion     : CVaR confidence level (0,1)
    paths_valid      : Validation set (normalized stock price and features)
    epochs           : Number of epochs, training iterations 
    option           : Option name {"Call","Put"}
    moneyness        : Moneyness option {"ATM","ITM","OTM"}
    isput            : Put or Call {True,False}
    prepro_stock     : Price preprocessing {"Log", "Log-moneyness", "Nothing"}

    Output:
    loss_train_epoch : Loss history per epochs

    """
    
    owd = os.getcwd()
    try:
        os.chdir(os.path.join(main_folder, f"models/Random_{63}/{option}/{moneyness}/TC_{transacion_cost*100}"))

        cash_constraint_name = f"CashC_{constraint_max}" if cash_constraint==True else "NoCashC"
        if loss_type == "CVaR":
            name = f"{network}_{state_space}_dropout_{str(int(dropout_par*100))}_{loss_type}_{str(int(riskaversion*100))}_TC_{ str(transacion_cost*100)}_{option}_{moneyness}_{cash_constraint_name}"
        else:
            name = f"{network}_{state_space}_dropout_{str(int(dropout_par*100))}_{loss_type}_TC_{ str(transacion_cost*100)}_{option}_{moneyness}_{cash_constraint_name}"

        #Re-defining the input based on the state space
        paths = paths if state_space=="Full" else paths[:,:,:8]
        paths_valid = paths_valid if state_space=="Full" else paths_valid[:,:,:8]
        nbs_input       = paths.shape[2]

        # Compile the neural network
        rl_network = DeepAgent(network, state_space, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, cash_constraint, constraint_max, loss_type, lr, dropout_par, isput, prepro_stock, name)

        print("-------------------------------------------------------------")
        print(name)
        print("-------------------------------------------------------------")

        # Start training
        print('---Training start---')
        with tf.Session() as sess:
            loss_train_epoch = rl_network.training(paths, V_0, V_test, strikes, disc_batch, dividend_batch, transacion_cost, riskaversion, paths_valid, sess, epochs)
        print('---Training end---')

        if display_plot == True:
            # Plot the learning curve on the train set
            sns.set_theme(font_scale=3.5)
            plt.rcParams['figure.figsize']=45,20
            plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["#8B2323", "#093885","#093885", "#FFB90F", "#FFB90F", "#CD5B45", "#FFB90F","#458B00"]) 
            sns.set_style("whitegrid")
            lin_nb_epoch = np.linspace(1, loss_train_epoch.shape[0], loss_train_epoch.shape[0])
            plt.plot(lin_nb_epoch[1:], loss_train_epoch[1:,0],label="Train error",linewidth=5.0)
            plt.plot(lin_nb_epoch[1:], loss_train_epoch[1:,1],label="Test error",linestyle='dashed',linewidth=5.0)
            plt.legend(fontsize="40")
            plt.xlabel("Epochs")
            plt.ylabel("Penalty function")
            plt.show()
    finally:
        #change dir back to original working directory (owd)
        os.chdir(owd)

    return loss_train_epoch

def retrain_network(network, state_space, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, cash_constraint, constraint_max, loss_type, lr, dropout_par, paths, V_0, V_test, strikes, disc_batch,
dividend_batch, transacion_cost, riskaversion, paths_valid, epochs, display_plot, option, moneyness, isput, prepro_stock):

    # Function to train deep hedging algorithm
    """
    Input:
    # First layer of parameters 
    network          : Neural network architecture {"LSTM","RNNFNN","FFNN"}
    nbs_point_traj   : time steps 
    batch_size       : batch size {296,1000}
    nbs_input        : number of features
    nbs_units        : neurons per layer/cell
    nbs_assets       : number of hedging intruments
    cash_constraint   : lower bound for position in hedging instruments
    constraint_max   : upper bound for position in hedging instruments
    loss_type        : loss function {"CVaR","MSE","SMSE"}
    lr               : learning rate of the Adam optimizer
    dropout_par      : dropout regularization parameter 

    # Second layer of parameters
    paths            : Training set (normalized stock price and features)
    V_0              : Initial portfolio value for training set
    V_test           : Initial portfolio value for validation set
    strikes          : Strike value of the option hedged 
    disc_batch       : Risk-free rate update factor exp(h*r)
    dividend_batch   : Dividend yield update factor exp(h*d)
    transacion_cost  : Proportional transaction cost [0,1]
    riskaversion     : CVaR confidence level (0,1)
    paths_valid      : Validation set (normalized stock price and features)
    epochs           : Number of epochs, training iterations 
    option           : Option name {"Call","Put"}
    moneyness        : Moneyness option {"ATM","ITM","OTM"}
    isput            : Put or Call {True,False}
    prepro_stock     : Price preprocessing {"Log", "Log-moneyness", "Nothing"}

    Output:
    loss_train_epoch : Loss history per epochs

    """

    owd = os.getcwd()
    try:
        #os.chdir(os.path.join(main_folder, f"models/Random_{nbs_point_traj-1}/{option}/{moneyness}/TC_{transacion_cost*100}"))
        os.chdir(os.path.join(main_folder, f"models/Random_{63}/{option}/{moneyness}/TC_{transacion_cost*100}"))

        cash_constraint_name = f"CashC_{constraint_max}" if cash_constraint==True else "NoCashC"
        if loss_type == "CVaR":
            name = f"{network}_{state_space}_dropout_{str(int(dropout_par*100))}_{loss_type}_{str(int(riskaversion*100))}_TC_{ str(transacion_cost*100)}_{option}_{moneyness}_{cash_constraint_name}"
        else:
            name = f"{network}_{state_space}_dropout_{str(int(dropout_par*100))}_{loss_type}_TC_{ str(transacion_cost*100)}_{option}_{moneyness}_{cash_constraint_name}"

        #Re-defining the input based on the state space
        paths = paths if state_space=="Full" else paths[:,:,:8]
        paths_valid = paths_valid if state_space=="Full" else paths_valid[:,:,:8]
        nbs_input       = paths.shape[2]

        # Compile the neural network
        rl_network = DeepAgent(network, state_space, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, cash_constraint, constraint_max, loss_type, lr, dropout_par, isput, prepro_stock, name)

        print("-------------------------------------------------------------")
        print(name)
        print("-------------------------------------------------------------")

        # Start training
        print('---Training start---')
        with tf.Session() as sess:
            rl_network.restore(sess, f"{name}.ckpt")
            loss_train_epoch = rl_network.train_deephedging(paths, V_0, V_test, strikes, disc_batch, dividend_batch, transacion_cost, riskaversion, paths_valid, sess, epochs)
        print('---Training end---')

        if display_plot == True:
            # Plot the learning curve on the train set, i.e. the CVaR on the train set at the end of each epoch
            lin_nb_epoch = np.linspace(1, loss_train_epoch.shape[0], loss_train_epoch.shape[0])
            plt.figure(figsize=(5, 5), dpi=100)
            plt.plot(lin_nb_epoch[1:], loss_train_epoch[1:,0],label="Train error")
            plt.plot(lin_nb_epoch[1:], loss_train_epoch[1:,1],label="Test error")
            plt.title(f"Loss function: {loss_type}")
            plt.legend(loc='upper center', shadow=True, fontsize='x-large')
            plt.show()
    finally:
        #change dir back to original working directory (owd)
        os.chdir(owd)
    
    return loss_train_epoch


def network_inference(network, state_space, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, cash_constraint, constraint_max, loss_type, lr, dropout_par, V_test, strikes, disc_batch,
dividend_batch, transacion_cost, riskaversion, paths_valid, option, moneyness, isput, prepro_stock, backtest):
    
    # Function to test deep hedging algorithm
    """
    Input:
    # First layer of parameters 
    network          : Neural network architecture {"LSTM","RNNFNN","FFNN"}
    nbs_point_traj   : time steps 
    batch_size       : batch size {296,1000}
    nbs_input        : number of features
    nbs_units        : neurons per layer/cell
    nbs_assets       : number of hedging intruments
    cash_constraint   : lower bound for position in hedging instruments
    constraint_max   : upper bound for position in hedging instruments
    loss_type        : loss function {"CVaR","MSE","SMSE"}
    lr               : learning rate of the Adam optimizer
    dropout_par      : dropout regularization parameter 

    # Second layer of parameters
    V_test           : Initial portfolio value for validation set
    strikes          : Strike value of the option hedged 
    disc_batch       : Risk-free rate update factor exp(h*r)
    dividend_batch   : Dividend yield update factor exp(h*d)
    transacion_cost  : Proportional transaction cost [0,1]
    riskaversion     : CVaR confidence level (0,1)
    paths_valid      : Validation set (normalized stock price and features)
    option           : Option name {"Call","Put"}
    moneyness        : Moneyness option {"ATM","ITM","OTM"}
    isput            : Put or Call {True,False}
    prepro_stock     : Price preprocessing {"Log", "Log-moneyness", "Nothing"}
    backtest         : Parameter backtest {"True","False"}

    Output:
    deltas           : Position in the hedging instruments

    """
    
    owd = os.getcwd()
    
    #os.chdir(os.path.join(main_folder, f"models/Random_{nbs_point_traj-1}/{option}/{moneyness}/TC_{transacion_cost*100}"))
    os.chdir(os.path.join(main_folder, f"models/Random_{63}/{option}/{moneyness}/TC_{transacion_cost*100}"))
    cash_constraint_name = f"CashC_{constraint_max}" if cash_constraint==True else "NoCashC"
    if loss_type == "CVaR":
        name = f"{network}_{state_space}_dropout_{str(int(dropout_par*100))}_{loss_type}_{str(int(riskaversion*100))}_TC_{ str(transacion_cost*100)}_{option}_{moneyness}_{cash_constraint_name}"
    else:
        name = f"{network}_{state_space}_dropout_{str(int(dropout_par*100))}_{loss_type}_TC_{ str(transacion_cost*100)}_{option}_{moneyness}_{cash_constraint_name}"

    #Re-defining the input based on the state space
    paths = paths if state_space=="Full" else paths[:,:,:8]
    paths_valid = paths_valid if state_space=="Full" else paths_valid[:,:,:8]
    nbs_input       = paths.shape[2]

    # Compile the neural network
    rl_network = DeepAgent(network, state_space, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, cash_constraint, constraint_max, loss_type, lr, dropout_par, isput, prepro_stock, name)

    print("-------------------------------------------------------------")
    print(name)
    print("-------------------------------------------------------------")

    # Start training
    print('---Inference start---')
    with tf.Session() as sess:
        rl_network.restore(sess, f"{name}.ckpt")
        deltas  = rl_network.predict(paths_valid, V_test, strikes, disc_batch, dividend_batch, transacion_cost, riskaversion, sess)
        os.chdir(owd)
        if backtest==True:
            os.chdir(os.path.join(main_folder, f"data/results/Backtest/Random_{nbs_point_traj-1}/{option}/{moneyness}/TC_{transacion_cost*100}"))
        else:
            #os.chdir(os.path.join(main_folder, f"data/results/Training/Random_{nbs_point_traj-1}/{option}/{moneyness}/TC_{transacion_cost*100}"))
            os.chdir(os.path.join(main_folder, f"data/results/Training/Random_{63}/{option}/{moneyness}/TC_{transacion_cost*100}"))
        
        np.save(f"{name}",np.transpose(deltas[:,:,0]))

    print('---Inference end---')
    os.chdir(owd)

    return np.transpose(deltas[:,:,0])


def sage_network(network, state_space, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, cash_constraint, constraint_max, loss_type, lr, dropout_par, paths, V_0, V_test, strikes, disc_batch,
dividend_batch, transacion_cost, riskaversion, paths_valid, epochs, display_plot, option, moneyness, isput, prepro_stock):
    
    # Function to train deep hedging algorithm
    """
    Input:
    # First layer of parameters 
    network          : Neural network architecture {"LSTM","RNNFNN","FFNN"}
    nbs_point_traj   : time steps 
    batch_size       : batch size {296,1000}
    nbs_input        : number of features
    nbs_units        : neurons per layer/cell
    nbs_assets       : number of hedging intruments
    constraint_min   : lower bound for position in hedging instruments
    constraint_max   : upper bound for position in hedging instruments
    loss_type        : loss function {"CVaR","MSE","SMSE"}
    lr               : learning rate of the Adam optimizer
    dropout_par      : dropout regularization parameter 

    # Second layer of parameters
    paths            : Training set (normalized stock price and features)
    V_0              : Initial portfolio value for training set
    V_test           : Initial portfolio value for validation set
    strikes          : Strike value of the option hedged 
    disc_batch       : Risk-free rate update factor exp(h*r)
    dividend_batch   : Dividend yield update factor exp(h*d)
    transacion_cost  : Proportional transaction cost [0,1]
    riskaversion     : CVaR confidence level (0,1)
    paths_valid      : Validation set (normalized stock price and features)
    epochs           : Number of epochs, training iterations 
    option           : Option name {"Call","Put"}
    moneyness        : Moneyness option {"ATM","ITM","OTM"}
    isput            : Put or Call {True,False}
    prepro_stock     : Price preprocessing {"Log", "Log-moneyness", "Nothing"}

    Output:
    loss_train_epoch : Loss history per epochs

    """
    
    owd = os.getcwd()
    try:
        os.chdir(os.path.join(main_folder, f"models/Random_{nbs_point_traj-1}/{option}/{moneyness}/TC_{transacion_cost*100}"))
        #os.chdir(os.path.join(main_folder, f"models/Random_{63}/{option}/{moneyness}/TC_{transacion_cost*100}"))

        if loss_type == "CVaR":
            name = f"{network}_dropout_{str(int(dropout_par*100))}_{loss_type}_{str(int(riskaversion*100))}_TC_{ str(transacion_cost*100)}_{option}_{moneyness}_cons_{constraint_max}"
        else:
            name = f"{network}_dropout_{str(int(dropout_par*100))}_{loss_type}_TC_{ str(transacion_cost*100)}_{option}_{moneyness}_cons_{constraint_max}"

        print("---------------------------------------------------------------------")
        print(f"---------------------- SAGE values - {loss_type} ----------------------------")
        print("---------------------------------------------------------------------")

        cases  = [[0,1,2,3,4,5,6,7],[0,1,2,3,4,5,6],[0,1,2,3,4,5,7],[0,1,2,3,4,5],[0,1,2,3,4,6,7],[0,1,2,3,4,6],
                    [0,1,2,3,4,7],[0,1,2,3,4],[0,1,2,3,5,6,7],[0,1,2,3,5,6],[0,1,2,3,5,7],[0,1,2,3,5],[0,1,2,3,6,7],
                    [0,1,2,3,6],[0,1,2,3,7],[0,1,2,3],[0,1,2,4,5,6,7],[0,1,2,4,5,6],[0,1,2,4,5,7],[0,1,2,4,5],[0,1,2,4,6,7],
                    [0,1,2,4,6],[0,1,2,4,7],[0,1,2,4],[0,1,2,5,6,7],[0,1,2,5,6],[0,1,2,5,7],[0,1,2,5],[0,1,2,6,7],[0,1,2,6],
                    [0,1,2,7],[0,1,2],[0,1,3,4,5,6,7],[0,1,3,4,5,6],[0,1,3,4,5,7],[0,1,3,4,5],[0,1,3,4,6,7],[0,1,3,4,6],
                    [0,1,3,4,7],[0,1,3,4],[0,1,3,5,6,7],[0,1,3,5,6],[0,1,3,5,7],[0,1,3,5],[0,1,3,6,7],[0,1,3,6],[0,1,3,7],[0,1,3],
                    [0,1,4,5,6,7],[0,1,4,5,6],[0,1,4,5,7],[0,1,4,5],[0,1,4,6,7],[0,1,4,6],[0,1,4,7],[0,1,4],[0,1,5,6,7],[0,1,5,6],
                    [0,1,5,7],[0,1,5],[0,1,6,7],[0,1,6],[0,1,7],[0,1]]
        sage_loss = []
        for i in range(len(cases)):
            print(f"-------- Case {i} out of {len(cases)} --------")
            train_input = paths[:,:,cases[i]]
            test_input  = paths_valid[:,:,cases[i]]
            nbs_input   = train_input.shape[2]
            
            # Compile the neural network
            rl_network = DeepAgent(network, state_space, nbs_point_traj, batch_size, nbs_input, nbs_units, nbs_assets, cash_constraint, constraint_max, loss_type, lr, dropout_par, isput, prepro_stock, name)

            # Start training
            print('---Training start---')
            with tf.Session() as sess:
                loss_train_epoch = rl_network.sage(train_input, V_0, V_test, strikes, disc_batch, dividend_batch, transacion_cost, riskaversion, test_input, sess, epochs)
            print('---Training end---')
            sage_loss.append(loss_train_epoch)

    finally:
        #change dir back to original working directory (owd)
        os.chdir(owd)

    return sage_loss