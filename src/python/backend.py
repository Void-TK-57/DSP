# base libs
import sys
import os
import logging
import logging.config
import json
import itertools

#  main libs
import numpy as np
import pandas as pd
import scipy.stats as sts
# plot libs
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import ternary
#import seaborn as sns; sns.set()

# set logger
from os import path
log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logger.conf')
logging.config.fileConfig(fname=log_file_path, disable_existing_loggers=True)
logger = logging.getLogger('backend')


logger.info("Importing Machine Learning Libs")
# machine learning libs
from sklearn.covariance import empirical_covariance
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.manifold import Isomap
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve, learning_curve, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import sklearn.utils._cython_blas
import sklearn.neighbors.typedefs
import sklearn.neighbors.quad_tree
import sklearn.tree
import sklearn.tree._utils

# neural network lib
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Flatten, MaxPool2D, Conv2D, Dense, Reshape, Dropout
from keras.utils import np_utils, plot_model, to_categorical
from keras.datasets import mnist
from keras.optimizers import SGD, Adam, Nadam, RMSprop
from keras import losses
from keras.wrappers.scikit_learn import KerasClassifier
from keras.metrics import CategoricalAccuracy
from keras.callbacks import Callback

# deisgn of experiments libs
logger.info("Importing Design Libs")
from dexpy.simplex_centroid import build_simplex_centroid
from dexpy.simplex_lattice import build_simplex_lattice
from dexpy.model import ModelOrder
import pyDOE2
        
# function to create a artifical neural network model
def build_model(input_layer, output_layer, hidden_layers, activation_functions, loss, metrics, optimizer, learning_rate):
    # create model
    model = Sequential()
    # add first layer
    model.add( Dense( hidden_layers[0], input_shape = ( input_layer, ), activation = activation_functions[0] ) )
    # for every other layer
    for i in range(1, len(hidden_layers)):
        # add layers
        model.add( Dense( hidden_layers[i], activation = activation_functions[i] ) )
    # add last layer
    model.add( Dense( output_layer, activation = activation_functions[-1] ) )
    # check optimizers
    optimizers = {'sgd': SGD, 'adam': Adam, 'nadam': Nadam, 'RMSprop': RMSprop}
    # compile it
    model.compile(loss = loss, optimizer = optimizers[optimizer](learning_rate=learning_rate), metrics=metrics )
    # return model
    return model

# r_square metric to be passed in keras train function
def r_square(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def det_coeff(y_true, y_pred):
    SS_reg =  K.sum(K.square( y_pred - K.mean(y_true) )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( SS_reg/(SS_tot) )


# main design regression class
class Design:
    """
    Design Regression.

    Design class accepts a dataset and then creates a regression based on the 
    degree given, also calculating other statistical information. Then it can
    be used to predict values based on the regression and plot the dataset.
        
    Args:
        data (Dataframe): the dataset used on the regression
        dependent_value (string): name of the column in the dataset to be used in the analisys
        confiance_interval (int): value of confiance used to calculate the interval (between 0 and 1)
        degree (int): degree of the regression
        use_pure_error (bool): whether to use the pure error as an estimative of the variance
        include_bias (bool): whether to include the zero (Used for Mixture Design)
        interaction_only (bool): whether to include the interacions only in higher degree regressions (Used for Mixture Design)
        use_log (bool): whether to show on console the results and analisys of the design

    """


    def __init__(self, data, dependent_value, confiance_interval = 0.95, degree = 1, use_pure_error = False, include_bias = True, interaction_only = False, use_log = False):
        try:
            self.use_log = use_log
            self.degree = degree
            self.use_pure_error = use_pure_error
            self.include_bias = include_bias
            self.interaction_only = interaction_only

            self.data = data
            self._log("Data Pura:\n" + str(self.data) )
            self.confiance_interval_value = confiance_interval
            self._log("Confiance Value:\t" + str(self.confiance_interval_value) )
            self.dependent_value = dependent_value
            self._log( "Varivel Dependente\t:"+ str(self.dependent_value)  )
            self.independent_values = list(self.data)
            self.independent_values.remove(self.dependent_value)
            self._log("Varivel Independente\t:" + str(self.independent_values) )

            # slice the x and y values
            self.y_values = self.data[self.dependent_value]
            self._log("Valores Dependentes\n:" + str(self.y_values) )
            self.x_values = self.data[self.independent_values]
            self._log("Valores Independentes\n:" + str(self.x_values) )
        
            # create model by using the degre feature into linera regression
            self.regression_model = LinearRegression(fit_intercept=self.include_bias)
            self.polynomial_feature = PolynomialFeatures(degree=degree, include_bias=self.include_bias, interaction_only=interaction_only)
            self.model = make_pipeline( self.polynomial_feature, self.regression_model)

            # fit model to the inputs of the Desing
            self.model.fit(self.x_values, self.y_values)
            
            self.coefficients = self.get_coefficients()
            self._log("Coeficientes:\n"+ str( self.coefficients ) )
        
            # predict values
            self.y_predict = self.predict(self.x_values)
            self._log("Valores Calculados\n:"+ str( self.y_predict) )
        
            # residuos
            self.residuals = self.y_values - self.y_predict      
            self._log("Residuals:\n"+ str( self.residuals) ) 

            # mean squared error     
            self.mse = mean_squared_error(self.y_values, self.y_predict)
            self._log("Mean Squared Error:\t"+ str( self.mse) )
        
            # calculate variance table
            self.anova = self.calculate_anova()
            self._log("ANOVA:\n"+ str( self.anova) )
        
            # Explained Variance
            self.R = explained_variance_score(self.y_values, self.y_predict)
            self._log("R:\t"+ str( self.R) )

            # maxium explained variance
            self.max_R = 1 - self.anova["Soma Quadrática"]["Erro Puro"]/self.anova["Soma Quadrática"]["Total"]
            self._log("Max R:\t"+ str( self.max_R) )

            # variance estimate
            self.variance = self.get_variance(self.use_pure_error)
            self._log("Varianca Estimada:\t"+ str( self.variance) )
        
            # calculate covariance matrix
            self.covariance_matrix = self.get_variance_matrix()
            self._log("Matrix de Covariancia:\n"+ str( self.covariance_matrix) )

            # calculate confiance_interval
            self.confiance_interval = self.get_confiance_interval(self.confiance_interval_value)
            self._log("Intervalo de Confianca:\n"+ str( self.confiance_interval) )

            # F value
            self.F_value = self.anova["Média Quadrática"]["Regressão"]/self.anova["Média Quadrática"]["Resíduos"]
            self._log("F value:\t"+ str( self.F_value) )
        

        except Exception as err:
            self.valid = False
            logger.error("Error Creating Design: " + str(err) )
        else:
            self.valid = True

    def _log(self, message):
        if self.use_log:
            logger.debug(message)
        #logger.debug("Matriz de Covarianca:\t", self.covariance_matrix)
        # self.plot()

    # method to predict y values based on x dataframe passed
    def predict(self, x):
        """Calculate the Predict Values of the Model based on the Dataframe x given.
        
        Args:
            x (Dataframe): the values of the independent factors to be used in the prediction

        Returns:
            predict (Series): Predicted values for each line of independent values of x
        """
        y = self.model.predict(x)
        return pd.Series(y, name = self.y_values.name, index = x.index)

    # method to get dataframe of coefficients of the regression
    def get_coefficients(self):
        """Return a Series with the Coefficients of the Model."""
        coefficients = self.regression_model.coef_.copy()
        # if bias included change first coeficient (bias) to the intercept
        if self.include_bias:
            coefficients[0] = self.regression_model.intercept_
        # get features names
        index_list = self.polynomial_feature.get_feature_names(self.independent_values)
        # return a series with the coefficients and their names
        return pd.Series(coefficients, index = index_list )

    # method to plot the data
    def plot(self, original_values = True, figure = None):
        """Plot the Regression of the Model.
        
        Args:
            origina_values (bool): Boolean that indicates if the original values used on the model should be added on the model.
            figure (Figure): Matplotlib Figure to add the plot, if None given, a figure is created.

        Returns:
            plotted (bool): Boolean that indicates if the model could be plotted.
        """
        
        # check number of independe_values
        if len(self.independent_values) != 1:
            return False

        show_plot= False
        # check figure
        if figure is None:
            show_plot = True
            figure = plt.figure()

        ax = figure.add_subplot(111)
        # set labels adn title
        ax.set_xlabel(self.independent_values[0])
        ax.set_ylabel(self.dependent_value)
        ax.set_title("Regressão")

        
        # x and y of the regression
        x = np.linspace( np.min(self.x_values), np.max(self.x_values) )

        y = self.predict( x )

        # plot
        ax.plot(x, y)

        # if original vaues
        if original_values:
            ax.scatter(self.x_values, self.y_values, c = 'red', marker = '.')


        if show_plot:
            plt.show()

        return True

        # method to plot the residues
    
    # method to plot residuals
    def plot_residuals(self, figure = None, use_y_values = False, use_pred_values = False):
        """Plot the Residues of the Regression of the Model.
        
        Args:
            figure (Figure): Matplotlib Figure to add the plot, if None given, a figure is created.
            use_y_values (bool): boolean wheter one of the axis should be the dependent values.
            use_pred_values (bool): boolean wheter one of the axis should be the dependent values calculated by the model.

        Returns:
            plotted (bool): Boolean that indicates if the model could be plotted.
        """
        
        # check number of independe_values
        #if len(self.independent_values) != 1:
        #    return False

        show_plot= False
        # check figure
        if figure is None:
            show_plot = True
            figure = plt.figure()
        ax = figure.add_subplot(111)

        # set labels adn title
        if use_y_values and use_pred_values:
            y_label = "Valores Previstos"
            x_label = "Valores Observados"
            y_plot = self.y_predict
            x_plot = self.y_values
            line_x = [np.min(x_plot), np.max(x_plot)]
            line_y = [np.min(x_plot), np.max(x_plot)]
            
        elif use_y_values:
            y_label = "Resíduos"
            x_label = "Valores Observados"
            y_plot = self.residuals
            x_plot = self.y_values
            line_x = [np.min(x_plot), np.max(x_plot)]
            line_y = [0.0, 0.0]

        elif use_pred_values:
            y_label = "Resíduos"
            x_label = "Valores Previstos"
            y_plot = self.residuals
            x_plot = self.y_predict
            line_x = [np.min(x_plot), np.max(x_plot)]
            line_y = [0.0, 0.0]

        else:
            y_label = "Resíduos"
            x_label = "Ensaios"
            y_plot = self.residuals
            x_plot = list(self.residuals.index)
            line_x = [x_plot[0], x_plot[-1]]
            line_y = [0.0, 0.0]



        ax.set_title(y_label + " x " + x_label)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)

        ax.scatter(x_plot, y_plot)
        ax.plot(line_x, line_y, color='red')
        if show_plot:
            plt.show()

        return True

    # method to get the dataframe with the variance table (ANOVA)
    def calculate_anova(self):
        """Return the variance table of the design (ANOVA)."""
        # copy data
        data = self.data.copy()
        # group the data by dependent value
        group_values = data.groupby(self.independent_values)
        mean_values = group_values.mean()

        # number of distincts smaples (aka m)
        m = mean_values.shape[0]
        # number of variables (aka n)
        n = self.y_values.size
        # number of parameters (aka p)
        p = len(self.coefficients)

        y_mean_per_level = data.copy()
        # original y and predict
        y_values = self.y_values.copy()
        y_mean = data.copy()[self.dependent_value]
        y_mean = y_values.mean()
        y_predict = self.y_predict.copy()

        # change y_per_level by mean
        for key in mean_values.index:
            cond = y_mean_per_level[self.independent_values] == key
            index = cond.all(axis=1)
            #print(mean_values.loc[key][self.dependent_value])
            y_mean_per_level.loc[index, self.dependent_value] = mean_values.loc[key, self.dependent_value]
            #print(data[index])
        y_mean_per_level = y_mean_per_level[self.dependent_value]

        # change in data to group by the independent values, get the sum per levl (group) and then get the sum of all levels
        total = np.square(y_values - y_mean)
        data[self.dependent_value] = total
        # squared sum, degrees of freedom and squared mean of total
        total_ss = data.groupby(self.independent_values).sum().sum()[self.dependent_value]
        total_df = n-1

        pure_error = np.square(y_values - y_mean_per_level)
        data[self.dependent_value] = pure_error
        # squared sum, degrees of freedom and squared mean of pure error
        pure_error_ss = data.groupby(self.independent_values).sum().sum()[self.dependent_value]
        pure_error_df = n-m
        pure_error_ms = pure_error_ss/pure_error_df

        residuos = np.square(y_values - y_predict)
        data[self.dependent_value] = residuos
        # squared sum, degrees of freedom and squared mean of residuos
        residuos_ss = data.groupby(self.independent_values).sum().sum()[self.dependent_value]
        residuos_df = n-p
        residuos_ms = residuos_ss/residuos_df

        # squared sum, degrees of freedom and squared mean of regression
        regression_ss = np.sum(np.square(y_predict - y_mean))
        regression_df = p-1
        regression_ms = regression_ss/regression_df

        # squared sum, degrees of freedom and squared mean of lack of fit
        lack_of_fit_ss = np.sum( np.square( y_predict - y_mean_per_level) )
        lack_of_fit_df = m-p
        lack_of_fit_ms = lack_of_fit_ss/lack_of_fit_df

        # F value
        F = regression_ms/residuos_ms

        # p value
        p = 1 - sts.f.cdf(F, regression_df, residuos_df)

        # table
        matrix = [[regression_ss  , regression_df , regression_ms ,    F  ,    p  ], 
                  [ residuos_ss   , residuos_df   , residuos_ms   , np.nan, np.nan],
                  [ lack_of_fit_ss, lack_of_fit_df, lack_of_fit_ms, np.nan, np.nan], 
                  [ pure_error_ss , pure_error_df , pure_error_ms , np.nan, np.nan],  
                  [   total_ss    ,    total_df   ,    np.nan     , np.nan, np.nan]]


        # indexs and columns of the table
        indexs = ["Regressão", "Resíduos", "Falta de Ajuste", "Erro Puro", "Total"]
        columns = ["Soma Quadrática", "Graus de Liberdade", "Média Quadrática", "F", "p"]
        
        # create dataframe
        anova = pd.DataFrame(matrix, index = indexs, columns = columns)

        # return the dataframe
        return anova

    # method to get the matrix of variance
    def get_variance_matrix(self):
        """Return the variance matrix."""
        # Calcualte X matrix by using polynomial features
        X = self.polynomial_feature.fit_transform(self.data[self.independent_values])
        # return the variance matrix based on the actual formula
        return np.linalg.inv(X.T @ X)*self.variance

    # method to get the variance of the regression itself
    def get_variance(self, use_pure_error):
        """Return the estimated variance.
        
        Args:
            use_pure_error (bool): Whether to use the pure error as an estimative for the variance or the residues.

        Returns:
            variance (float) : estimated variance.
        """
        variance = self.anova["Média Quadrática"]["Erro Puro"]
        if not use_pure_error or variance == np.nan:
            variance = self.anova["Média Quadrática"]["Resíduos"]
        return variance

    # method to get the dataframe with the confiance interval for each coefficient
    def get_confiance_interval(self, confiance):
        """Return the Interval of the values for each Coefficient on a certain confiance per centage.
        
        Args:
            confiance (float): confiance of the interval (from 0 to 1)

        Returns:
            Interval (Dataframe) : Dataframe with the minimum and maximum value of the inteval for each coefficient.
        """
        
        # get standard derivation
        std = np.diagonal( np.sqrt(np.abs(self.covariance_matrix)) )
        std = pd.Series(std, index = self.coefficients.index, name = "Desvio")
        # t distribution
        t = sts.t.ppf( 1.0 - (1.0 - confiance)/2.0, self.y_values.size - 2)

        min_ = self.coefficients - t*std
        max_ = self.coefficients + t*std
       # creeate dataframe from min e max interval
        df = pd.DataFrame({"Desvio": std,"Mínimo": min_, "Máximo": max_})
        return df

    # method to change degre
    def change_degree(self, degree):
        self.__init__(data=self.data, dependent_value=self.dependent_value, confiance_interval = self.confiance_interval_value, degree = degree, use_pure_error = self.use_pure_error, include_bias = self.include_bias, interaction_only = self.interaction_only, use_log = self.use_log)

# factorial design main class
class Factorial_Design(Design):

    def __init__(self, data, dependent_value, degree = 1, *args, **kwargs):
        super().__init__(data, dependent_value, degree = degree, *args, **kwargs)

    # method to plot the data
    def plot(self, original_values = True, figure = None):
        """Plot the Surface Response of the Design.
        
        Args:
            origina_values (bool): Boolean that indicates if the original values used on the model should be added on the model.
            figure (Figure): Matplotlib Figure to add the plot, if None given, a figure is created.

        Returns:
            plotted (bool): Boolean that indicates if the model could be plotted.
        """

        # check number of independe_values
        if len(self.independent_values) != 2:
            return False
        # check figure
        if figure is None:
            #create figure
            fig = plt.figure()
            #get axis
            ax = fig.add_subplot(111, projection="3d")
        else:
            # create axis
            ax = figure.add_subplot(111, projection="3d")

        # set labels adn title
        ax.set_xlabel(self.independent_values[0])
        ax.set_ylabel(self.independent_values[1])
        ax.set_zlabel(self.dependent_value)
        
        # x and y of the regression
        x = np.linspace( np.min(self.x_values[ self.independent_values[0] ]), np.max(self.x_values[ self.independent_values[0] ]) , num=20 )
        y = np.linspace( np.min(self.x_values[ self.independent_values[1] ]), np.max(self.x_values[ self.independent_values[1] ]) , num=20 )

        # mesh of x and y
        x_mesh, y_mesh = np.meshgrid(x, y)

        #shape then again as colmun vectors
        x_mesh_col = x_mesh.reshape(len(x)*len(y), 1)
        y_mesh_col = y_mesh.reshape(len(x)*len(y), 1)

        # concatenate them to create a matrix of independent values
        X = np.concatenate((x_mesh_col, y_mesh_col), 1)

        # create a dataframe to calculate the dependent values
        X = pd.DataFrame(X, columns = self.independent_values)

        # calculate the dependent values
        z = self.predict( X )
        # reshape as matrix
        z = z.values.reshape(len(x), len(y))

        # plot
        surface = ax.plot_surface(x_mesh, y_mesh, z, cmap=cm.viridis)
        # set title
        ax.set_title('Superfíce de Resposta')

        if figure is None:
            plt.show()

        return True

    # static method to create a matrix design
    @staticmethod
    def design(n_factors):
        """Calculate the Matrix of a 2-Level Factorial Design.
        
        Args:
            n_factors (int): number of factors

        Returns:
            matrix (Numpy Matrix): matrix of the design
        """
        return pyDOE2.ff2n(n_factors)

    # static method to create a matrix design
    @staticmethod
    def general_design(level_per_factors):
        """Calculate the Matrix of a Full-Level Factorial Design.
        
        Args:
            level_per_factors (list): number of levels for each factor

        Returns:
            matrix (Numpy Matrix): matrix of the design
        """
        return pyDOE2.fullfact(level_per_factors)

# central composite main class
class Central_Composite_Design(Design):
    
    def __init__(self, data, dependent_value, degree = 2, *args, **kwargs):
        super().__init__(data, dependent_value, degree = degree, *args, **kwargs)

    # method to plot the data
    def plot(self, original_values = True, figure = None):
        """Plot the Surface Response of the Design.
        
        Args:
            origina_values (bool): Boolean that indicates if the original values used on the model should be added on the model.
            figure (Figure): Matplotlib Figure to add the plot, if None given, a figure is created.

        Returns:
            plotted (bool): Boolean that indicates if the model could be plotted.
        """

        # check number of independe_values
        if len(self.independent_values) != 2:
            return False
        # check figure
        if figure is None:
            #create figure
            fig = plt.figure()
            #get axis
            ax = fig.add_subplot(111, projection="3d")
        else:
            # create axis
            ax = figure.add_subplot(111, projection="3d")

        # set labels adn title
        ax.set_xlabel(self.independent_values[0])
        ax.set_ylabel(self.independent_values[1])
        ax.set_zlabel(self.dependent_value)
        
        # x and y of the regression
        x = np.linspace( np.min(self.x_values[ self.independent_values[0] ]), np.max(self.x_values[ self.independent_values[0] ]) , num=20 )
        y = np.linspace( np.min(self.x_values[ self.independent_values[1] ]), np.max(self.x_values[ self.independent_values[1] ]) , num=20 )

        # mesh of x and y
        x_mesh, y_mesh = np.meshgrid(x, y)

        #shape then again as colmun vectors
        x_mesh_col = x_mesh.reshape(len(x)*len(y), 1)
        y_mesh_col = y_mesh.reshape(len(x)*len(y), 1)

        # concatenate them to create a matrix of independent values
        X = np.concatenate((x_mesh_col, y_mesh_col), 1)

        # create a dataframe to calculate the dependent values
        X = pd.DataFrame(X, columns = self.independent_values)

        # calculate the dependent values
        z = self.predict( X )
        # reshape as matrix
        z = z.values.reshape(len(x), len(y))

        # plot
        surface = ax.plot_surface(x_mesh, y_mesh, z, cmap=cm.viridis)
        # set title
        ax.set_title('Superfíce de Resposta')

        if figure is None:
            plt.show()

        return True

    # static method to create a matrix design
    @staticmethod
    def design(n_factors, center = [4, 4], alpha = 'orthogonal', face = 'ccc'):
        """Calculate the Matrix of a Central Composite Design.
        
        Args:
            n_factors (int): number of factors
            center (2-tuple): number of center points for factorial and star respectively
            alpha (string): alpha paramter ('orthogonal'/'o' or 'rotatable'/'r')
            face (string): type of face ('circumscribed'/'ccc' or 'inscribed'/'cci' or 'faced'/'ccf')

        Returns:
            matrix (Numpy Matrix): matrix of the design
        """
        return pyDOE2.ccdesign(n_factors, center, alpha, face)

# mixture main class
class Mixture_Design(Design):

    def __init__(self, data, dependent_value, degree= 3, include_bias = False, interaction_only = True, *args, **kwargs):
        super().__init__(data, dependent_value, degree = degree, include_bias = include_bias, interaction_only = interaction_only, *args, **kwargs)

    # method to plot the data
    def plot(self, scale = 30, figure = None):
        """Plot the Triangular Surface of the Design.
        
        Args:
            scale (int): Scale of the axis
            figure (Figure): Matplotlib Figure to add the plot, if None given, a figure is created.

        Returns:
            plotted (bool): Boolean that indicates if the model could be plotted.
        """
        #check if number of variables = 3 (else, returns False)
        if len(self.independent_values) != 3:
            #return None
            return False

        # check if figure was not given
        if figure is None:
            # then create figure and ternary ax
            fig, tax = ternary.figure(scale=scale)
        else:
            # else, create a ax from the figure, and then create a ternary ax from it
            ax = figure.add_subplot(111)
            tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale, permutation=None)

        # method to calculate the 
        def calculate(x):
            # convert to numpy array
            x = np.array(x)
            # reshape x
            x = x.reshape(1, len(x))
            # create as dataframe
            x_dataframe = pd.DataFrame(x, columns = self.independent_values)
            # get predicted value based on the x dataframe
            y_predicted = self.predict(x_dataframe)
            # return the first (and only) value of the series
            return y_predicted[0]
        
        #create heatmap based on the self.calculate method (which calculate the value of y based on the B values calculated and the x values passed)
        #tax.heatmapf(self.calculate, boundary=True, style="triangular", cmap=cm.plasma)
        tax.heatmapf(calculate, boundary=True, style="triangular", cmap=cm.plasma)
        # set boundary
        tax.boundary(linewidth=1.0)
        # set title
        tax.set_title("Gráfico")
        #set axis label
        tax.left_axis_label(self.independent_values[0], fontsize=10)
        tax.right_axis_label(self.independent_values[1], fontsize=10)
        tax.bottom_axis_label(self.independent_values[2], fontsize=10)
        # check if figure is None
        if figure is None:
            #show plot
            tax.show()
        #return True
        return True

    # static method to create a matrix design
    @staticmethod
    def simplex_centroid_design(n_factors):
        """Calculate the Matrix of a Simplex Centroid Design.
        
        Args:
            n_factors (int): number of factors

        Returns:
            matrix (Numpy Matrix): matrix of the design
        """
        return build_simplex_centroid(n_factors).values

    # static method to create a matrix design
    @staticmethod
    def simplex_lattice_design(n_factors, model_type = "linear"):
        """Calculate the Matrix of a Simplex Lattice Design.
        
        Args:
            n_factors (int): number of factors
            model_type (string): type of the model ('linear' or 'quadratic' or 'cubic')

        Returns:
            matrix (Numpy Matrix): matrix of the design
        """
        # check model type
        if model_type == "linear":
            model_order = ModelOrder.linear
        elif model_type == "quadratic":
            model_order = ModelOrder.quadratic
        elif model_type == "cubic":
            model_order = ModelOrder.cubic
        else:
            raise Exception("Invalid model type.")
        return build_simplex_lattice(n_factors, model_order).values

# artificial neural network main class
class ANN_Design:

    """
    Artifical Neural Network for Regression.

    ANN class accepts a dataset and then creates diverses models with all possible combinations
    of hyperparmeters passed. The ANN saves all models and graphics.
        
    Args:
        name (string): name of the model
        topoloyy (list of tuple of int): list of tuples of the range os neuros to be used in each hidden layer
        n_input_layer (int): number or neurons in the input layer
        n_output_layer (int): number of neurons in the output layer
        activation_function (list of strings): list of activation functions for each hidden layer, possible values:
        optimizers (list of strings): list of optimizer to use
        learning_rates (list of floats): list of learning rates to use (each value must be between 0 and 1)
        epochs (list of int): list of epochs to be used in each training
        metrics (list of strings): list of each metric to save history
        loss_function (string): loss function to be used
        k_crossfold (int): value of k to be used in the crossfold validation
        training_slice (float): slice of the dataset to be used in training (between 0 and 1) (ignored if k_crossfold > 0)

    """
    def __init__(self, data, outputs, n_input_layer, n_output_layer, hidden_layers, activation_functions, optimizers, learning_rates, epochs, metrics, loss_function, k_crossfold = False, training_slice = 0.5, name = "model", max_models = 10):
        self.name = name
        self.n_input_layer = n_input_layer
        self.n_output_layer = n_output_layer
        self.hidden_layers = hidden_layers
        self.activation_functions = activation_functions
        self.optimizers = optimizers
        self.learning_rates = learning_rates
        self.epochs = epochs
        self.metrics = metrics
        self.loss_function = loss_function
        self.k_crossfold = k_crossfold
        self.training_slice = training_slice
        self.max_models = max_models
        
        self.data = data
        self.inputs = list(data)
        self.inputs.remove(outputs)
        self.outputs = outputs
        self.models = []

        # log atributes
        # self.__log()

    # function to log attributes
    def __log(self):
        logger.debug("==Model Atributes==")
        log(logger.debug, "Input Layer", self.n_input_layer)
        log(logger.debug, "Output Layer", self.n_output_layer)
        log(logger.debug, "Hidden Layers", self.hidden_layers)
        log(logger.debug, "Activation Functions", self.activation_functions)
        log(logger.debug, "Optimizers", self.optimizers)
        log(logger.debug, "Learning Rates", self.learning_rates)
        log(logger.debug, "Epochs", self.epochs)
        log(logger.debug, "Metrics", self.metrics)
        log(logger.debug, "Loss Function", self.loss_function)
        log(logger.debug, "K Crossfold", self.k_crossfold)
        log(logger.debug, "Training Slice", self.training_slice)
        log(logger.debug, "Input", self.input_values, pre_message = "\n")
        log(logger.debug, "Output", self.output_values, pre_message = "\n")

    # method to start the training
    def train(self):
        # get data as raw matrix
        input_values_matrix = self.input_values
        input_values_matrix = input_values_matrix.values
        output_values_matrix = self.output_values
        output_values_matrix = output_values_matrix.values


        # transform hidden layers range as list of all combinations of hidden layers
        hidden_layers = list(self.hidden_layers)
        # exapand hidden layers range
        for i in range(len(hidden_layers)):
            hidden_layers[i] = list( range( hidden_layers[i][0], hidden_layers[i][1]+1) )
        # get all possible combinations
        hidden_layers = list(itertools.product(*hidden_layers))

        # transform activation function as combinations of possible activation function
        activation_functions = list(itertools.product(self.activation_functions, repeat=len(self.hidden_layers)))
        activation_functions = list([ values + ('linear',) for values in activation_functions])


        # check if should be applied k fold
        if self.k_crossfold < 0 or self.k_crossfold == False or self.k_crossfold is None:
            # slice data in the crossfold style
            indexs = list(range(input_values_matrix.shape[0]))
            i = round( input_values_matrix.shape[0] * self.training_slice)
            train_index = indexs[:i]
            test_index = indexs[i:]
            # create list
            index_split = [ ( np.array( train_index ), np.array( test_index) ) , ]
        else:
            index_split = list( KFold(n_splits = self.k_crossfold).split(input_values_matrix) )

        # relative path
        relative_path = './results'
        # for all possible values of each hyperparameter
        for hidden_layer in hidden_layers:
            # add hidden layer path
            hidden_layer_dir = list([str(n) for n in hidden_layer])
            hidden_layer_dir = '-'.join(hidden_layer_dir)
            hidden_layer_dir = relative_path + "/" + hidden_layer_dir

            for learning_rate in self.learning_rates:
                # add learning rate path
                learning_rate_dir = str(learning_rate)
                learning_rate_dir = hidden_layer_dir + "/" + learning_rate_dir
                
                for activation_function in activation_functions:
                    # add activation function path
                    activation_function_dir = '-'.join(activation_function)
                    activation_function_dir = learning_rate_dir + "/" + activation_function_dir

                    for optimizer in self.optimizers:
                        # add optimizer path
                        optimizer_dir = str(optimizer)
                        optimizer_dir = activation_function_dir + "/" + optimizer_dir

                        for epoch in self.epochs:
                            # add epoch path
                            epoch_dir = str(epoch)
                            epoch_dir = optimizer_dir + "/" + epoch_dir

                            # train that specific model
                            full_path = epoch_dir
                            name = "[" + full_path.replace('./results/', '').replace("/", "]-[") + "]"
                            # logging trining
                            log(logger.info, "Training", name)
                            self.train_model(input_values_matrix, output_values_matrix, index_split, hidden_layer, activation_function, optimizer, learning_rate, epoch, full_path)
                        
                        # end of epoch loop
                    # end of optimizer loop
                # end of activation function loop
            # end of learning rate loop
        # end of hidden layers loop
        
        logger.info("All Models Trained")
                
        # save models
        self._save_models()

    # method to train a specific model
    def train_model(self, input_values_matrix, output_values_matrix, index_split, hidden_layer, activation_functions, optimizer, learning_rate, epoch, relative_path):
        # for each index
        for train_index, test_index in index_split:
            # get input and output splitted
            input_train, input_test, output_train, output_test = input_values_matrix[train_index], input_values_matrix[test_index], output_values_matrix[train_index], output_values_matrix[test_index]
            
            # name of the model
            model_name = self.name
            
            # create model
            model = ANN_Model(self.n_input_layer, self.n_output_layer, hidden_layer, activation_functions, optimizer, learning_rate, self.metrics, self.loss_function)
            # train model
            history = model.train(input_train, output_train, epochs = epoch)
            # evaluate model
            result = model.evaluate(input_test,output_test)

            # add model 
            self.models.append( ( model, result, relative_path, model_name , history.history) )
            # sort based on the loss function
            self.models.sort(key = lambda x : x[1][0])
            # check max models
            if self.max_models != -1 : self.models = self.models[:self.max_models]

    # method to plot a history
    def _plot_history(self, history, path, name, keys):
        # create subplots
        fig, axes = plt.subplots(len(keys), 1)
        index = 0
        # dict of limits
        limits = {'loss': 200,'mean_squared_error': 200, 'mean_absolute_error': 200, 'mean_absolute_percentage_error': 200, 'mean_squared_logarithmic_error': 200, 'logcosh': 200, 'r_square': 1}
        # for each key
        for key in keys:
            axes[index].plot(history[key], color="green")
            axes[index].set_xlabel("Epochs")
            axes[index].set_ylabel( str(key) )
            axes[index].set_ylim( bottom = 0, top = limits[key] )
            index += 1

        try:    
            os.makedirs(path, exist_ok = True)
        except Exception:
            logger.error("Could not create or open directory: " + path)
        else:
            plt.savefig('./' + path + '/' + name + '.png')
        # finally:
        #     plt.show()

    # method to save models
    def _save_models(self):
        # for each model
        for i in range(len(self.models)):
            # get model
            model = self.models[i]
            # get keys
            keys = list(model[4].keys())
            
            # model info
            model_info = "Directory: " + str(model[2]) + "/"
            for j in range(len(keys)):
                model_info += "\n" + str(keys[j]) + ": " + str(model[1][j])

            # log model
            log(logger.info, "Best Model " + str(i), model_info, pre_message = "\n")

            # plot its history
            self._plot_history(model[4], model[2], model[3], keys)

    
    # set as properties
    @property
    def input_values(self): return self.data[self.inputs]

    @property
    def output_values(self): return self.data[self.outputs]

    @property
    def n_input_layer(self): return self._n_input_layer

    @n_input_layer.setter
    def n_input_layer(self, n): 
        # check if n is higher then 0
        if not ( isinstance(n, int) and n > 0) : raise Exception("Number of neurons in the input layer must be higher then 0")
        # set the property finaly
        self._n_input_layer = n

    @property
    def n_output_layer(self): return self._n_output_layer

    @n_output_layer.setter
    def n_output_layer(self, n): 
        # check if n is higher then 0
        if not ( isinstance(n, int) and n > 0) : raise Exception("Number of neurons in the output layer must be higher then 0")
        # set the property finaly
        self._n_output_layer = n

    @property
    def hidden_layers(self): return self._hidden_layers

    @hidden_layers.setter
    def hidden_layers(self, t): 
        # check if value passed is either a list or tuple
        if not (isinstance(t, list) or isinstance(t, tuple)) : raise Exception("Hidden Layers must be a list or tuple")
        # check if values within t are all either list or tuple
        if not all(( isinstance(n, list) or isinstance(n, tuple)) for n in t): raise Exception("Values of Hidden Layers must be a list or tuple")
        # check if values within t are all either list or tuple of int min and max
        if not all( ( len(n) == 2 and n[1] >= n[0] ) for n in t): raise Exception("Values of Hidden Layers must have the format: [min, max]")
        # set the property finaly
        self._hidden_layers = t

    @property
    def activation_functions(self): return self._activation_functions

    @activation_functions.setter
    def activation_functions(self, l): 
        # check if value passed is either a list or tuple
        if not (isinstance(l, list) or isinstance(l, tuple)) : raise Exception("Activation Functions must be a list or tuple")
        # check if values within t are all string
        if not all(isinstance(n, str) for n in l): raise Exception("Activation Functions must be string")
        # valids activation functions
        valids = ['tanh', 'sigmoid', 'linear', 'exponential', 'relu', 'elu']
        # check if values are valid
        if not all(  ( n in valids ) for n in l): raise Exception("Invalid Activation Function")
        # set the property finaly
        self._activation_functions = l

    @property
    def optimizers(self): return self._optimizers

    @optimizers.setter
    def optimizers(self, l): 
        # check if value passed is either a list or tuple
        if not (isinstance(l, list) or isinstance(l, tuple)) : raise Exception("Optimizers must be a list or tuple")
        # check if values within t are all string
        if not all(isinstance(n, str) for n in l): raise Exception("Optimizers must be string")
        # valids activation functions
        valids = ['sgd', 'adam', 'nadam', 'RMSprop']
        # check if values are valid
        if not all(  ( n in valids ) for n in l): raise Exception("Invalid Optimizer")
        # set the property finaly
        self._optimizers = l

    @property
    def learning_rates(self): return self._learning_rates

    @learning_rates.setter
    def learning_rates(self, l): 
        # check if value passed is either a list or tuple
        if not (isinstance(l, list) or isinstance(l, tuple)) : raise Exception("Learning Rates must be a list or tuple")
        # check if is a float and if it is between 0 and 1
        if not all( (isinstance(n, float) and 0 < n < 1) for n in l) : raise Exception("Learning Rates values must be a float between 0 and 1")
        # set the property finaly
        self._learning_rates = l

    @property
    def epochs(self): return self._epochs

    @epochs.setter
    def epochs(self, l): 
        # check if value passed is either a list or tuple
        if not (isinstance(l, list) or isinstance(l, tuple)) : raise Exception("Epochs must be a list or tuple")
        # check if values are int and higher than 0
        if not all( ( isinstance(n, int) and n > 0) for n in l): raise Exception("Values of epoch must be a int higher than 0")
        # set the property finaly
        self._epochs = l

    @property
    def metrics(self): return self._metrics

    @metrics.setter
    def metrics(self, l): 
        # check if value passed is either a list or tuple
        if not (isinstance(l, list) or isinstance(l, tuple)) : raise Exception("Metrics must be a list or tuple")
        # check if values within t are all string
        if not all(isinstance(n, str) for n in l): raise Exception("Metrics must be string")
        # valids metrics
        valids = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'logcosh', 'r_square']
        # check if values are valid
        if not all(  ( n in valids ) for n in l): raise Exception("Invalid Metric")
        # filter soft acc
        for i in range(len(l)):
            if l[i] == 'r_square':
                l[i] = r_square
        # set the property finaly
        self._metrics = l

    @property
    def loss_function(self): return self._loss_function

    @loss_function.setter
    def loss_function(self, n): 
        # check if values within t are all string
        if not isinstance(n, str) : raise Exception("Loss Function must be string")
        # valids loss functions
        valids = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'logcosh']
        # check if values are valid
        if n not in valids: raise Exception("Invalid Metric")
        # set the property finaly
        self._loss_function = n

    @property
    def k_crossfold(self): return self._k_crossfold

    @k_crossfold.setter
    def k_crossfold(self, n): 
        # check if values within t are all string
        if not (isinstance(n, int) or n == False or n is None) : raise Exception("Loss Function must be string")
        # set the property finaly
        self._k_crossfold = n

    @property
    def training_slice(self): return self._training_slice

    @training_slice.setter
    def training_slice(self, n): 
        # check if values within t are all string
        if not (isinstance(n, float) and 0 < n <= 1) : raise Exception("Training Slice must be a float between 0 and 1")
        # set the property finaly
        self._training_slice = n

    @property
    def max_models(self): return self._max_models

    @max_models.setter
    def max_models(self, n): 
        # check if values within t are all string
        if not (isinstance(n, int) and n >= -1) : raise Exception("Max number of models must be an integer higher or equal to -1")
        # set the property finaly
        self._max_models = n
    
# Artificial Neural Network Model
class ANN_Model:

    """
    Artifical Neural Network for Regression.

    ANN class accepts a dataset and then creates diverses models with all possible combinations
    of hyperparmeters passed. The ANN saves all models and graphics.
        
    Args:
        name (string): name of the model
        topoloyy (list of tuple of int): list of tuples of the range os neuros to be used in each hidden layer
        n_input_layer (int): number or neurons in the input layer
        n_output_layer (int): number of neurons in the output layer
        activation_function (list of strings): list of activation functions for each hidden layer, possible values:
        optimizers (list of strings): list of optimizer to use
        learning_rates (list of floats): list of learning rates to use (each value must be between 0 and 1)
        epochs (list of int): list of epochs to be used in each training
        metrics (list of strings): list of each metric to save history
        loss_function (string): loss function to be used
        k_crossfold (int): value of k to be used in the crossfold validation
        training_slice (float): slice of the dataset to be used in training (between 0 and 1) (ignored if k_crossfold > 0)

    """
    def __init__(self, n_input_layer, n_output_layer, n_hidden_layers, activation_functions, optimizer, learning_rate, metrics, loss_function):
        self.n_input_layer = n_input_layer
        self.n_output_layer = n_output_layer
        self.n_hidden_layers = n_hidden_layers
        self.activation_functions = activation_functions
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.metrics = metrics
        self.loss_function = loss_function
        # compile model
        self.model = self.build_model()

    # method to build a model
    def build_model(self):
        # create model
        model = Sequential()
        # add first layer
        model.add( Dense( self.n_hidden_layers[0], input_shape = ( self.n_input_layer, ), activation = self.activation_functions[0] ) )
        # for every other layer
        for i in range(1, len(self.n_hidden_layers)):
            # add layers
            model.add( Dense( self.n_hidden_layers[i], activation = self.activation_functions[i] ) )
        # add last layer
        model.add( Dense( self.n_output_layer, activation = self.activation_functions[-1] ) )
        # check optimizers
        optimizers = {'sgd': SGD, 'adam': Adam, 'nadam': Nadam, 'RMSprop': RMSprop}
        # compile it
        model.compile(loss = self.loss_function, optimizer = optimizers[self.optimizer](learning_rate=self.learning_rate), metrics=self.metrics )
        # return model
        return model

    # method to start the training
    def train(self, x=None, y=None, batch_size=None, epochs=1, callbacks=None):
        self.history = self.model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
        return self.history

    # method to evaluate the model
    def evaluate(self, x=None, y=None, batch_size=None, callbacks=None):
        return self.model.evaluate(x=x, y=y, batch_size=batch_size, callbacks=callbacks)

    # method to predict values
    def predict(self, x, batch_size=None, callbacks=None):
        return self.model.predict(x=x, batch_size=batch_size, callbacks=callbacks)

    # method to predict values
    def predict_classes(self, x, batch_size=None, callbacks=None):
        return self.model.predict_classes(x=x)

    @property
    def n_input_layer(self): return self._n_input_layer

    @n_input_layer.setter
    def n_input_layer(self, n): 
        # check if n is higher then 0
        if not ( isinstance(n, int) and n > 0) : raise Exception("Number of neurons in the input layer must be higher then 0")
        # set the property finaly
        self._n_input_layer = n

    @property
    def n_output_layer(self): return self._n_output_layer

    @n_output_layer.setter
    def n_output_layer(self, n): 
        # check if n is higher then 0
        if not ( isinstance(n, int) and n > 0) : raise Exception("Number of neurons in the output layer must be higher then 0")
        # set the property finaly
        self._n_output_layer = n

    @property
    def n_hidden_layers(self): return self._n_hidden_layers

    @n_hidden_layers.setter
    def n_hidden_layers(self, t): 
        # check if value passed is either a list or tuple
        if not (isinstance(t, list) or isinstance(t, tuple)) : raise Exception("Hidden Layers must be a list or tuple")
        # check if values within t are all either list or tuple
        if not all( ( isinstance(n, int) and n > 0 ) for n in t): raise Exception("Values of Hidden Layers must be a int higher than 0")
        # set the property finaly
        self._n_hidden_layers = t

    @property
    def activation_functions(self): return self._activation_functions

    @activation_functions.setter
    def activation_functions(self, l): 
        # check if value passed is either a list or tuple
        if not (isinstance(l, list) or isinstance(l, tuple)) : raise Exception("Activation Functions must be a list or tuple")
        # check if values within t are all string
        if not all(isinstance(n, str) for n in l): raise Exception("Activation Functions must be string")
        # valids activation functions
        valids = ['tanh', 'sigmoid', 'linear', 'exponential', 'relu', 'elu', 'softmax']
        # check if values are valid
        if not all(  ( n in valids ) for n in l): raise Exception("Invalid Activation Function")
        # check number of activation function
        if len( self.n_hidden_layers ) + 1 != len(l) : raise Exception("Invalid Number of Activation Functions")
        # set the property finaly
        self._activation_functions = l

    @property
    def optimizer(self): return self._optimizer

    @optimizer.setter
    def optimizer(self, l): 
        # check if values within t are all string
        if not isinstance(l, str): raise Exception("Optimizer must be string")
        # valids activation functions
        valids = ['sgd', 'adam', 'nadam', 'RMSprop']
        # check if values are valid
        if not ( l in valids ): raise Exception("Invalid Optimizer")
        # set the property finaly
        self._optimizer = l

    @property
    def learning_rate(self): return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, l): 
        # check if is a float and if it is between 0 and 1
        if not (isinstance(l, float) and 0 < l < 1) : raise Exception("Learning Rate value must be a float between 0 and 1")
        # set the property finaly
        self._learning_rate = l

    @property
    def metrics(self): return self._metrics

    @metrics.setter
    def metrics(self, l): 
        # check if value passed is either a list or tuple
        if not (isinstance(l, list) or isinstance(l, tuple)) : raise Exception("Metrics must be a list or tuple")
        # check if values within t are all string
        if not all(isinstance(n, str) for n in l): raise Exception("Metrics must be string")
        # valids metrics
        valids = ['mean_squared_error', 'mean_absolute_error', 'accuracy','mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'logcosh', 'r_square']
        # check if values are valid
        if not all(  ( n in valids ) for n in l): raise Exception("Invalid Metric")
        # filter soft acc
        l = list( [ r_square if x == 'r_square' else x for x in l ] )
        # set the property finaly
        self._metrics = l

    @property
    def loss_function(self): return self._loss_function

    @loss_function.setter
    def loss_function(self, n): 
        # check if values within t are all string
        if not isinstance(n, str) : raise Exception("Loss Function must be string")
        # valids loss functions
        valids = ['mean_squared_error', 'categorical_crossentropy', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'logcosh']
        # check if values are valid
        if n not in valids: raise Exception("Invalid Metric")
        # set the property finaly
        self._loss_function = n



# one way anova
def one_way_anova(data, y_values, groups):
    # get only wanted data
    data = data[ [y_values, groups] ]
    # data grouped
    data_grouped = data.groupby(groups)
    # get mean of every group
    group_mean = data_grouped.mean()
    # group count
    group_count = data_grouped.count()
    # total mean
    mean = data[y_values].mean()
    # total of samples
    total_samples = data.shape[0]
    # total groups
    total_groups = len(group_count)

    # total values
    total_ss = np.sum(np.square( data[y_values] - mean))
    total_df = total_samples - 1

    # treatments values
    treatments_ss = np.sum(group_count*np.square(group_mean - mean)) [y_values]
    treatments_df = total_groups - 1
    treatments_ms =  treatments_ss/treatments_df

    # residuals values
    residuals_ss = np.sum( data_grouped.transform(lambda x: np.square(x - x.mean() ) ) ) [y_values]
    residuals_df = total_samples - total_groups
    residuals_ms =  residuals_ss/residuals_df

    # F value
    F = treatments_ms/residuals_ms

    # p value
    p = 1 - sts.f.cdf(F, treatments_df,residuals_df )

    # convert as matrix
    matrix = [[treatments_ss , treatments_df, treatments_ms,    F  ,    p  ], 
              [ residuals_ss , residuals_df , residuals_ms , np.nan, np.nan], 
              [   total_ss   ,    total_df  ,    np.nan    , np.nan, np.nan]]

    # create dataframe
    dataframe = pd.DataFrame(matrix, index= ["Entre Grupos", "Dentro dos Grupos", "Total"], columns=["Soma Quadrática", "Graus de Liberdade", "Média Quadrática", "F", "p"])
    # return dataframe
    return dataframe
            

def main(mode, data, degree, model):
    use_log = True
    try:
        data = pd.read_excel("./data/" + data)
        #data = pd.read_csv("./data/" + data, sep = ",")
    except:
        logger.critical("Invalid Data")
    else:
        if mode == "regression":
            doe = Design(data, "Rendimento", degree = degree, use_log=use_log)
        elif mode == "factorial":
            doe = Factorial_Design(data = data, dependent_value = "Rendimento", use_log = use_log)
        elif mode == "central_composite":
            doe = Central_Composite_Design(use_pure_error = False, data = data, dependent_value = "Rendimento", use_log = use_log)
        elif mode == "mixture":
            doe = Mixture_Design(use_pure_error = False, data = data, dependent_value = "Número de cetanos", use_log = use_log, degree = degree)
        elif mode == "read":
            print("Head:")
            print(data.head())
            print("Describe:")
            print(data.describe())
        elif mode == "ann":
            # read json
            with open("./model/" + model) as json_file:
                model_parameters = json.load(json_file)
            # create ann design
            # log(logger.debug, "Mode", mode)
            logger.info("Creating ANN Model")
            # model = ANN_Design(data, "Número de cetanos", model_parameters["n_input_layer"], model_parameters["n_output_layer"], model_parameters["hidden_layers"], model_parameters["activation_functions"], model_parameters["optimizers"], model_parameters["learning_rates"], model_parameters["epochs"], model_parameters["metrics"], model_parameters["loss_function"], model_parameters["k_crossfold"], model_parameters["training_slice"], model_parameters["name"], model_parameters["max_models"])
            model = ANN_Model(model_parameters["n_input_layer"], model_parameters["n_output_layer"], model_parameters["hidden_layers"], model_parameters["activation_functions"], model_parameters["optimizer"], model_parameters["learning_rate"], model_parameters["metrics"], model_parameters["loss_function"])
            # get response values
            data_y = data["Número de cetanos"]
            # get input values
            x_labels = list(data)
            x_labels.remove("Número de cetanos")
            # data x
            data_x = data[x_labels]
            # train model
            model.train(data_x, data_y, epochs=model_parameters['epochs'])
        elif mode == "anova":
            # call anova function
            anova = one_way_anova(data, "Time", "Company")
            logger.debug("One-Way ANOVA:\n" + str(anova) )
        else:
            logger.error("Invalid Mode")
        logger.info("Analisys Done")
    finally:
        logger.info("Exiting...")


if __name__ == "__main__":
    #parameters
    data = None
    model = None
    degree = 1
    mode = None
    # for every arg
    for arg in sys.argv:
        if "--model=" in arg:
            # get without parameters
            model = arg.replace("--model=", '', 1)
        elif "--data=" in arg:
            # get without parameters
            data = arg.replace("--data=", '', 1)
        elif "--degree=" in arg:
            # get without parameters
            degree = arg.replace("--degree=", '', 1)
        elif "--mode=" in arg:
            # get without parameters
            mode = arg.replace("--mode=", '', 1)

    # call main function
    main(mode, data, degree, model)     

