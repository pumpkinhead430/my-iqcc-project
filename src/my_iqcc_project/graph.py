import pandas as pd
import numpy as np
from scipy import stats
from scipy.odr import Model, ODR, RealData
from multipledispatch import dispatch
import matplotlib.pyplot as plt

class Graph:
    def __init__(self,point_set,delta_point_set, method ,initial_guess , title='', label='', x_label='', y_label='', x_unit='', y_unit='', log=False):
        self.x_unit = x_unit
        self.y_unit = y_unit
        self.method = method
        self.title = title
        self.label = label
        self.x_label = x_label
        self.y_label = y_label
        self.log = log
        self.extra_graphs = []
        self._only_draw = False
        
        x,y = zip(*point_set)
        self.x,self.y = np.array(x,dtype=float),np.array(y,dtype=float)
        if delta_point_set:
            delta_x,delta_y = zip(*delta_point_set)
            self.delta_x,self.delta_y = np.array(delta_x,dtype=float),np.array(delta_y,dtype=float)
        else:
            self._only_draw = True
        if method:
            self.fit_params,self.params_error,self.fit  = self.generate_fit(point_set, delta_point_set,initial_guess)
            self.chi_square =self.fit.sum_square
            self.chi_square_red = self.fit.sum_square/(len(point_set)-len(initial_guess))
            self.p_value = 1 - stats.chi2.cdf(self.chi_square, len(point_set)-len(initial_guess))
            self.residuals = self.y - self.method(self.fit_params, self.x)
    
    def generate_fit(self, point_set, delta_point_set,initial_guess):
        model = Model(self.method)
        x,y = zip(*point_set)
        x,y = np.array(x,dtype=float),np.array(y,dtype=float)
        delta_x, delta_y = zip(*delta_point_set)
        delta_x, delta_y = np.array(delta_x,dtype=float),np.array(delta_y,dtype=float)

        data = RealData(x, y, sx=delta_x, sy=delta_y)
        odr = ODR(data, model, beta0=initial_guess)
        output = odr.run()
        return output.beta, output.sd_beta, output
    
    def display_fit(self):
        if self.method:
            i =0 
            for param in zip(self.fit_params, self.params_error):
                print(f"a[{i}] =  {param[0]} \u00B1 {param[1]} ({abs(param[1]/param[0]*100)}%)")
                i +=1
            print(f"chi_square_red = {self.chi_square_red}")
            print(f"p_value = {self.p_value}")
        else:
            print('no Fit')
    
    def display_fit_short(self):
        if self.method:
            i =0 
            for param in zip(self.fit_params, self.params_error):
                error_percent = abs(round_significant(param[1]/param[0]*100))
                round_error = round_significant(param[1])
                if round_error == 0:
                    round_error = 1e-2
                
                round_value = round_significant(param[0], round_error)
                print(f"a_{i} =  {round_value} \\pm {round_error} ({error_percent} \\%)")
                i +=1
            print(f"chi_square_red = {round_significant(self.chi_square_red)}")
            print(f"p_value = {round_significant(self.p_value)}")
        else:
            print('no Fit')

    def plot_fit(self):
        plt.style.use('classic')
        plt.figure(facecolor='w')
        plt.grid(True)
        if self.log:
            plt.xscale('log')
            plt.yscale('log')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        title = self.title
         
        self._draw_fit()
        for graph in self.extra_graphs:
            graph._draw_fit()

        if self.method:
            title = title  + "- Fit"

        plt.title(title)
        plt.xlabel(self.x_label + self.x_unit)
        plt.ylabel(self.y_label + self.y_unit)

    def _draw_fit(self):
        if self._only_draw:
            plt.plot(self.x, self.y,linestyle='-', label=self.label)
        else:
            plt.errorbar(self.x, self.y, xerr=self.delta_x, yerr=self.delta_y, marker='o',linestyle='', label=self.label)
        fit_x = np.linspace(min(self.x), max(self.x), 10* len(self.x))
        if self.method:
            plt.plot(fit_x, self.method(self.fit_params, fit_x), '-',label = self.label + " Fit") 

    
    def plot_residuals(self):
        if self.method:
            plt.style.use('classic')
            plt.figure(facecolor='w')
            plt.grid(True)

            self._draw_residuals()
            for  graph in self.extra_graphs:
                graph._draw_residuals()

            plt.axhline(0, color='red', linestyle='--')
            plt.title(self.title + "- Residuals")
            plt.xlabel(self.x_label + self.x_unit)
            plt.ylabel(f"F({self.x_label}) - " + self.y_label + self.y_unit)
            plt.legend()
            plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            plt.show()
        else:
            print("No Fit")

    def _draw_residuals(self):
        plt.errorbar(self.x, self.residuals, xerr=self.delta_x, yerr=self.delta_y, marker='o',linestyle='', label=self.label)
    
    def get_fit_params(self):
        if self.method:
            return list(zip(self.fit_params, self.params_error))
        return None
    
    def __add__(self, other):
        self.extra_graphs.append(other)
        return self

@dispatch(object, object)
def round_significant(value, referance):
    if referance == 0:
        return value
    else:
        return round(value, - int(np.floor(np.log10(abs(referance)))-1))
@dispatch(object)
def round_significant(value):
    if value == 0:
        return value
    else:
        expo = -int(np.floor(np.log10(abs(value))))+1
        return round(value,expo)

        