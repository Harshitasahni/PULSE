from distutils.version import LooseVersion
import json
import math
import numpy as np
import scipy.stats
from PIL import Image as im
from os.path import exists
import matplotlib.pyplot as plt
import trajwrapper as TW
#import EncodingGenerator_new
from matplotlib.pyplot import figure
from sklearn.decomposition import NMF
import matplotlib.patches as mpatches
from itertools import permutations
from collections import defaultdict
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import copy



def rbf_kernel(x, y, gamma=1.0):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    dists = cdist(x, y, 'sqeuclidean')
    return np.exp(-gamma * dists)

def compute_mmd(x, y, gamma=1.0):
    gamma = 1 / (2 * np.median(pdist(np.vstack([x, y])) ** 2) + 1e-8)
    K_xx = rbf_kernel(x, x, gamma)
    K_yy = rbf_kernel(y, y, gamma)
    K_xy = rbf_kernel(x, y, gamma)

    n = len(x)
    m = len(y)

    mmd = (np.sum(K_xx) - np.trace(K_xx)) / (n * (n - 1)) + \
          (np.sum(K_yy) - np.trace(K_yy)) / (m * (m - 1)) - \
          2 * np.sum(K_xy) / (n * m)

    return mmd





def ftest(s1,n1,s2,n2,alpha):
    alpha /= 2
    # Test statistics, on upper critical value
    if s1 > s2:
        F_score = s1**2 / s2**2
        critical_value = stats.f.ppf(1-alpha, n1-1, n2-1)
    else:
        F_score = s2**2 / s1**2
        critical_value = stats.f.ppf(1-alpha, n2-1, n1-1)

    if F_score < critical_value:
        return True
    else:
        return False


# def eval_window_4all(window_of_frames,models,States, alpha, newTraj, my_states, verbose=0):

#     loss = {}
#     absE = {}
#     errR = {}
#     prob = 0
#     change = True
#     flag = False
#     last_err = False
#     explain_temp = [1000] * len(States)
#     maxLastErr = [0] * len(States)
#     prob_temp = []
#     explain = [] # whether the traj is unstable at this point (the frame cannot
#     #be explained by the current model, but can from a previous one)
#     #last = order[0]
#     flag = True

#     mode = 'mmd'

#     if mode == 'obg':
#         treshold = 0.01
#     elif mode == 'rbf':
#         treshold = 0.99
#     elif mode =='mmd':
#         treshold = 0.99
#     elif mode == 'histogram':
#         treshold = 0.6


#     # collect losses for all the models of the last state
#     for nm in range(States[-1].nmodels):

#     #====================================
#     # compute the loss for every model
#         loss[nm],errR[nm],absE[nm] = eval_reconstruction_loss(models[nm], window_of_frames)
#         if nm == 0:
#             err_perR = np.zeros_like(absE[0])
#             abs_perR = np.zeros_like(absE[0])
#         if nm==0 or max(errR[nm]) < max(err_perR_l):
#             err_perR_l = errR[nm]
#             abs_perR_l = absE[nm]


#     #====================================
#     fmax = np.argmax(loss[States[-1].nmodels-1])
#     # test all states in order


#     #for i in order:
#     for i,state in enumerate(States):
#         #state = States[i]

#         prob,avgloss = state.evaluate_state(loss,mode,verbose)
#         #print("prob {}  err {}".format(prob,max(absE[i])))

#         avgerr = sum(absE[i])/len(absE[i])
#         maxLastErr[i] = max(absE[i])

#         #if (not newTraj and avgloss < 1.0 and max(absE[i]) < 2.5) or (not newTraj and max(absE[i]) < 0.7) or (prob>0 and max(absE[i]) < 2.5): # or( prob > state.nmodels/2 and avgerr < 1 and max(absE[i]) < 2): #  2  or max(absE[i]) <= 0.11:  # or max(absE[i]) < 0.2:
#         #if prob > 0.99 and avgerr <= 1.0 and max(absE[i]) < 2.5:
#         if prob > treshold and avgerr <= 1.0 and max(absE[i]) < 2.5:

#             change = False

#             score = ((avgerr + (0.1 * max(absE[i])))/prob)  #0.05

#             if i in my_states:
#                 explain_temp[i] = 0.8 * score
#             else:
#                 explain_temp[i] = score

#         #     if verbose:
#         #         print("Explain {}  prob {}  avgerr {}  maxerr {}, explain_temp {} avgloss {}".format(i, prob, avgerr, max(absE[i]),explain_temp[i],avgloss))


#         # elif verbose:
#         #     print("Not explained {}  prob {}  avgerr {}  maxerr {} avgloss {}".format(i, prob, avgerr,max(absE[i]),avgloss))


#     min_error = min(explain_temp)
#     if min_error < 1000:
#         j = explain_temp.index(min_error)
#         explain.append(j)
#         err_perR=errR[j]
#         abs_perR=absE[j]


#     return(loss, err_perR_l, abs_perR_l, fmax, maxLastErr, change, explain, False)

def ftest(s1,n1,s2,n2,alpha):
   alpha /= 2
   # Test statistics, on upper critical value
   if s1 > s2:
       F_score = s1**2 / s2**2
       critical_value = stats.f.ppf(1-alpha, n1-1, n2-1)
   else:
       F_score = s2**2 / s1**2
       critical_value = stats.f.ppf(1-alpha, n2-1, n1-1)


   if F_score < critical_value:
       return True
   else:
       return False



class State:

#     generator =
    def __init__(self,  id1, start, nmodels, losses, wsize, alpha=0.8):
        self.id = id1
        self.nmodels = nmodels  # number of models
        self.nx = 0        # number of data points in this state
        self.sumx = [0]*nmodels    # cumulative sum
        self.sumxq = [0]*nmodels  # cumulative square sum
        self.mean = [0]*nmodels  # current mean
        self.stdev = [0]*nmodels    # current standard deviation
        self.var = [0]*nmodels    # current variance
        self.ub = [0]*nmodels    # upper bound of CI
        self.lb = [0]*nmodels    # lower bound of CI
        self.dist = {}  # normal distribution per model
        self.start = start  # frame where this state starts
        self.end = start
        self.alpha = alpha  # confidence interval parameter
        self.models = []
        self.losses = {}
        self.emb_losses = {}
        self.losses = copy.deepcopy(losses)

        self.add_window(losses, wsize, False)  # add the current window to the data
        self.calculate_dist() # compute current mean and stdev
        self.embed_losses(losses, 'histogram')

    def print_state(self):
        print("State {}, start {}, nmodels {}, nx {}".format(self.id, self.start, self.nmodels, self.nx))
        print("  sumx {} \n  sumxq {} \n  mean {} \n  var {} \n stdev {}".format(self.sumx,self.sumxq,self.mean,self.var,self.stdev))
        for m in range(self.nmodels):
            print("  CI model {} [{}, {}]".format(m+1, self.lb[m],self.ub[m]))


    def add_window(self, losses, wsize, add_losses=True):
        # assumes that losses is a dictionary of numpy arrays containing the losses
        # for each model within the current window
        nwindows = self.nx / wsize
        for m in range(self.nmodels):
            self.sumx[m] += sum(losses[m])
            self.sumxq[m] += sum(np.square(losses[m]))
            if add_losses:
                self.losses[m] = (((nwindows-1)/nwindows)*self.losses[m]) + ((1/nwindows)*losses[m])
        self.nx += wsize
        self.end += wsize
        self.calculate_dist()

    def calculate_dist(self):
    # calculates the normal distribution given the mean and stdev
    # calculates as well confidence intervals
        ci=(1-self.alpha)/2
        for m in range(self.nmodels):
            self.mean[m] = self.sumx[m]/self.nx # mean
            self.var[m] = (self.sumxq[m]/self.nx) - (self.mean[m]*self.mean[m]) # variance
            self.stdev[m] = np.sqrt(abs(self.var[m])) # standard deviation
            self.dist[m] = scipy.stats.norm(self.mean[m],self.stdev[m])
            self.ub[m]=self.dist[m].ppf(1-ci)
            self.lb[m]=self.dist[m].ppf(ci)


    def evaluate_state_last(self,loss,verbose):  # use confidence intervals and binary output
    # determines if the given state can explain the window loss
        prob = 0
        cv = 0.75 # .99
        alpha = 0.0000000001
        #alpha = 0.05
        alpha = 0.0000000000001
        m = self.nmodels-1
        #ks_res = stats.ks_1samp(loss[m], self.dist[m].cdf)
        p=np.mean(loss[m])
        std_dist = self.stdev[m]
        std_samp=np.std(loss[m],ddof=1)
        margin = stats.norm.ppf(cv)*std_samp/math.sqrt(len(loss[m]))
        # p +/- margin

        ftest_val = ftest(self.stdev[m],self.nx,std_samp,len(loss[m]),alpha)

        if (self.lb[m] <= p - margin and p + margin <= self.ub[m]) and ftest_val:
            perc = self.dist[m].cdf(p)
            prob += perc

        #verbose = 1
        # if verbose > 0:
        #     #print("m{} [{}, {}, {}] {}".format(self.nmodels, self.lb[m], p, self.ub[m],prob))
        #     print("m{} [err {}] prob {}".format(self.nmodels-1,  p, prob))

        return(prob,p)


    def evaluate_state(self, loss, mode, verbose):
        m = self.nmodels-1
        p=np.mean(loss[m])

        if mode == 'mmd':
            mmd_values = []
            threshold=0.05 #0.01 #0.025
            gamma=1.0
            mmd_c = 0

            for m in range(self.nmodels):
                mmd = compute_mmd(loss[m], self.losses[m], gamma=gamma)
                mmd_values.append(mmd)
                mmd_c += mmd
                # if verbose:
                #     print("mmd value ",mmd)
                if mmd > threshold:
                    return(0,p) #mmd_values)
            prob = 1 - (mmd_c/self.nmodels)
            return(prob,p) #mmd_values)

        elif mode == 'obg':
            prob = 0
            cv = 0.75 # .99
            alpha = 0.0000000001
            #alpha = 0.05
            alpha = 0.0000000000001
            std_dist = self.stdev[m]
            std_samp=np.std(loss[m],ddof=1)
            margin = stats.norm.ppf(cv)*std_samp/math.sqrt(len(loss[m]))
            # p +/- margin

            ftest_val = ftest(self.stdev[m],self.nx,std_samp,len(loss[m]),alpha)

            if (self.lb[m] <= p - margin and p + margin <= self.ub[m]) and ftest_val:
                perc = self.dist[m].cdf(p)
                prob += perc

            #verbose = 1
            # if verbose > 0:
            #     #print("m{} [{}, {}, {}] {}".format(self.nmodels, self.lb[m], p, self.ub[m],prob))
            #     print("m{} [err {}] prob {}".format(self.nmodels-1,  p, prob))
            return(prob,p)

        elif mode == 'rbf':

            con_values = []
            treshold=0.5 #0.01 #0.025
            sim_c = 0
            for m in range(self.nmodels):
                x = loss[m]
                x = x.reshape(-1, 1)
                gamma=self.mean[m] # 1.0
                hist_range = (self.mean[m]-(3*self.stdev[m]), self.mean[m]+(3*self.stdev[m]))
                emb_loss = self.embed_rbf(x,gamma,hist_range)
                simm = cosine_similarity(self.emb_losses[m].reshape(1, -1),emb_loss.reshape(1, -1))[0]
                sim_c += simm
                #if simm < treshold:
                #    return(0,p)
            prob = sim_c/self.nmodels
            return(prob,p)

        elif mode == 'histogram':

            con_values = []
            treshold=0.5 #0.01 #0.025
            sim_c = 0
            for m in range(self.nmodels):
                x = loss[m]
                x = x.reshape(-1, 1)
                hist_range = (self.mean[m]-(3*self.stdev[m]), self.mean[m]+(3*self.stdev[m]))
                emb_loss = self.embed_histogram(x,hist_range)
                simm = cosine_similarity(self.emb_losses[m].reshape(1, -1),emb_loss.reshape(1, -1))[0]
                sim_c += simm
                if simm < treshold:
                    return(0,p)
            prob = sim_c/self.nmodels
            return(prob,p)



    def embed_rbf(self,x,gamma,hist_range):
        nbins = 10

        centers = np.linspace(hist_range[0], hist_range[1], nbins).reshape(-1, 1)
        dists = (x - centers.T)**2  # shape: (n_points, nbins)
        kernel_vals = np.exp(-gamma * dists)
        kernel_vals = np.exp(-gamma * dists)
        return(np.mean(kernel_vals, axis=0))

    def embed_histogram(self, x, hist_range):
        nbins = 10
        # Histogram-based distributional embedding (normalized)
        hist, _ = np.histogram(x, bins=nbins, range=hist_range, density=False)
        hist = hist.astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()
        return hist


    def embed_losses(self, loss, mode='rbf'):

        # Compute Gaussian RBF kernel mean embedding against fixed centers
        if mode == 'rbf':
            for m in range(self.nmodels):
                x = loss[m]
                x = x.reshape(-1, 1)
                gamma=self.mean[m] # 1.0
                hist_range = (self.mean[m]-(3*self.stdev[m]), self.mean[m]+(3*self.stdev[m]))
                self.emb_losses[m] = self.embed_rbf(x,gamma,hist_range)
        if mode == 'histogram':
            for m in range(self.nmodels):
                x = loss[m]
                x = x.reshape(-1, 1)
                hist_range = (self.mean[m]-(3*self.stdev[m]), self.mean[m]+(3*self.stdev[m]))
                self.emb_losses[m] = self.embed_histogram(x,hist_range)
        else:
            for key in self.losses:
                self.emb_losses[key] = self.losses[key].copy()


