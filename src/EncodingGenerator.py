import mdtraj as md
import os.path
import os
import numpy as np
# import matplotlib.cm as cm
# import matplotlib.pyplot as plt
import scipy.spatial.distance as distance
from functools import partial
import numpy as np
from PIL import Image, ImageOps, ImageFont, ImageDraw
# import cv2
# import matplotlib
# import imageio
import numpy as np
# import matplotlib.image as mpimg


class EncodingGenerator():
    def __init__(self, traj_wrapper, transform_function,alpha_CA = True, select = ""):
       self.traj_wrapper = traj_wrapper
       self.transform_function = transform_function
       self.image_number = 0
       self.select = select
       self.alpha_CA = alpha_CA
       self.ss_indices = np.array([])
       self.xyz_indices = np.array([])


    def set_indices(self,topology):
       # encode by residues
       if self.alpha_CA:  # select only alpha CA
           self.xyz_indices = topology.select(self.select) #'name CA'
#             print(" self.xyz_indices ", self.xyz_indices )
           if self.select != "name CA":
           # select specific residues
               s_idx=[]
               for index in self.xyz_indices:
                   s_idx.append(topology.atom(index).residue.index)


               self.ss_indices = np.array(s_idx)
#                 print(" self.ss_indices ", self.ss_indices )
       #encode by atoms
       else:
           sec_s = [None] * topology.n_atoms
           index=0
           for atom in topology.atoms:
               sec_s[index] = atom.residue.index
               index+=1
           self.ss_indices = np.array(sec_s)


           if self.select != "":
           # select specific residues
               indices = topology.select(self.select)
               self.ss_indices = self.ss_indices[indices.astype(int)]
               self.xyz_indices = indices



    def __iter__(self):
       return self


    def frame(self,nframe):
       try:
           traj = self.traj_wrapper.load_frame(nframe)
           self.image_number = nframe


           topology = traj.topology
           self.set_indices(topology)
           #select coordinates
           if self.xyz_indices.size > 0:
               xyz = traj.xyz[0, self.xyz_indices]
           else:
               xyz = traj.xyz[0, :]


           # select secondary structure
           # secondary_structure = md.compute_dssp(traj, simplified=False)[0]
           # if self.ss_indices.size > 0:
           #     secondary_structure = secondary_structure[self.ss_indices]




           # the function produces an array of strings denoting secondary structure of each residue
           # possible secondary structures are:


           '''
           ‘H’ : Alpha helix
           ‘B’ : Residue in isolated beta-bridge
           ‘E’ : Extended strand, participates in beta ladder
           ‘G’ : 3-helix (3/10 helix)
           ‘I’ : 5 helix (pi helix)
           ‘T’ : hydrogen bonded turn
           ‘S’ : bend
           ‘ ‘ : Loops and irregular elements
           '''
           secondary_structure = []
           im = self.transform_function(xyz, secondary_structure)


           self.image_number += 1  # notice I'm incrementing image_number inside of next


           return im


       except StopIteration as E:
           # just pass it on
           raise E




    def __next__(self):
       try:
           traj = next(self.traj_wrapper)
           if self.image_number == 0: # notice I'm using image_number here to do this only once
               topology = traj.topology
               self.set_indices(topology)


           #select coordinates
           if self.xyz_indices.size > 0:
               xyz = traj.xyz[0, self.xyz_indices]
           else:
               xyz = traj.xyz[0, :]


           # select secondary structure
           # secondary_structure = md.compute_dssp(traj, simplified=False)[0]
           # if self.ss_indices.size > 0:
           #     secondary_structure = secondary_structure[self.ss_indices]


           secondary_structure = []
           # the function produces an array of strings denoting secondary structure of each residue
           # possible secondary structures are:


           im = self.transform_function(xyz, secondary_structure)


           self.image_number += 1  # notice I'm incrementing image_number inside of next


           return im


       except StopIteration as E:
           # just pass it on
           raise E
