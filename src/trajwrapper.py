import numpy as np
import mdtraj as md


class TrajWrapper(object):
   '''
   traj_path: path to trajectory file or list of paths
   top: path to topology file


   '''
   def __init__(self, traj_path, top=None):
       self.top = top
       if isinstance(traj_path, (list,np.ndarray)):
           self.islist = True
           self.traj_paths_iter = iter(traj_path)
       elif isinstance(traj_path, str):
           self.islist = False
           self.traj = md.iterload(traj_path, top=top, chunk=1)
           self.traj_path = traj_path
           self.top = top
           #print(self.traj)


       else:
           raise Exception('Wrong type: traj_path must be a list, ndarray, or a string')


   def __iter__(self):

       return self


   def __next__(self):
       try:
           if self.islist:
               return md.load(next(self.traj_paths_iter), self.top)
           else:
               #print(self.traj)
               return next(self.traj)
       except StopIteration as E:
           # just pass it on
           raise E


   def load_frame(self,nframe):
       
       return md.load_frame(self.traj_path, nframe, top = self.top)


       # except StopIteration as E:
       #     # just pass it on
       #     raise E


