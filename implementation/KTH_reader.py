import numpy as np
import pickle
import os
import re
import video




class KTH_videos:
  def __init__(self):
    ''' Declare path to the folder where every avi were extracted from the zips
    and declare an array with the class labels '''
    self.KTH_folder = "/home/mmoreaux/datasets/KTH/"
    self.avi_dir = "/home/mmoreaux/datasets/KTH/"
    self.labels = ["running", "jogging", "walking", "boxing", "handclapping", "handwaving"]
    self.subsets = {"train":[1,2,3,4],  "valid":[5], "test":[6] }
  
  def db_path(self, subset):
    ''' return the path of a file '''
    return self.KTH_folder+"db"+str(subset)+".npy"
  
  def create_db(self):
    ''' Create dataset pickles from avi files
    Each pickle is a tupple : (video_nparray, video_labels) 
    where  video_nparray is an array of 500 items 
    each item is a video sequence composed by 15 frames (3s at 5fps)'''
    f = os.listdir(self.avi_dir)[0]
    vid = video.asvideo(os.path.join(self.avi_dir,f)).V
    # 500 samples of 3sec videos
    ds_data   = np.ndarray( (500,15)+vid.shape[1:] , dtype='int8')
    ds_labels = np.ndarray( (500,3), dtype='uint8')
    
    # Divide dataset into x files
    # there is a mistake at (idx+idx2)%500 with overlap possibilities at 501 == 1
    idx = 0
    for f in os.listdir(self.avi_dir):
      # print(f)
      if ".avi" in f:
        vid = video.asvideo(os.path.join(self.avi_dir,f)).V
        for idx2 in range(int(len(vid)/15)-1):
          ds_data[(idx)%500] = vid[idx2*15 : (idx2+1)*15 ]-125
          ds_labels[(idx)%500,0] = int(re.findall(r'\d+', f)[0])
          ds_labels[(idx)%500,1] = int(re.findall(r'\d+', f)[1])
          ds_labels[(idx)%500,2] = self.labels.index( re.findall(r'_(.*)_d+', f)[0] )
          if idx % 500 == 0 and idx > 0: 
            # Save subset in a file
            print(idx," saving to ", self.db_path(int(idx/500)))
            self.write_db(int(idx/500), ds_data, ds_labels)
          idx += 1
  
  def read_db(self, subset):
    fp = open(self.db_path(subset), "rb")
    tmp_data, tmp_labels = pickle.load(fp)
    fp.close()
    return tmp_data, tmp_labels
  
  def write_db(self, subset, data, labels):
    fp = open(self.db_path(subset), "wb")
    pickle.dump((data,labels), fp)
    fp.close()
  
  def shuffle_db(self):
    ''' As you should have 6 files, this function shuffles 
    the data between these 6 files'''
    for idx in [ (1,2), (3,4), (5,6), (1,4), (3,6), (2,5), (2,3), (4,5), (1,6) ] :
      # Open files
      print("shuffling %d and %d"%(idx[0],idx[1]) )
      data1, labels1 = self.read_db(idx[0])
      data2, labels2 = self.read_db(idx[1])
      
      # Concatenate and shuffle array
      tmp_data   = np.concatenate( (data1,data2), axis=0 )
      tmp_labels = np.concatenate( (labels1,labels2), axis=0 )
      p = np.random.permutation(len(tmp_data))
      tmp_data   = tmp_data[p]
      tmp_labels = tmp_labels[p]
      
      # Save arrays and close files
      print("saving files %d and %d"%(idx[0],idx[1]) )
      self.write_db(idx[0], tmp_data[:500], tmp_labels[:500])
      self.write_db(idx[1], tmp_data[500:], tmp_labels[500:])
      print("--")
  
  def batch_generator(self, batch_size=20, subsets=[1,2,3,4,5,6]):
    ''' Generator yielding following batch'''
    while True:
      for subset in subsets:
        data, labels = self.read_db(subset)
        goodShape = [batch_size,data.shape[1],227,227,data.shape[-1]]
        goodData = np.zeros(goodShape, dtype='float16')
        for curr_bIdx in range(int(len(labels)/batch_size-1)):
          _idx1 = curr_bIdx*batch_size
          _idx2 = _idx1+batch_size
          goodData[:,:,:data.shape[2],:data.shape[3],:] = data[_idx1:_idx2]/150
          yield (goodData, labels[_idx1:_idx2])
  
  def train_batch_generator(self, batch_size=20):
    ''' training batch generator '''
    return self.batch_generator(batch_size,self.subsets['train'])
  
  def valid_dataset(self):
    tmp = self.batch_generator(500,self.subsets['valid'])
    return next(tmp)
  
  def train_dataset(self):
    tmp = self.batch_generator(500,self.subsets['test'])
    return next(tmp)





if __name__ == '__main__':
  a = KTH_videos()
  a.create_db()
  a.shuffle_db()
  y = a.read_db(1)
  y[1][:20]
  b = a.batch_generator(20)
  next(b)