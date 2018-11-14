
# python wrapper for basic ffmpeg operations
# resize video, check if a video is corrupted, etc.

import subprocess, re, os
from PIL import Image
import numpy as np
import random
import pickle
import video




#########################
###  Usage example
#########################
# import UCF_reader
# a = UCF_reader.UCF_videos()
# b = a.next_batch(batchSize=1)
# y = next(b)



class UCF_videos:
  def __init__(self):
    self.pwd = os.path.abspath("./")
    self.path = self.pwd+"/UCF-101"
    self.classes = ["Apply Eye Makeup", "Apply Lipstick", "Baby Crawling", "Bench Press", "Biking", "Blow Dry Hair", "Blowing Candles", "Brushing Teeth", "Cutting In Kitchen", "Frisbee Catch", "Haircut", "Handstand Pushups", "Handstand Walking", "Head Massage", "Jump Rope", "Jumping Jack", "Knitting", "Lunges", "Mopping Floor", "Pizza Tossing", "Playing Guitar", "Playing Piano", "Playing Violin", "Playing Flute", "Shaving Beard", "Tai Chi", "Typing", "Walking with a dog", "Writing On Board"]
  
  def path_of_pkl(self, subset, idx):
    """
        Return pickle path
    """
    return "%s/videos_s%d_idx%05d.pkl" %(self.path, subset, idx)
    return self.path+'/videos_s'+str(subset)+'_idx'+str(idx)+'.pkl'
  
  def save_and_reset_ds(self, filePath):
    """ 
        Save self.dataset and reset it 
    """
    with open(filePath, 'wb') as pickle_file:
      pickle.dump(self.dataset, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
      self.dataset = list()
  
  def load_ds(self, filePath):
    """
        Loads a pickle file
    """
    with open(filePath, 'rb') as pickle_file:
      return pickle.load(pickle_file)
  
  def video_paths(self, subset=1):
    """ 
        Make a path list with the videos to extract
        returns a tupple (video-path / video-class)
    """
    if subset == -1:
      vidFolders = [os.path.join(self.path ,f) for f in os.listdir(self.path)]
      vidFiles = [ os.path.join(vidFold,vidName) for vidFold in vidFolders for vidName in os.listdir(vidFold)]
      return vidFiles
    elif subset > 3:
      print("subset variable is in between 1, 2 and 3")
      return
    else:
      f1 = open(os.path.join(self.path,"trainlist0"+str(subset)+".txt"))
      f2 = open(os.path.join(self.path,"testlist0" +str(subset)+".txt"))
      vidFiles = f1.readlines()
      vidClass = [ f.rstrip().split(' ')[1] for f in vidFiles]
      vidFiles = [ f.rstrip().split(' ')[0] for f in vidFiles]
      vidFiles = [os.path.join(self.path, f) for f in vidFiles]
      f1.close()
      f2.close()
  
    return (vidFiles, vidClass)
  
  def extract_video_subset(self, subset=1, framerate=5):
    """
        for each video path, 
         - extract the images (ffmepg)
         - make a numpy array with them
         - append them to <dataset>
    """
    self.framerate = framerate
    self.dataset = list()
    
    (vidFiles, vidClass) = self.video_paths(1)
    idx=0
    for idx,(vid,vidclass) in enumerate(zip(vidFiles,vidClass)):
      # Extract every videos with ffmpeg
      # call ffmpeg and grab its stderr output
      # forces extracted frames to be 227*227 dim.
      if not os.path.exists(vid):
        print('%s does not exist!' % vid)
        return False
      p = subprocess.call('ffmpeg -i %s -s 227*227 -r %d /tmp/tmp%%4d.jpg' % (vid,framerate), shell=True)
            
      # transorm images into numpy arrays
      frames_p = [f for f in os.listdir("/tmp") if ("tmp" in f and ".jpg" in f)]
      frames_p = [os.path.join("/tmp/",f) for f in frames_p]
      frames_p.sort()
      pix = np.array(Image.open(frames_p[0]))
      npFrames = np.ndarray((len(frames_p),) + pix.shape, dtype="uint8")
      for idx2,frame_p in enumerate(frames_p):
        npFrames[idx2] = np.array(Image.open(frame_p))
      p = subprocess.call('rm /tmp/tmp*.jpg', shell=True)
      
      # Append numpy array video and video's class
      self.dataset.append((npFrames,vidclass))
      
      #
      # Save the dataset every 2 videos
      #
      print(idx)
      if idx % 2 == 0:
        print("saving at",self.path_of_pkl(subset, int(idx/2)))
        self.save_and_reset_ds(self.path_of_pkl(subset, int(idx/2)))
    
    # don't save the last videos
    # self.save_and_reset_ds(self.path_of_pkl(subset, int(idx/2)+1))
  
  def next_random_video(self, subset=1):
    # 
    # Create an array with shuffled ids
    #
    while(True):
      if not( hasattr(self, 'shuffledIdx') and len(self.shuffledIdx) >= 0 ):
        allPkl = [os.path.join(self.path ,f) for f in os.listdir(self.path) if ("s"+str(subset)and".pkl") in f]
        self.shuffledIdx = list(range(len(allPkl)))
        random.shuffle(self.shuffledIdx)
      
      while True:
        loaded_file = self.load_ds( self.path_of_pkl(1,self.shuffledIdx.pop()) )
        if len(loaded_file) == 2 :
          (video1, video2) = loaded_file
          break
      
      yield (video1[0], int(video1[1]))
      yield (video2[0], int(video2[1]))
  
  def next_batch(self, subset=1, batchSize=20):
    """
        Provide the next batch of <batchSize> videos
        of 15 frames of size (227*227*3)
    """
    #
    # create an array of videos which will be decomposed
    #  into sequences of 15 frames in <batch_train>
    #  set <batch_labels> to their proper values
    #
    video_gen = self.next_random_video(1)
    batch_train = np.ndarray((batchSize,15,227,227,3))
    batch_labels = np.zeros(batchSize, dtype='uint8')
    video_train = [0 for idx in range(batchSize)]
    video_cur_idx = [0 for idx in range(batchSize)]
    for idx in range(batchSize):
      (video_train[idx], batch_labels[idx]) = next(video_gen)
    
    #
    # Generate the following batch to yield
    #
    while(True):
      for idx in range(batchSize):
        # train on following sequence of the video (or...)
        if(len(video_train[idx]) > video_cur_idx[idx]+15): 
          batch_train[idx,:,:,:,:] = video_train[idx][video_cur_idx[idx]:video_cur_idx[idx]+15,:,:,:]
          video_cur_idx[idx] += 10 # train on next 2 seconds
        else:
          # load following video (if greather than 15 frames)
          while True:
            (video_train[idx], batch_labels[idx]) = next(video_gen)
            if len(video_train[idx]) > 15:
              break
          
          video_cur_idx[idx] = 0
          batch_train[idx,:,:,:,:] = video_train[idx][video_cur_idx[idx]:video_cur_idx[idx]+15,:,:,:]
          batch_train = batch_train / np.max(batch_train)
          batch_train = batch_train - np.mean(batch_train)
      # Here is next batch
      yield batch_train, batch_labels
  















class UCF_videos:
  def __init__():
    ''' Declare path to the folder where every avi were extracted from the zips
    and declare an array with the class labels '''
    self.UCF_folder = "/home/mmoreaux/datasets/UCF/"
    self.avi_dir = "/home/mmoreaux/datasets/UCF/"
    self.labels = ["ApplyEyeMakeup","ApplyLipstick","BabyCrawling","BenchPress","Biking","BlowDryHair","BlowingCandles","BrushingTeeth","CuttingInKitchen","FrisbeeCatch","Haircut","HandstandPushups","HandstandWalking","HeadMassage","JumpRope","JumpingJack","Knitting","Lunges","MoppingFloor","PizzaTossing","PlayingGuitar","PlayingPiano","PlayingViolin","PlayingFlute","ShavingBeard","TaiChi","Typing","WalkingWithDog","WritingOnBoard"]
    self.subsets = {"train":[1,2,3,4],  "valid":[5], "test":[6] }

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

