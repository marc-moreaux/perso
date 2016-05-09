
# python wrapper for basic ffmpeg operations
# resize video, check if a video is corrupted, etc.

import subprocess, re, os
from PIL import Image
import numpy as np
import random
# import matplotlib.pyplot as plt
import pickle

# provide your own ffmpeg here
ffmpeg = 'ffmpeg'



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
    self.train01 = list()
    self.test01  = list()
    self.dataset = list()
  
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
  
#  def play_batch(self, npVideoFrames, framerate=5):
#    self.framerate = 5
#    im = plt.imshow(npVideoFrames[0])
#    for (idx,frame) in enumerate(npVideoFrames):
#     plt.pause(1/self.framerate)
#     im.set_data(frame)
#   plt.show()



# a = UCF_videos()
# a.extract_video_subset(1)
# b = a.next_batch(batchSize=5)
# y = next(b)
# # a.play_batch(y[0][0])





# play(a.dataset[0][0])



# # resize videoName to 320x240 and store in resizedName
# # if succeed return True
# def resize(videoName, resizedName):
#     if not os.path.exists(videoName):
#         print('%s does not exist!' % videoName)
#         return False
#     # call ffmpeg and grab its stderr output
#     p = subprocess.Popen([ffmpeg, "-i", videoName], stderr=subprocess.PIPE)
#     out, err = p.communicate()
#     # search resolution info
#     if err.find('differs from') > -1:
#         return False
#     reso = re.findall(r'Video.*, ([0-9]+)x([0-9]+)', err)
#     if len(reso) < 1:
#         return False
#     # call ffmpeg again to resize
#     subprocess.call([ffmpeg, '-i', videoName, '-s', '320x240', resizedName])
#     return check(resizedName)

# # check if the video file is corrupted or not
# def check(videoName):
#     if not os.path.exists(videoName):
#         return False
#     p = subprocess.Popen([ffmpeg, "-i", videoName], stderr=subprocess.PIPE)
#     out, err = p.communicate()
#     if err.find('Invalid') > -1:
#         return False
#     return True

# def extract_frame(videoName,frameName):
#     """Doc
#     Extracts the frames from the input video (videoName)
#     and saves them at the location (frameName)
#     """
#     #forces extracted frames to be 320x240 dim.
#     if not os.path.exists(videoName):
#         print('%s does not exist!' % videoName)
#         return False
#     # call ffmpeg and grab its stderr output
#     print('ffmpeg -i %s -r 1 -s qvga --f image2 %s' % (videoName,frameName))
#     p = subprocess.call('ffmpeg -i %s -r 10 %s' % (videoName,frameName), shell=True)
#     return


# def extract_frames(vidlist,vidDir,outputDir):
#   f = open(vidlist, 'r')
#   vids = f.readlines()
#   f.close()
#   fp = open("out.txt", "w+")
#   vids = [video.rstrip() for video in vids]
#   vids = [line.split(' ')[0] for line in vids]
#   for vid in vids:
#       videoName = os.path.join(vidDir,vid)
#       frameName = os.path.join(outputDir, vid.split('.')[0]+"_%4d.jpeg")
#       newFrameName = ""
#       for piece in (frameName.split("/")[1:-3] + frameName.split("/")[-1:]):
#         newFrameName += "/"+piece 
#       fp.write(newFrameName)
#       fp.write("\n")
#       #extract_frame(videoName,newFrameName)
#   fp.close()





# orig_dir = "./UCF-101"
# tmp_frames = "./UFC101_raw/train/tmp"

# #playing a video saved in disk
# video_pwd = os.path.join(orig_dir,"v_ApplyEyeMakeup_g01_c03.avi")

# pwd = os.path.abspath("./")
# ucf101_path = pwd+"/UCF-101"
# trainlists = pwd+"/ucfTrainTestlist/"
# trainlist01 = os.path.join(trainlists, 'trainlist01.txt')
# testlist01 = os.path.join(trainlists, 'testlist01.txt')
# training_output = pwd+"/UFC-101_raw/train"
# testing_output = pwd+"/UFC-101_raw/test"


# extract_frames(trainlist01, ucf101_path,training_output)
# extract_frames(testlist01, ucf101_path,testing_output)
