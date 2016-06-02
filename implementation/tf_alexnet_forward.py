import os
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import matplotlib.cbook as cbook
import time
import scipy
from PIL import Image
import tensorflow as tf

# own stuff
import tf_alexnet2
from caffe_classes import class_names

imgs = [
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_175712.jpg", 0],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_175716.jpg", 0],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_175729.jpg", 0],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_175736.jpg", 0],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_175747.jpg", 0],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_175752.jpg", 0],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_175804.jpg", 0],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_175814.jpg", 0],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_175838.jpg", 0],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_175844.jpg", 0],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_175908.jpg", 0],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_175928.jpg", 0],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180009.jpg", 0],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180020.jpg", 0],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180031.jpg", 0],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180232.jpg", 0],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180334.jpg", 0],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180355.jpg", 0],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180403.jpg", 0],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180411.jpg", 0],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180431.jpg", 0],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180728.jpg", 1],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180733.jpg", 1],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180740.jpg", 1],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180743.jpg", 1],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180745.jpg", 1],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180747.jpg", 1],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180749.jpg", 1],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180756.jpg", 1],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180758.jpg", 1],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180806.jpg", 1],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180808.jpg", 1],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180811.jpg", 1],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180813.jpg", 1],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180844.jpg", 1],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180911.jpg", 1],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180923.jpg", 1],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_180933.jpg", 1],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_181056.jpg", 2],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_181059.jpg", 2],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_181102.jpg", 2],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_181110.jpg", 2],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_181116.jpg", 2],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_181130.jpg", 2],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_181138.jpg", 2],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_181211.jpg", 2],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_181257.jpg", 2],["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_181401.jpg", 2],
["/home/mmoreaux/Desktop/photos/Photos/IMG_20160526_181411.jpg", 2]]

imgPath = [img[0] for img in imgs]
results = list()

# Initalize alexnet
graph = tf.Graph()
mAlexnet = tf_alexnet2.Alexnet()
mAlexnet.get_graph()
init_op = tf.initialize_all_variables()



sess = tf.Session()
sess.run(init_op)
print('Initialized')
for imgP in imgPath:
  img = Image.open(imgP)
  img = img.resize([227,227], Image.ANTIALIAS)
  img = np.array(img)
  img = img.reshape(1,227,227,3)
  feed_dict = {}
  feed_dict[mAlexnet.mInput] = img
  prediction = sess.run([mAlexnet.prob], feed_dict=feed_dict)
  inds = np.argsort(prediction)[0,:][0]
  results.append([imgP,]+[(class_names[inds[-1-i]], prediction[0][0][inds[-1-i]]) for i in range(20)])




# analyse results
from collections import Counter
Counter([preds[0] for result in results[0:20] for preds in result])
Counter([preds[0] for result in results[20:35] for preds in result])
Counter([preds[0] for result in results[36:48] for preds in result])

labels = {}
labels["coffee"] = ["paper towel","toilet tissue, toilet paper, bathroom tissue","coffee mug","coffeepot","cup","measuring cup"]
labels["comput"] = ["screen, CRT screen","mouse, computer mouse","monitor","desktop computer","notebook, notebook computer","laptop, laptop computer","computer keyboard, keypad"]
labels["chairs"] = ["toilet seat","folding chair","crutch","rocking chair, rocker"]

preds = {}
preds["coffee"] = [result[1:] for result in results[0:20]  ]
preds["comput"] = [result[1:] for result in results[20:35] ]
preds["chairs"] = [result[1:] for result in results[36:48] ]


def eval_prediction(className, nbElems):
  score = 0
  for pred in preds[className]:
    for label in labels[className]:
      if label in [x[0] for x in pred][0:nbElems]:
        score += 1
        break
  return score/len(preds[className])


for nbElems in [1,5,10]:
  print("\n ---  Top %d"%(nbElems))
  for class_names in ["coffee", "comput", "chairs"]:
    print("%s : %f" % (class_names, eval_prediction(class_names,nbElems)))







# import video
# a = video.asvideo("/home/mmoreaux/Desktop/recording2.avi")

# vids = a.V

# import matplotlib.pyplot as plt
# im = plt.imshow(vids[0])
# for (idx,frame) in enumerate(vids):
#   plt.pause(1/5)
#   im.set_data(frame)

# plt.show()