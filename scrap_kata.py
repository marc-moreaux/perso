import selenium
from selenium import webdriver
import time
import pandas as pd
import os
import zipfile

fNames = {}
driver = webdriver.Chrome('/home/mmoreaux/Downloads/chromedriver')
time.sleep(5)

for i in range(341):
    url = 'https://bibliotheques-specialisees.paris.fr/'
    url += 'ark:/73873/pf0000855431/{:04}/v0001'.format(i+1)

    driver.get(url)
    time.sleep(5)

    title = driver.find_element_by_id('IR-bookTitle')
    title = title.text
    fNames[i+1] = title

    dl_button = driver.find_element_by_id('hires')
    dl_button.click()
    time.sleep(2)

    confirm_button = driver.find_element_by_id('checkConditions')
    confirm_button.click()
    time.sleep(2)

    dl_button2 = driver.find_element_by_class_name('ui-dialog-buttonset')
    dl_button2.click()

    time.sleep(20)


dl_path = '/home/mmoreaux/Downloads/'
kata_path = dl_path + 'kata/'
if not os.path.isdir(kata_path):
    os.makedirs(kata_path)

dl_files = os.listdir(dl_path)
for f in dl_files:
    if f.startswith('ark__73873'):
        if f.endswith('.zip'):
            zip_ref = zipfile.ZipFile(dl_path + f)
            zip_ref.extractall(kata_path)
            zip_ref.close()

dl_files = os.listdir(kata_path)
for f in dl_files:
    if f.startswith('ark__73873'):
        if f.endswith('.jpg'):
            fIdx = int(f.split('_')[4])
            if fIdx >= 26:
                os.rename(kata_path + f, kata_path + fNames[fIdx])
        







