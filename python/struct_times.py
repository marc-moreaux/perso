import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import functools
import random
import time
sns.set()


df = pd.DataFrame({'index': list(range(0, 1000, 50))})
df = df.set_index('index')

def time_and_save(func):
    '''Time a function and plot its time'''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ys = []
        for x in df.index:
            start = time.time()
            for _ in range(200):
                func(n=x, *args, **kwargs)
            duration = time.time() - start
            ys.append(duration)
        df[func.__name__] = ys
    return wrapper


def populate_dict(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        m_dict = dict()
        for i in range(df.index.max() + 5):
            m_dict[i] = True
        func(m_dict=m_dict, *args, **kwargs)
    return wrapper


def populate_list(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        m_list = list()
        for i in range(df.index.max() + 5):
            m_list.append(True)
        func(m_list=m_list, *args, **kwargs)
    return wrapper


@time_and_save
@populate_dict
def dict_populate(m_dict, n):
    pass
 
@time_and_save
@populate_dict
def dict_insert(m_dict, n):
    for i in range(n):
        m_dict[i] = False

@time_and_save
@populate_list
def list_populate(m_list, n):
    pass

@time_and_save
@populate_list
def list_insert(m_list, n):
    for i in range(n):
        m_list.append(i)

@time_and_save
@populate_list
def list_pop(m_list, n):
    for _ in range(n):
        m_list.pop()

@time_and_save
@populate_list
def list_pop_first(m_list, n):
    for _ in range(n):
        m_list.pop(0)

@time_and_save
@populate_list
def list_pop_random(m_list, n):
    list_len_init = len(m_list)
    for i in range(n):
        m_list.pop(random.randint(0, list_len_init - i - 1))


dict_populate()
dict_insert()

list_populate()
list_insert()
list_pop()
list_pop_first()
list_pop_random()

ligs = cols = int(np.sqrt(len(df.columns)) + 0.9999)
fig, axs = plt.subplots(ligs, cols)
for i, (key, col) in enumerate(df.items()):
    _ax = axs[int(i/ligs), i%cols]
    col.plot(ax=_ax)
    _ax.set_title(key)
plt.show()
