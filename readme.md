# 数据说明

## 训练数据
训练数据共有10万条光谱，光谱的类型有星系、类星体和恒星。每条光谱被插值到相同的波长采样点：3900A-9000A，均匀采3000个点。10万条光谱被存储在10个fits文件中，每个fits文件存储1万条光谱。fits文件的结果如下：

| No.  | Name      | Type         |   Dimensions     |  Format    |
|------|---------|-------------|-------------|---------|
|0     | PRIMARY   | PrimaryHDU    |  (3000, 10000)   |  float64.  |
| 1    |           | BinTableHDU   |  10000R x 2C    |  ['K', 'K']  |

第一个块（block）中每一行是一条光谱，第二个块是一个表，有两列，分别是objid和 label。objid与光谱是对应的，label记录着这条光谱的类型。如下表：

|类型        |      label   |
|--------|---------|
|GALAXY   |     0         |
|QSO        |     1          |
|STAR       |     2         |

## 测试数据
测试数据有1000条光谱，其文件格式与训练数据类似，只是在第二个块（block）中，去掉了label列。光谱的分类需要通过模型预测。

## 提交文件

提交文件为 CSV 格式，即一个文本文件，第一行是表头，为固定的 “#objid,label”。每行代表一条光谱的结果，包含两个数，中间用逗号分隔。例子如下：

``` 
#objid,label
20001,0
20002,1
```
## 文件读取

文件读取可以参考读取文件 sample_read.py

``` python
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

hdulist = fits.open('train_data_05.fits')
num = 5001 # the 5001st spectra in this fits file

flux = hdulist[0].data[num-1]
objid = hdulist[1].data['objid'][num-1]
label = hdulist[1].data['label'][num-1]
wavelength = np.linspace(3900,9000,3000)

c = {0:'GALAXY',1:'QSO',2:'STAR'}
plt.plot(wavelength,flux)
plt.title(f'class:{c[label]}')
plt.xlabel('wavelength ({})'.format(f'$\AA$'))
plt.ylabel('flux')
plt.show()
```



