import numpy as np
from scipy.io import wavfile
import math
# frombuffer将data以流的形式读入转化成ndarray对象
# 第一参数为stream,第二参数为返回值的数据类型，第三参数指定从stream的第几位开始读入

# data是字符串的时候，Python3默认str是Unicode类型，所以要转成bytestring在原str前加上b
# s = b'abc'
# a = np.frombuffer(s, dtype='S1', offset=1)
#
# def splitChannel(secMusicFile):
#     sampleRate,musicData= wavfile.read(secMusicFile)
#     left = []
#     right = []
#     for item in musicData:
#         left.append(item[0])
#         right.append(item[1])
#     wavfile.write('lfet.wav',sampleRate,np.array(left))
#     wavfile.write('right.wav', sampleRate, np.array(right))
#
#
# splitChannel("./WAV_File/20181020_174015.wav");
#
# from pydub import AudioSegment
#
# song = AudioSegment.from_mp3("./MP3_File/Left-2003472462-1528614615830.mp3")
# f = open("Ssegment.txt", "w+")
#
# for k, v in enumerate(song):
#     f.write(str(v))
# f.close()

# for k, v in enumerate(Wav_Data):
#     sumValue = v + sumValue
#     if (k % 2 == 0 and k != 0):
#         l.append(sumValue/2)
#         sumValue = 0



# a1 = np.array([[2,1],[2,2],[2,3],[2,4]],dtype=np.int16)
# a1 = np.array([1,2,3,4,5,6,7,8],dtype=np.int16)
# a1 = np.reshape(a1,[4,2])
# print(a1)
# print("数据类型",type(a1))           #打印数组数据类型
# print("数组元素数据类型：",a1.dtype) #打印数组元素数据类型
# print("数组元素总数：",a1.size)      #打印数组尺寸，即数组元素总数
# print("数组形状：",a1.shape)         #打印数组形状
# print("数组的维度数目",a1.ndim)      #打印数组的维度数目

x = "./WAV_File/2003472462-1528614615830.wav"
y = x.split('/')
print(y[2])

