import numpy as np
from pydub import AudioSegment

import math
import os
import wave
import json
from matplotlib import pyplot as plt

def MP32WAV(mp3_path,wav_path):
    """
    这是MP3文件转化成WAV文件的函数
    :param mp3_path: MP3文件的地址
    :param wav_path: WAV文件的地址
    """
    print(mp3_path)
    # pydub.AudioSegment.converter = "D:\\ffmpeg\\bin\\ffmpeg.exe"            #说明ffmpeg的地址
    MP3_File = AudioSegment.from_mp3(file=mp3_path)
    MP3_File.export(wav_path,format="wav")

def Read_WAV_Left(wav_path):
    """
    这是读取wav文件的函数，音频数据是单通道的。返回json
    :param wav_path: WAV文件的地址
    """
    wav_file = wave.open(wav_path,'r')
    numchannel = wav_file.getnchannels()          # 声道数
    samplewidth = wav_file.getsampwidth()      # 量化位数
    framerate = wav_file.getframerate()        # 采样频率
    numframes = wav_file.getnframes()           # 采样点数
    print("channel", numchannel)
    print("sample_width", samplewidth)
    print("framerate", framerate)
    print("numframes", numframes)
    Wav_Data = wav_file.readframes(numframes)
    Wav_Data = np.frombuffer(Wav_Data,dtype=np.int16)

    #将多声道分开
    Wav_Data = np.reshape(Wav_Data, [numframes, numchannel])
    Wav_Data = Wav_Data[:,0]

    Wav_Data = Wav_Data * 1.0 / (max(abs(Wav_Data)))  # wave幅值归一化

    # print(Wav_Data.shape)
    # print(Wav_Data.size)
    # min_v = min((abs(Wav_Data)))
    # max_v = max((abs(Wav_Data)))

    # test_sum = 0
    # for v in Wav_Data:
    #     test_sum = test_sum + abs(v)
    # print("test+sum :"+ str(test_sum))

    #取多少个samples求平均值得到一个点
    wav_sum = 0
    second_list = [];
    rate = 16000

    for i in Wav_Data:
        wav_sum =wav_sum +abs(i)
    threshold = wav_sum/numframes
    wav_sum = 0

    for k,v in enumerate(Wav_Data):
        if(k%rate == 0 and k != 0):
            if(wav_sum/rate > threshold):
                second_list.append(1)
            else:
                second_list.append(0)
            wav_sum = 0
        wav_sum = abs(v) + wav_sum

    # f = open("out_left.txt", "w+")
    # for k,v in enumerate(second_list):
        # if(v > threshold):
        #     f.write("第"+str(k//60)+"分"+"第"+str(k%60)+"秒："+str(1)+'\n')
        # else:
        #     f.write("第" + str(k // 60) + "分" + "第" + str(k % 60) + "秒：" + str(0) + '\n')

        # f.write("第" + str(k // 60) + "分" + "第" + str(k % 60) + "秒：" + str(v) + '\n')
    # f.write(str(second_list))

    #按阀值分类
    # for k, v in enumerate(Wav_Data):
    #     if((v > 0.5) & (k*rate % 16000 ==0)):
    #         f.write(str(k*rate/16000)+ ":"+ str(v) + "\n")

    # f.close()

    # 生成音频数据,ndarray不能进行json化，必须转化为list，生成JSON
    # dict = {"channel":numchannel,
    #         "samplewidth":samplewidth,
    #         "framerate":framerate,
    #         "numframes":numframes,
    #         "WaveData":list(Wav_Data)}
    dict = {"channel":numchannel,
            "samplewidth":samplewidth,
            "framerate":framerate/rate,
            "numframes":numframes,
            "WaveData":second_list}


    return json.dumps(dict)

def Read_WAV_Right(wav_path):
    """
    这是读取wav文件的函数，音频数据是单通道的。返回json
    :param wav_path: WAV文件的地址
    """
    wav_file = wave.open(wav_path,'r')
    numchannel = wav_file.getnchannels()          # 声道数
    samplewidth = wav_file.getsampwidth()      # 量化位数
    framerate = wav_file.getframerate()        # 采样频率
    numframes = wav_file.getnframes()           # 采样点数
    print("channel", numchannel)
    print("sample_width", samplewidth)
    print("framerate", framerate)
    print("numframes", numframes)
    Wav_Data = wav_file.readframes(numframes)
    Wav_Data = np.frombuffer(Wav_Data,dtype=np.int16)

    #将多声道分开
    Wav_Data = np.reshape(Wav_Data, [numframes, numchannel])
    Wav_Data = Wav_Data[:,1]

    Wav_Data = Wav_Data * 1.0 / (max(abs(Wav_Data)))  # wave幅值归一化

    # print(Wav_Data.shape)
    # print(Wav_Data.size)
    # min_v = min((abs(Wav_Data)))
    # max_v = max((abs(Wav_Data)))

    # test_sum = 0
    # for v in Wav_Data:
    #     test_sum = test_sum + abs(v)
    # print("test+sum :"+ str(test_sum))

    #取多少个samples求平均值得到一个点
    wav_sum = 0
    second_list = [];
    rate = 16000

    for i in Wav_Data:
        wav_sum =wav_sum + abs(i)
    threshold = wav_sum/numframes
    wav_sum = 0

    for k,v in enumerate(Wav_Data):
        if(k%rate == 0 and k != 0):
            if(wav_sum/rate > threshold):
                second_list.append(1)
            else:
                second_list.append(0)
            wav_sum = 0
        wav_sum = abs(v) + wav_sum


    # f = open("out_right.txt", "w+")
    # for k,v in enumerate(second_list):
        # if(v > threshold):
        #     f.write("第"+str(k//60)+"分"+"第"+str(k%60)+"秒："+str(1)+'\n')
        # else:
        #     f.write("第" + str(k // 60) + "分" + "第" + str(k % 60) + "秒：" + str(0) + '\n')

        # f.write("第" + str(k // 60) + "分" + "第" + str(k % 60) + "秒：" + str(v) + '\n')
    # f.write(str(second_list))

    #按阀值分类
    # for k, v in enumerate(Wav_Data):
    #     if((v > 0.5) & (k*rate % 16000 ==0)):
    #         f.write(str(k*rate/16000)+ ":"+ str(v) + "\n")

    # f.close()

    # 生成音频数据,ndarray不能进行json化，必须转化为list，生成JSON
    # dict = {"channel":numchannel,
    #         "samplewidth":samplewidth,
    #         "framerate":framerate,
    #         "numframes":numframes,
    #         "WaveData":list(Wav_Data)}
    dict = {"channel":numchannel,
            "samplewidth":samplewidth,
            "framerate":framerate/rate,
            "numframes":numframes,
            "WaveData":second_list}


    return json.dumps(dict)


def DrawSpectrum(wav_data,framerate):
    """
    这是画音频的频谱函数
    :param wav_data: 音频数据
    :param framerate: 采样频率
    """

    Time = np.linspace(0,len(wav_data)/framerate*1.0,num=len(wav_data))
    plt.figure(1)
    plt.title("samples per points")
    plt.title("rate = "+str(16000/framerate))
    plt.plot(Time,wav_data)
    plt.grid(True)
    plt.show()

def run_main():
    """
        这是主函数
    """
    # MP3文件和WAV文件的地址
    path1 = './MP3_File'
    path2 = "./WAV_File"
    paths = os.listdir(path1)
    mp3_paths = []
    # 获取mp3文件的相对地址
    for mp3_path in paths:
        mp3_paths.append(path1+"/"+mp3_path)
    print(mp3_paths)

    # 得到MP3文件对应的WAV文件的相对地址
    wav_paths = []
    for mp3_path in mp3_paths:
       wav_path = path2+"/"+mp3_path[1:].split('.')[0].split('/')[-1]+'.wav'
       wav_paths.append(wav_path)
    print(wav_paths)

    # 将MP3文件转化成WAV文件
    for(mp3_path,wav_path) in zip(mp3_paths,wav_paths):
        MP32WAV(mp3_path,wav_path)
    # for wav_path in wav_paths:
    #     Read_WAV(wav_path)

    # 开始对音频文件进行数据化
    for wav_path in wav_paths:
        wav_json_left = Read_WAV_Left(wav_path)
        print(wav_json_left)
        wav_json_right = Read_WAV_Right(wav_path)
        print(wav_json_right)
        wav_left = json.loads(wav_json_left)
        wav_data_left = np.array(wav_left['WaveData'])
        wav_right = json.loads(wav_json_right)
        wav_data_right = np.array(wav_right['WaveData'])
        sum_1 = 0
        sum_equal = 0
        for k,v in enumerate(wav_data_right):
            if(v == 1):
                sum_1 += 1
                if(wav_data_left[k] == 1):
                    sum_equal += 1

        print("比例为"+str(sum_equal/sum_1))
        wav_path = wav_path.split('/')[2]
        f = open("比对结果.txt", "a")
        if((sum_equal/sum_1)> 0.5):
            f.write(wav_path +" "+ ":左右分声道重合"+ " "+"重合率为"+str(sum_equal/sum_1)+'\n')
        else:
            f.write(wav_path +" "+ ":左右分声道分开"+" " +"重合率为"+str(sum_equal/sum_1)+'\n')
        f.close()

        # wav = json.loads(wav_json)
        # wav_data = np.array(wav['WaveData'])
        # framerate = int(wav['framerate'])
        # DrawSpectrum(wav_data,framerate)

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    run_main()
