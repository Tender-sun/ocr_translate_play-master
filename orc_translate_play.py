#-*- coding:utf-8 -*-
import os
import ocr
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob
image_files = glob('./test_images/*.*')
from transformers import pipeline

# 这部分因为数据集过大，训练复现难度过大。我们使用基于预训练的方法进行构建模型
zh_en = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
en_zh = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")
en_fr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
fr_en = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
en_de = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")
de_en = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")
def zho2eng(zho_str):
  return zh_en(zho_str)[0]['translation_text']
def zho2fr(zho_str):
  en_str = zh_en(zho_str)[0]['translation_text']
  return en_fr(en_str)[0]['translation_text']
def zho2de(zho_str):
  en_str = zh_en(zho_str)[0]['translation_text']
  return en_de(en_str)[0]['translation_text']
def en2zh(en_str):
  return en_zh(en_str)[0]['translation_text']
def en2fr(en_str):
  return en_fr(en_str)[0]['translation_text']
def en2de(en_str):
  return en_de(en_str)[0]['translation_text']
def fr2en(fr_str):
  return fr_en(fr_str)[0]['translation_text']
def de2en(de_str):
  return de_en(de_str)[0]['translation_text']

if __name__ == '__main__':
    result_dir = './test_result'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)
    res = ''
    for image_file in sorted(image_files):
        image = np.array(Image.open(image_file).convert('RGB'))
        t = time.time()
        result, image_framed = ocr.model(image)
        output_file = os.path.join(result_dir, image_file.split('/')[-1])
        Image.fromarray(image_framed).save(output_file)
        print("Mission complete, it took {:.3f}s".format(time.time() - t))
        print("\nRecognition Result:\n")
        for key in result:
            print(result[key][1])
            res+=result[key][1]
    '''
    调用预训练模型进行翻译功能的实现
    '''

    #将原始结果保存到txt文件
    with open('res.txt','w') as r:
        r.write(res+'\n')
    eng = zho2eng(res)
    de = zho2de(res)
    fr = zho2fr(res)
    #翻译好的内容追加至txt文件中
    print("这段话的英语翻译为{}".format(eng))
    with open('res.txt', 'a', encoding='utf-8') as r:
        r.write('en:'+eng+'\n')
    print("这段话的德语翻译为{}".format(de))
    with open('res.txt', 'a', encoding='utf-8') as r:
        r.write('de:'+de+'\n')
    print("这段话的法语翻译为{}".format(fr))
    with open('res.txt', 'a', encoding='utf-8') as r:
        r.write('fr:'+fr+'\n')



    '''
    下面实现最后一个任务：语音播放
    '''
    import pyttsx3

    # 初始化一个朗读引擎
    engine = pyttsx3.init()
    # 阅读模型的识别中文文本，并保存播放音频
    engine.say(res)
    engine.save_to_file(res, "chinese.mp3")
    engine.say(eng)
    engine.save_to_file(eng, "english.mp3")
    # 运行并且等到播放完毕
    engine.runAndWait()
