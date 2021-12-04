'''
Author: zhonzxad
Date: 2021-10-25 20:59:11
LastEditTime: 2021-11-26 20:03:54
LastEditors: zhonzxad
'''
# -*- coding: UTF-8 -*- 
import datetime as dt
import sys
sys.path.append("..")

class GetWriteLog:
    def __init__(self, writerpath='', trace_func=print):
        super(GetWriteLog, self).__init__()
        self.writerpath = writerpath
        self.trace_func = trace_func

        self.init()

    def init(self):
        assert (self.writerpath is not "")

        creat_time = dt.datetime.now().strftime("%m-%d_%H:%M") # 10-25_15:08
        # frontpath  = os.path.abspath()
        self.file_name = self.writerpath + creat_time + ".txt"
        # self.file = open(file=self.file_name, mode='w', encoding="utf-8") # 用于读写，且清空文件

    def __del__(self):
        # self.write('这是析构函数')
        self.trace_func('日志对象析构函数')

    def write(self, log, coutprint=False):
        time = dt.datetime.now().strftime("%H:%M:%S") # 15:08:03
        log  = time + " >> " + "{}".format(log) + "\r\n"
        self.trace_func(log)  # 重定向到print

        if coutprint == True:
            with open(file=self.file_name, mode='w', encoding="utf-8") as f:
                f.write(log + "\n")
