# -*- coding: UTF-8 -*- 
import datetime as dt
import os


class WriteLog:
    def __init__(self, writerpath='./../../log/log/', trace_func=print):
        super(WriteLog, self).__init__()
        self.writerpath = writerpath
        self.trace_func = trace_func

        self.init()

    def init(self):
        creat_time = dt.datetime.now().strftime("%F_%H-%M-%S") # 2021-10-25_15:08:03
        # frontpath  = os.path.abspath()
        self.file_name = self.writerpath + creat_time + ".txt"

        self.file = open(file=self.file_name, mode='w', encoding="utf-8") # 用于读写，且清空文件

    def __del__(self):
        # self.write('这是析构函数')
        self.trace_func('日志对象析构函数')
        self.file.close()

    def write(self, log, coutprint=True):
        assert (self.file)    # 文件对象必须先存在
        time = dt.datetime.now().strftime("%T") # 2021-10-25_15:08:03
        log = time + " >> " + "{}".format(log)
        self.trace_func(log)  # 重定向到print
        if coutprint:
            self.file.write(log + "\n")
