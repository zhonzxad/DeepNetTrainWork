'''
Author: zhonzxad
Date: 2021-10-25 20:59:11
LastEditTime: 2021-11-26 20:03:54
LastEditors: zhonzxad
'''
# -*- coding: UTF-8 -*- 
import datetime as dt
import os
import sys
from loguru import logger
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

class GetWriteLog:
    def __init__(self, writerpath='', trace_func=print):
        super(GetWriteLog, self).__init__()
        self.writerpath = writerpath
        self.trace_func = trace_func

        self.init()

    def init(self):
        assert (self.writerpath is not "")

        creat_time = dt.datetime.now().strftime("%m-%d_%H:%M") # 10-25_15:08
        # frontpath  = os._path.abspath()
        self.file_name = self.writerpath + creat_time + ".txt"
        # self.file = open(file=self.file_name, mode='w', encoding="utf-8") # 用于读写，且清空文件

    def __del__(self):
        # self.write('这是析构函数')
        self.trace_func('日志对象析构函数')

    def write(self, log, type="info", coutprint=False):
        time = dt.datetime.now().strftime("%H:%M:%S") # 15:08:03
        log  = time + "| {}} |".format(type) + "{}".format(log) + "\n"
        self.trace_func(log)  # 重定向到print

        if coutprint == True:
            with open(file=self.file_name, mode='w', encoding="utf-8") as f:
                f.write(log + "\n")

    def info(self, log, coutprint=False):
        self.write(log, type="info", coutprint=coutprint)

    def debug(self, log, coutprint=False):
        self.write(log, type="debug", coutprint=coutprint)

    def warning(self, log, coutprint=False):
        self.write(log, type="warning", coutprint=coutprint)

    def critical(self, log, coutprint=False):
        self.write(log, type="critical", coutprint=coutprint)

    def error(self, log, coutprint=False):
        self.write(log, type="error", coutprint=coutprint)

    def success(self, log, coutprint=False):
        self.write(log, type="success", coutprint=coutprint)

class Getloguru:
    """根据loguru全局日志系统"""
    def __init__(self, writerpath=''):
        self.write_log = logger
        self.write_log.add(writerpath + "logfile_{time:MM-DD_HH:mm}.log", format="{time:DD Day HH:mm:ss} | {level} | {message}", filter="",
                   enqueue=True, encoding='utf-8', rotation="50 MB")

    def write(self, log, type="info"):
        if type == "info":
            self.write_log.info(log)
        elif type == "debug":
            self.write_log.debug(log)
        elif type == "warning":
            self.write_log.warning(log)
        elif type == "critical":
            self.write_log.critical(log)
        elif type == "error":
            self.write_log.error(log)