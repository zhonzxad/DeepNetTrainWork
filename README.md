<!--
 * @Author: zhonzxad
 * @Date: 2021-06-24 10:10:02
 * @LastEditTime: 2021-12-16 21:15:18
 * @LastEditors: zhonzxad
-->
# UNet 分割类型框架集合
---
#### 使用集成环境安装
* 导出conda安装使用命令conda list -e > requirements.txt 
* 导入使用conda install --yes --file requirements.txt
* 导出pip安装使用命令pip freeze > requirements.txt
* 导入使用pip install -r requirements.txt

#### 自主创建虚拟环境安装
* pytorch 1.7.1(CUDA 10.2) 使用: 
`conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch`
* tensotboardX 2.4.1 方便记录日志情况, 使用: 
`pip install tensorboardX`
* torchsummary 1.5.1 方便测试模型大小情况, 使用: 
`pip install torchsummary`
* pynvml 11.4.1 处理显卡相关参数的信息, 使用: 
`pip install pynvml`
* tqdm 4.62.3 使用进度条来计算时间, 使用:
`pip install tqdm`
* 依赖platform, argparse等基础库
---
其余参考 
`profile&store 下 pip-requirements.txt && conda-requirements.txt`

---
## LICENSE
GNU Lesser General Public License v2.1

[![](https://img.shields.io/badge/license-GPL%20v2.1-brightgreen?style=plastic)](https://github.com/zhonzxad/DeepNetTrainWork/tree/main) &nbsp;&nbsp;
![](https://img.shields.io/badge/language-python-blue.svg?style=plastic)

