#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""获取数据"""

__author__ = 'yp'


import requests
import time
import random

# with open("huoliquankai.txt", encoding="utf-8", mode="a") as f1:
#     for i in range(20000):
#         try:
#             r = requests.get('https://nmsl.shadiao.app/api.php?lang=zh_cn', timeout=(2, 4))
#             if r.status_code == 200:
#                 if r.text[:5] == "访问太频繁":
#                     time.sleep(60)
#                 else:
#                     f1.writelines("{}\n".format(r.text))
#             else:
#                 print(r.status_code)
#             time.sleep(2 * random.random())
#         except Exception as e:
#             print("error:{}".format(e))
#             time.sleep(60)


with open("koutulianhua.txt", encoding="utf-8", mode="a") as f1:
    counter = 0
    for i in range(20000):
        try:
            r = requests.get('https://nmsl.shadiao.app/api.php?level=min&lang=zh_cn', timeout=(2, 4))
            if r.status_code == 200:
                if r.text[:5] == "访问太频繁":
                    print(r.text)
                    time.sleep(60)
                else:
                    f1.writelines("{}\n".format(r.text))
            else:
                print(r.status_code)
            time.sleep(2 * random.random())
        except Exception as e:
            print("error:{}".format(e))
            time.sleep(60)
