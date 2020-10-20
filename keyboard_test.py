# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 10:52:21 2020

@author: 11037
"""

def key_down(key):
  """
  函数功能：按下按键
  参    数：key:按键值
  """
    key = key.upper()
    vk_code = key_map[key]
    win32api.keybd_event(vk_code, win32api.MapVirtualKey(vk_code, 0), 0, 0)


def key_up(key):
    """
  函数功能：抬起按键
  参    数：key:按键值
  """
    key = key.upper()
    vk_code = key_map[key]
    win32api.keybd_event(vk_code, win32api.MapVirtualKey(vk_code, 0), win32con.KEYEVENTF_KEYUP, 0)


def key_press(key):
    """
  函数功能：点击按键（按下并抬起）
  参    数：key:按键值
  """
    key_down(key)
    time.sleep(0.02)
    key_up(key)

# 输入 a
key_press("a")
