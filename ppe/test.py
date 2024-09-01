import pygetwindow as gw
from PIL import ImageGrab

import win32gui
import win32ui
import win32con
from PIL import Image
import time

for prc in gw.getAllWindows():
    print(prc)


def get_window_handle(window_name):
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd:
        print(f"Window handle for '{window_name}' is: {hwnd}")
    else:
        print(f"Window '{window_name}' not found.")
    return hwnd


def screenshot_window(hwnd):
    # 获取窗口的设备上下文
    window_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(window_dc)
    save_dc = mfc_dc.CreateCompatibleDC()

    # 获取窗口的宽度和高度
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top

    # 创建位图对象
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
    save_dc.SelectObject(bitmap)

    # 将窗口内容复制到位图对象
    save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)

    # 将位图转为图像数据
    bmpinfo = bitmap.GetInfo()
    bmpstr = bitmap.GetBitmapBits(True)

    img = Image.frombuffer(
        'RGB',
        (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
        bmpstr, 'raw', 'BGRX', 0, 1
    )

    # 释放资源
    win32gui.DeleteObject(bitmap.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, window_dc)

    return img


def save_screenshot(img, save_path):
    img.save(save_path)


if __name__ == '__main__':
    window_name = "虎牙直播"  # 这里替换为目标窗口的名称
    hwnd = get_window_handle(window_name)

    if hwnd:
        while True:
            img = screenshot_window(hwnd)
            save_path = "screenshot.png"  # 保存截图的位置
            save_screenshot(img, save_path)
            print(f"Screenshot saved to {save_path}")

            time.sleep(1)  # 每隔 1 秒截图一次，调整频率以满足需求
    else:
        print("Failed to find the window handle.")
