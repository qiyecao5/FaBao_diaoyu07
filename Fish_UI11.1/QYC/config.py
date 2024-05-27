开yolov5 = False
# IMAGE = 'qyc_data/images/a.jpg'
# 把所有的常量都放在这里
#模型
MODEL_DICT={
    "圣山传送人":'qyc_data/moxing/shcsr.pt',
    "卷轴":'qyc_data/moxing/juanzhou.pt',
}


# IMGSZ = [480,480]
# CONF_THRES = 0.25  # 置信度
IOU_THRES = 0.6  # 用来去除差不多概率的锚框,非最大值抑制

# classes是需要检测的类别，默认是None 代表全检测，我们如果要灵活运行 比如只检测某个物体，那么就填某个类别
CLASSES = None
# CLASSES = [1]  # 列表类型

# 每张图最多检测多少次，如果嫌慢可调小
MAX_DET = 1000     # 用于限制返回的最大检测数量。如果未设置或设置为None，则返回所有满足条件的检测框。
LINE_THICKNESS = 1   # 线宽


# 显示标注图像，假如设置成False,我们就不用计算画框了。不然你每次都给你画框，虽然也很快，不用画就不画不是更快  50次没发现区别
VIEW_IMG = True
# 隐藏标签 隐藏置信度，我们最终只要一个方框
HIDE_CONF = False    # 隐藏置信度
HIDE_LABELS = False  # 隐藏标签  (隐藏的画)

'''
VIEW_IMG 这是总显示，False的话，不会显示标签和框
'''

