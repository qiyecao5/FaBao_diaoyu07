import numpy as np
import torch
from QYC.config import *
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

''' 我要建一个类，然后有检测什么的各种方法，比如实时，比如只传进来一张返回结果的方法。'''
class QYC_yolov5():
    def __init__(self,model01,IMGSZ,CONF_THRES):
        '''

        Args:
            model01: 模型路径:'qyc_data/moxing/shcsr.pt'
            IMGSZ:  测试的图片尺寸,要32的倍数,这里默认480,我们使用的时候,用的不是这个
            CONF_THRES:置信度,源项目默认的是0.25,我们如果要准确建议0.5以上
        '''
        self.model = model01
        self.CONF_THRES = CONF_THRES
        # model获取模型, device设备GOU,
        # cuda显卡half是True, self.stride模型中提取步长（stride）信息,
        # self.names所有类别名字['npc', 'laohu'], self.imgsz检查图像大小[640, 640]
        self.model, self.device, self.half, self.stride, self.names, self.imgsz = self.get_model(IMGSZ = IMGSZ)


    # 加载模型 等初始化
    def get_model(self,half=False,IMGSZ = [480,480]):
        '''
        Args:
            half: device.type只有是cuda， half才是True,否则全是False。电脑会自动选择
            IMGSZ:默认[480,480]   你可以改
        Returns:返回5个参数
            model：模型
            device,
            half,
            stride,
            names
        我们不能每识别一张图就加载一次模型，那样太没效率了，所以我们把加载模型提取出来了
        '''
        # 选择设备 GPU 还是CPU
        device = select_device('')  # 会自动选择我们的GPU
        # print(device.type) # cuda
        # 如果当前显卡 不等于
        # device.type只有是cuda， half才是True,否则全是False
        half &= device.type != 'cpu'  # half precision only supported on CUDA

        # 加载模型
        model = attempt_load(self.model, map_location=device)
        # 模型中提取步长（stride）信息。
        stride = int(model.stride.max())  # model stride
        # 获取类名
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        # 如果我们是cuda显卡 就使用
        if half:
            # 将模型的参数和缓冲区的数据类型转换为半精度浮点数（也称为float16）。
            model.half()  # to FP16
            # 注意 原本是有分类器的，我们去掉了分类，可能我们需要分类器

        # 检查图像大小  check image size
        imgsz = check_img_size(IMGSZ, s=stride) # 验证图像大小是每个维度的步幅s的倍数
        return model, device, half, stride, names ,imgsz

    # 图像识别
    ''' 
    torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度。
    推理的不需要反向传播，所以不需要记录梯度，节省了内存，提高了推理速度'''
    @torch.no_grad()
    def pred_img(self,img0):
        '''
        Args:
            img0: 传入你的图像数组，正常我们截图那个数组
                img0 = cv2.imread(path)  # BGR
        Returns:  返回两个元素，参数1是你的图像，显示用的用不到。 参数2：列表类型，假如就一个目标应该是[0]
            im0 :画了框图像
            QYCdata：列表类型，检测到几个物体，就有几个列表，
                分别是【左上角x，左上角y,右下角x,右下角y,置信度，类别物体】没中心点，自己算，很简单
                [[589, 0, 640, 86, 0.3, 'npc'], [232, 160, 416, 339, 0.653, 'laohu']]
        '''
        # 用于将输入图像调整为指定的大小，并进行填充，以满足模型要求的尺寸和步幅的约束
        img = letterbox(img0, self.imgsz, stride = self.stride, auto=True)[0]

        # (高度, 宽度, 通道)转换为(通道, 高度, 宽度)，[::-1] 是对通道维度的翻转 这里需要RGB格式
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)  # 确保img是一个连续的数组

        # 运行推理 Run inference
        if self.device.type != 'cpu':  # 老老实实把这个if加上，不然你没有显卡的电脑怎么使用
            self.model(torch.zeros(1, 3, *self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        '''
        如果当前是cuda ,half就是True
        如果是True， img就等于img.half() 图像数据从32位浮点数（float32）转换为16位浮点数（float16）
        如果不是True,img就等于img.float()将图像数据从8位无符号整数（uint8）转换为32位浮点数（float32）
        '''
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32

        #  图像归一化，将图像数据从其原始的0-255范围转换到0.0-1.0范围后，根据需要添加批处理维度。
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # 增加了一个维度

        # 这行代码的意思是使用model对输入图像img进行推理，并将推理结果存储在变量pred中
        pred = self.model(img, augment=False, visualize=False)[0]
        # print('七叶草pred1:', pred, '---类型：%s 结束' % type(pred))

        '''
        非极大值抑制 NMS
        pred是模型预测的检测框及其相关属性 ,conf_thres置信度，iou_thres阈值去重，classes指定识别标签，agnostic_nms增强检测，max_det每个图像的最大检测数默认1000
        非极大值抑制是一种后处理步骤，用于消除重叠或冗余的检测框，从而提高检测的准确性。
        classes是需要检测的类别，默认是None 代表全检测，我们如果要灵活运行 比如只检测某个物体，那么就填某个类别
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, CLASSES, False, max_det=MAX_DET)
        # print('七叶草pred:', pred, '---类型：%s 结束' % type(pred))  # 这里就已经得到了所有数据了
        七叶草pred: [tensor([[ 44.17888,  52.87213,  98.92151, 115.51132,   0.72532,   0.00000]], device='cuda:0')] ---类型：<class 'list'> 结束
        '''
        pred = non_max_suppression(pred, self.CONF_THRES, IOU_THRES, CLASSES, False, max_det=MAX_DET)
        # print('七叶草pred:', pred, '---类型：%s 结束' % type(pred))  # 这里就已经得到了所有数据了

        det = pred[0]

        im0 = img0.copy()  # 拷贝原图
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化增益

        # 初始化画框画标注的对象
        annotator = Annotator(im0, line_width=LINE_THICKNESS, example=str(self.names))
        if len(det):
            # 具体来说，它将检测框从图像的原始尺寸（img_size）重新缩放到另一个图像的尺寸（im0的尺寸
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # QYCdata = det.tolist()
            # print('七叶草xyxy:', QYCdata, '---类型：%s 结束' % type(QYCdata))  # 这里就已经得到了所有数据了
            QYCdata =[]
            # 写的结果 Write results
            for *xyxy, conf, cls in reversed(det):
                # print(xyxy)  #[tensor(589.), tensor(0.), tensor(640.), tensor(86.)]
                int_list = [int(tensor.item()) for tensor in xyxy]
                # [589, 0, 640, 86]
                int_list.append(round(float(conf), 3))
                # [589, 0, 640, 86, 0.3]
                # print(int_list, type(int_list))

                '''
                注意这个c就是你找到的物体索引
                names是个列表，里面是你所有的标签名字。通过names[c]索引让你知道你找到的是谁'''
                c = int(cls)  # integer class
                int_list.append(self.names[c])
                # print(int_list, type(int_list))
                # [232, 160, 416, 339, 0.653, 'laohu']
                QYCdata.append(int_list)  # 这个就是我们要的

                if VIEW_IMG:  # 可以不显示，那就不用画了
                    label = None if HIDE_LABELS else (self.names[c] if HIDE_CONF else f'{self.names[c]} {conf:.2f}')
                    # laohu 0.65

                    # 这个函数annotator.box_label，成功画完边框 有标签的话也画完标签
                    annotator.box_label(xyxy, label, color=colors(c, True))
            '''
            这里就是显示结果了，我们假如是写脚本，这里都可以忽略，上面的代码我们早就拿到了坐标了
            返回带注释的图像作为数组
            annotator.result() 就是 np.asarray(self.im)  
            '''
            im0 = annotator.result()
            return im0, QYCdata
        else:
            return im0, False


# a=time.time()
# img0 = cv2.imread(IMAGE)  # BGR
# detect = QYC_yolov5()  # 加载模型
#
#
# img_result,list_result = detect.pred_img(img0)  # 检测物体
# if img_result is False:
#     print('没有找到任何东西')
#     cv2.imshow('input image', img_result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     # 如果没有识别到 这里是不可迭代对象
#     print('找到了',list_result)
#     cv2.imshow('input image', img_result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
