# from os import listdir, getcwd
# from os.path import join
import cv2
import os
from xml.etree.ElementTree import parse
import xml.etree.ElementTree as ET
# 不需要提前新建txt
 
# xmlfilepath = '/mnt/disk2/yuanzm/DIOR/Annotations/Oriented Bounding Boxes'  # 存放xml文件的文件夹路径
# total_xml = os.listdir(xmlfilepath)  # 遍历xml文件
# classes = set()
# for xml in total_xml:
#     xml_path = os.path.join(xmlfilepath, xml)  # xml的全路径，也就是文件的路径
#     # print(xml_path)
#     # xml_file = open(xml_path,encoding='gbk')
#     # root = ET.parse(xml_file).getroot()

#     tree = parse(xml_path)  # 获取ElementTree
#     root = tree.getroot()  # 获取根元素
 
#     filename = root.findtext('filename')[0:5]  # 提取到了xml对应的filename的名字，[0:5]是我只想要名字，不想要后缀.jpg
 
#     for obj in root.iter('object'):  # 如有很多object ，需要遍历object
#         cls = obj.find('name').text  # 将解析得到该keypoints下name中的标签
#         classes.add(cls)
#     if len(classes) == 20:
#         break
#         # if cls == 'bridge':  # 选择自己需要的标签类（如桥bridge），然后就可以把这类别下需要的坐标等信息输出了
 
#         #     '''xmlbox = obj.find('bndbox')  # 坐标信息
#         #     b = ((xmlbox.find('xmin').text), (xmlbox.find('ymin').text),
#         #          (xmlbox.find('xmax').text), (xmlbox.find('ymax').text))'''
#         #     with open('F:\\DIOR数据集\\2.txt', 'a') as f:  # txt存储路径
#         #         f.write(str(filename))
#         #         # print("b:", b[0])
#         #         '''for i in range(len(b)):
#         #             f.write(' ')
#         #             f.write(str(b[i]))'''
#         #         f.write('\n')
#         # break  # 因为我只要名称，所以不需要遍历object，只要有bridge就跳出最深层的for循环
# print(classes)

a = ('dam', 'stadium', 'bridge', 'Expressway-Service-area', 'Expressway-toll-station', 'chimney', 'trainstation', 'harbor', 'ship', 'groundtrackfield', 'airplane', 'windmill', 'tenniscourt', 'baseballfield', 'golffield', 'basketballcourt', 'vehicle', 'storagetank', 'overpass', 'airport')
print(len(a))