# reference to:
# https://zhuanlan.zhihu.com/p/29393415
# https://zhuanlan.zhihu.com/p/101984674
# https://blog.csdn.net/fireflychh/article/details/83040205

import time
import json

class Coco(object):
    def __init__(self, *args, **kwargs):
        cur = time.localtime()
        self.info= { #数据集信息
            "year":kwargs.get('year') if kwargs.get('year') else int(cur.tm_year), # 年份
            "version":kwargs.get('version'), # 版本
            "description":kwargs.get('description'), # 数据集描述
            "contributor":kwargs.get('contributor'), # 提供者
            "url":kwargs.get('url'), # 下载地址
            "date_created":kwargs.get('date_created') if kwargs.get('date_created') else f'{cur.tm_year}/{cur.tm_mon:02d}/{cur.tm_mday:02d}', # 数据创建日期
        }
        self.licenses= []
        self.images= []
        self.annotations= []
        self.categories= [{"supercategory": "person",
                           "id": 0,
                           "name": "person",
                           "keypoints": ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
                                         'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                                         'right_knee', 'left_ankle', 'right_ankle'],
                           "skeleton": [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10],
                                        [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]],
                         }]

        self.json = {"info": self.info,
                     "licenses": self.licenses,
                     "images": self.images,
                     "annotations": self.annotations,
                     "categories": self.categories,
                     }
        self.licID=0
        self.imgID=0
        self.catID=1

    def __repr__(self):
        retVal = ''
        for jsonKey in self.json:
            if jsonKey=="info":
                retVal += '\n'.join([f'{key:21}: {self.info[key]}' for key in self.info])
            else:
                retVal += '\n'+ f'number of {jsonKey:11}: {len(self.json[jsonKey])}'
        return retVal
    
    #key: id被image中的license引用
    def addLic(self,id_=None,name=None,url=None):
        license_ = {
            "id":id_,
            "name":name,
            "url":url,
        }
        self.licenses.append(license_)

    #key: id被annotation中的image_id引用
    def addImg(self,id_=None, width=None, height=None, file_name=None, license_=None, flickr_url=None, coco_url=None, date_captured=None):
        id_ = self.imgID
        self.imgID+=1
        image = {
            "id":id_,
            "width":width,
            "height":height,
            "file_name":file_name,
            "license":license_,
            "flickr_url":flickr_url,
            "coco_url":coco_url,
            "date_captured":date_captured,
        }
        self.images.append(image)
        self.annID = 0
        return id_
    
    #key: id
    def addAnn(self,keypoints=None, num_keypoints=None, id_=None, image_id=None, category_id=None, segmentation=None, area=None, bbox=None, iscrowd=None):
        id_ = self.annID
        self.annID+=1
        annotation = {
            "keypoints": keypoints,
            "num_keypoints": num_keypoints,
            "id": id_, # annotation id
            "image_id": image_id, #refer to image.id
            "category_id": category_id,
            "segmentation": segmentation,
            "area": area,
            "bbox": bbox,
            "iscrowd": iscrowd,
        }
        self.annotations.append(annotation)
    
    #key: id被annotation中的category_id引用
    def addCat(self,supercategory='person', id_=None, name="person", keypoints=None, skeleton=None):
        id_ = self.catID
        self.catID+=1
        category = {
            "supercategory": supercategory,
            "id": id_,
            "name": name,
            "keypoints": keypoints,
            "skeleton": skeleton,
        }
        self.categories.append(category)
    
    def getCat(self, id_):
        for cat in self.categories:
            if cat["id"]==id_:
                return cat
        return False
    
    def saveToFile(self, path):
        with open(path, 'w') as f:
            json.dump(self.json, f)
            print(f'Coco file saved to {path} in the form of json.')

if __name__ == "__main__":
    print(Coco())