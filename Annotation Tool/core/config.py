mapKP2ID = {'nose':0, 'sternum':1, 'Rshoulder':2, 'Relbow':3, 'Rwrist':4, 'Rhip':5,
            'Rknee':6, 'Rankle':7, 'Reye':8, 'Rear':9, 'Lshoulder':10, 'Lelbow':11,
            'Lwrist':12, 'Lhip':13, 'Lknee':14, 'Lankle':15, 'Leye':16, 'Lear':17}
            
skeleton = [(0,8),(8,9),(0,16),(16,17),(0,1),(1,2),(2,3),(3,4),(1,10),(10,11),(11,12),(1,5),(5,6),(6,7),(1,13),(13,14),(14,15)]

mapID2KP = {mapKP2ID[key]:key for key in mapKP2ID}

COCO_DATASET_KPTS=['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
                    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                    'right_knee', 'left_ankle', 'right_ankle']

def trans(kp):
        kp = kp.split('_')
        if len(kp)>1:
            return kp[0][0].upper()+kp[1]
        else:
            return kp[0]

mapCOCO2KPs = {kp: trans(kp) for kp in COCO_DATASET_KPTS}
