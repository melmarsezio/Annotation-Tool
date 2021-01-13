import math
import cv2 as cv

def selectKey(y,x,KPs,height,width,threshDist=10):
    """select key point within threshDist.

    :param y: y coordinate of mouse click
    :param x: x coordinate of mouse click
    :param KPs: dictionary of KeyPoint: (coorY, coorX)
    :param height: image height
    :param width: image width
    :param threshDist: maximum distance to select any key point
    :rtype: str
    """
    retKey = ''
    for key in KPs:
        dist = math.sqrt((KPs[key][0]*height-y)**2+(KPs[key][1]*width-x)**2) # Euclidean distance
        if dist < threshDist:
            threshDist = dist
            retKey = key
    return retKey

def add(*tups):
    """element-wise addition on tuple.

    :param *tups: list of tuples
    :rtype: tuple
    """
    assert len({len(tup) for tup in tups})==1 # check all tuples have same length
    tup_len = len(tups[0])
    retVal = [0 for _ in range(tup_len)]
    for tup in tups:
        for i in range(tup_len):
            retVal[i] += tup[i]
    return tuple(retVal)

def mult(*tups):
    """element-wise multiplication on tuple.

    :param *tups: list of tuples
    :rtype: tuple
    """
    assert len({len(tup) for tup in tups})==1 # check all tuples have same length
    tup_len = len(tups[0])
    retVal = [1 for _ in range(tup_len)]
    for tup in tups:
        for i in range(tup_len):
            retVal[i] *= tup[i]
    return tuple(retVal)

def glob_move(deltaDict):
    """calculate the global movement of all 
    keypoints exclude out-of-bound keypoints.

    :param moveDict: list of tuples
    :rtype: tuple
    """
    count,v,u = 0,0,0
    for key in deltaDict:
        if deltaDict[key]:
            count+=1
            v+=deltaDict[key][0]
            u+=deltaDict[key][1]
    if v or u:
        return v/count,u/count
    else:
        return 0,0

def deltaKP(optFlow,kp,H,W,boundary):
    """get interpolated coordinates
    return deltaV,deltaU of keypoint.

    :param optFlow: RAFT result
    :param kp: keypoint
    :param H: Height of image
    :param W: Width of image
    :param boundary: view boundary
    :rtype: tuple
    """
    y,x = kp
    u = optFlow[:,:,0]
    v = optFlow[:,:,1]
    if 0<=x<=1 and boundary<=y<=1:
        y = y*(H-1)
        x = x*(W-1)
        lx, hx, ly, hy = math.floor(x), math.ceil(x), math.floor(y), math.ceil(y)
        vintp = (float(v[ly,lx]) + float(v[hy,lx]) + float(v[ly,hx]) + float(v[hy,hx]))/4
        uintp = (float(u[ly,lx]) + float(u[hy,lx]) + float(u[ly,hx]) + float(u[hy,hx]))/4
        return vintp/(H-1),uintp/(W-1)
    else:
        return None

def calcMove(optFlow,kps,H,W,CT=False):
    """calculate (deltaV,deltaU) of all key.

    :param optFlow: RAFT result
    :param kps: keypoints
    :param H: Height of image
    :param W: Width of image
    :param CT: CT mode on/off
    :rtype: dict()
    """
    boundary = 0.17 if CT else 0
    deltaDict = {key:deltaKP(optFlow,kps[key],H,W,boundary) for key in kps}
    globMove = glob_move(deltaDict)
    for key in deltaDict:
        if not deltaDict[key]:
            deltaDict[key]=globMove
    return deltaDict

def coorCvt(coor,rotDeg,opDeg):
    """Coordinate conversion (stored in decemal).

    :param coor: coordinate to be converted
    :param rotDeg: current rotated degree
    :param opDeg: degree when called Openpose Server (as normalised degree)
    :rtype: tuple
    """
    y,x = coor
    degree = (rotDeg-opDeg)%360
    if degree==90:
        x,y = 1-y,x
    elif degree==180:
        x,y = 1-x,1-y
    elif degree==270:
        x,y = y,1-x
    return y,x

def coorCvtInv(coor,dim,rotDeg,opDeg):
    """Inversion of 'coorCvt'.

    :param coor: coordinate to be converted
    :param dim: dimensions of
    :param rotDeg: current rotated degree
    :param opDeg: degree when called Openpose Server (as normalised degree)
    :rtype: tuple
    """
    y,x = coor
    height,width = dim
    degree = (rotDeg-opDeg)%360
    if degree==90:
        y,x = width-x,y
        height,width = width,height
    elif degree==180:
        y,x = height-y,width-x
    elif degree==270:
        y,x = x,height-y
        height,width = width,height
    return y,x,height,width

def cvtImg2QImage(frame, QImage):
    """convert image frame to QImage object.

    :param frame: image frame matrix
    :param QImage: QImage class from PyQt5.QtGui.QImage
    :rtype: QImage
    """
    qImg = QImage(frame.data.tobytes(),frame.shape[1], frame.shape[0], QImage.Format_RGB888)
    return qImg

def validIP(listOfQLineEdit):
    """check if IP is valid.

    :param listOfQLineEdit: list of QLineEdit with first 4
                        contains each 8 bits of IP address
                        and last 1 contains port number
    :rtype: Empty String if invalid else IP_str
    """
    # IPa,IPb,IPc,IPd,IPe = listOfQLineEdit
    IP_str = [QLineEdit.text() for QLineEdit in listOfQLineEdit]
    ip = IP_str[:4]
    port = IP_str[-1]

    for i in ip:
        if not i.isdigit():
            return ''
        if not (0 <=int(i)<=255):
            return ''
    if not port.isdigit():
        return ''
    if not (0 <=int(port)<=65535):
        return ''
    return '.'.join(ip)+':'+port

def splitIP(IP_str):
    """convert IP_str into IP_tuple.

    :param IP_str: IP in string form
    :rtype: tuple
    """
    ip, port = IP_str.split(':')
    ip = ip.split('.')
    ip.append(port)
    return ip

def fillIP(IP_str, listOfQLineEdit):
    """fill IP_str into QLineEdits.

    :param IP_str: IP in string form
    :param listOfQLineEdit: list of QLineEdit
    :rtype: NoneType
    """
    ip = splitIP(IP_str)
    for i in range(len(ip)):
        listOfQLineEdit[i].setText(ip[i])

def checkRAFT(video, *RAFT):
    l,h,w,c = video.shape
    videoShape = l-1,w,h,c
    for raft in RAFT:
        if videoShape[:3] != raft.shape[:3]:
            return False
    return True
    