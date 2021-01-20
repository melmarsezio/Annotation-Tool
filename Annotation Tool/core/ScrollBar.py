import sys
import numpy as np
import pyqtgraph as pg

from PyQt5 import QtGui, QtCore


class CustomViewBox(pg.ViewBox):
    def __init__(self, vertical, *args, **kwds):
        pg.ViewBox.__init__(self, *args, **kwds)
        self.vertical = vertical
        self.cursor = None

    def addItem(self, item, xMax=None, yMax=None, minRange=None, ignoreBounds=False):
        pg.ViewBox.addItem(self,item, ignoreBounds)
        if isinstance(item, pg.graphicsItems.PlotCurveItem.PlotCurveItem):
            if xMax:
                self.setLimits(xMin=0,xMax=xMax)
            if yMax:
                self.setLimits(yMin=0,yMax=yMax)
            if minRange:
                if self.vertical:
                    self.setLimits(minYRange=minRange)
                else:
                    self.setLimits(minXRange=minRange)
        elif isinstance(item, pg.graphicsItems.InfiniteLine.InfiniteLine):
            self.cursor = item

    def wheelEvent(self, ev, axis=None):
        if self.vertical:
            self.initXRange = self.viewRange()[0]
            pg.ViewBox.wheelEvent(self, ev, axis)
            self.setXRange(self.initXRange[0],self.initXRange[1],padding=0)
        else:
            self.initYRange = self.viewRange()[1]
            pg.ViewBox.wheelEvent(self,ev,axis)
            self.setYRange(self.initYRange[0],self.initYRange[1],padding=0)

    def mouseDragEvent(self, ev, axis=None):
        # Disable Right Button Drag action
        if ev.button() & QtCore.Qt.RightButton:
            return
        pg.ViewBox.mouseDragEvent(self, ev, axis)
    
    def mouseClickEvent(self, ev):
        # Left Click to jump to frame directly
        if ev.button() == QtCore.Qt.LeftButton and self.cursor:
            x = self.mapSceneToView(ev.pos()).x()
            self.cursor.setValue(int(np.round(x)))
        
    def reset(self,invert):
        self.emptyAllItems()
        self.invertX(b=invert)
        self.vertical = invert
    
    def emptyAllItems(self):
        while len(self.addedItems)>=2:
            for item in self.addedItems:
                if not isinstance(item, Cursor):
                    self.removeItem(item)

class Region(pg.LinearRegionItem):
    def __init__(self, values=(0, 1), orientation='vertical', brush=None, pen=None,
                 hoverBrush=None, hoverPen=None, movable=True, bounds=None, 
                 span=(0, 1), swapMode='sort'):
        pg.GraphicsObject.__init__(self)
        self.orientation = orientation
        self.bounds = QtCore.QRectF()
        self.blockLineSignal = False
        self.moving = False
        self.mouseHovering = False
        self.span = span
        self.swapMode = swapMode
        self._bounds = None

        lineKwds = dict(
            movable=movable,
            bounds=bounds,
            span=span,
            pen=pen,
            hoverPen=hoverPen,
            )
            
        if orientation in ('horizontal', pg.LinearRegionItem.Horizontal):
            self.lines = [
                Cursor(pos=values[0], angle=0, label='',labelOpts={'color':'r','position':0.8}, **lineKwds), 
                Cursor(pos=values[1], angle=0, label='',labelOpts={'color':'r','position':0.8}, **lineKwds)]
            self.lines[0].scale(1, -1)
            self.lines[1].scale(1, -1)
        elif orientation in ('vertical', pg.LinearRegionItem.Vertical):
            self.lines = [
                Cursor(pos=values[0], angle=90, label='',labelOpts={'color':'r','position':0.8}, **lineKwds), 
                Cursor(pos=values[1], angle=90, label='',labelOpts={'color':'r','position':0.8}, **lineKwds)]
        else:
            raise Exception("Orientation must be 'vertical' or 'horizontal'.")
        
        for l in self.lines:
            l.setParentItem(self)
            l.sigPositionChangeFinished.connect(self.lineMoveFinished)
        self.lines[0].sigPositionChanged.connect(lambda: self.lineMoved(0))
        self.lines[1].sigPositionChanged.connect(lambda: self.lineMoved(1))

        if brush is None:
            brush = QtGui.QBrush(QtGui.QColor(0, 0, 255, 50))
        self.setBrush(brush)
        
        if hoverBrush is None:
            c = self.brush.color()
            c.setAlpha(min(c.alpha() * 2, 255))
            hoverBrush = pg.mkBrush(c)
        self.setHoverBrush(hoverBrush)
        
        self.setMovable(movable)
        
    def mouseClickEvent(self, ev):
        #Right Click
        if ev.button() == QtCore.Qt.RightButton:
            vb = self.parentWidget()
            vb.removeItem(self)
            ev.accept()
    
    def setMouseHover(self, hover):
        ## Inform the item that the mouse is(not) hovering over it
        if self.mouseHovering == hover:
            return
        self.mouseHovering = hover
        if hover:
            self.currentBrush = self.hoverBrush
            for l in self.lines:
                l.setMouseHover(True)
        else:
            self.currentBrush = self.brush
            for l in self.lines:
                l.setMouseHover(False)
        self.update()
        
class Cursor(pg.InfiniteLine):
    def __init__(self, *args, **kwds):
        pg.InfiniteLine.__init__(self, *args, **kwds)
    
    def setPos(self, v):
        v = np.round(v)
        pg.InfiniteLine.setPos(self, v)
        if hasattr(self, 'label'):
            self.label.setFormat(str(int(self.value())))
    
    def setMouseHover(self, hover):
        ## Inform the item that the mouse is (not) hovering over it
        if self.mouseHovering == hover:
            return
        self.mouseHovering = hover
        if hover:
            self.currentPen = self.hoverPen
            self.label.setFormat(str(int(self.value())))
        else:
            self.currentPen = self.pen
            self.label.setFormat('')
        self.update()

class ScrollBar(pg.PlotItem):
    def __init__(self):
        self.vb = CustomViewBox(vertical = False)
        pg.PlotItem.__init__(self, viewBox=self.vb)
        self.hideAxis('left')
        self.hideAxis('top')
        pg.setConfigOptions(antialias=True,background=(240,240,240,255),foreground='k')
        self.x = None
        self.y = None
        self.il = Cursor(pos=0,angle=90,pen=pg.mkPen('k',width=2),
                                  hoverPen=pg.mkPen('k',width=4),
                                  movable=True,bounds=[0,0],label='',labelOpts={'color':'k','position':0.8})
        self.vb.addItem(self.il)
        self.valueChanged = self.il.sigPositionChanged
        
    def newPlot(self, y, rotDeg=0):
        length = len(y)
        self.max = length
        self.x = np.array(range(1,length+1))
        self.y = y
        
        if (rotDeg%180): # rotateDegree=90/270
            self.vb.reset(invert=True)
            self.x,self.y = self.y,self.x
            self.hideAxis('bottom')
            self.showAxis('right')
            angle=0
        else: # rotateDegree = 0/180
            self.vb.reset(invert=False)
            self.showAxis('bottom')
            self.hideAxis('right')
            angle = 90

        plot = pg.PlotCurveItem(x=self.x,y=self.y,pen=pg.mkPen('b'))
        self.vb.addItem(plot,xMax=max(self.x)+1,yMax=max(self.y)+1,minRange=5)
        self.autoRange()

    def Pos(self):
        return int(self.il.value())
    
    def setPos(self, pos):
        self.il.setValue(pos)
    
    def setMaximum(self, max_):
        y = np.zeros((max_,))
        self.newPlot(y)
        self.il.setBounds([1,max_])
    
    def addHighLight(self):
        if self.y is None:
            return
        length = len(self.y)
        lr = Region(values=self.bound(self.Pos()-10,self.Pos()+10),
                                 pen=pg.mkPen('r'),hoverPen=pg.mkPen('r',width=4),
                                 brush=pg.mkBrush((255,0,0,80)),bounds=[1,length],
                                 swapMode='sort')
        self.vb.addItem(lr)
        # print(self.getHightLight())

    def bound(self,left,right):
        return max(1,left),min(right,self.max)
    
    def getHightLight(self):
        allHighRegion = []
        for item in self.vb.addedItems:
            if isinstance(item, Region):
                region = item.getRegion()
                allHighRegion.append([int(region[0]-1),int(region[1]-1)])
        return allHighRegion

if __name__=="__main__":    
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())

