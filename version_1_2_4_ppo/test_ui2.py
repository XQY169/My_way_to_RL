import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsItem
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QTimer
import random

class GraphView(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Real-time Graph Example')
        self.setGeometry(100, 100, 600, 400)

        # 创建一个图形场景
        self.scene = QGraphicsScene()
        self.addAxis(self.scene)
        # 创建一个图形视图并设置场景
        self.view = QGraphicsView(self.scene)
        self.setCentralWidget(self.view)

        # 创建一个定时器，每隔一段时间添加一个数据点
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.addRandomPoint)
        self.timer.start(1000)  # 每隔1秒触发一次定时器

    def addRandomPoint(self):
        # 随机生成一个数据点的坐标
        x = random.randint(50, 550)
        y = random.randint(50, 350)

        # 绘制数据点并添加到场景中
        point_item = self.scene.addEllipse(x, y, 5, 5, QPen(Qt.blue), Qt.blue)
        point_item.setFlag(QGraphicsItem.ItemIsSelectable, True)  # 允许选中数据点
        print([x,y])

    def addAxis(self, scene):
        # 创建水平坐标轴线
        scene.addLine(50, 350, 550, 350, QPen(Qt.black))

        # 创建垂直坐标轴线
        scene.addLine(50, 50, 50, 350, QPen(Qt.black))

        # 添加水平坐标刻度
        for i in range(1, 11):
            x = 50 + i * 50
            scene.addLine(x, 350, x, 345, QPen(Qt.black))

            # 添加刻度标签
            label = scene.addText(str(i * 10))
            label.setPos(x - 5, 355)

        # 添加垂直坐标刻度
        for i in range(1, 7):
            y = 350 - i * 50
            scene.addLine(45, y, 50, y, QPen(Qt.black))

            # 添加刻度标签
            label = scene.addText(str(i * 10))
            label.setPos(30, y - 5)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GraphView()
    window.show()
    sys.exit(app.exec_())