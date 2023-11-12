#import random
#import numpy as np
#import matplotlib.pyplot as plt
import pygame
import time
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
import time

class Point:
    def __init__(self, x, y, flag, group, isvisited, color):
        self.x = x
        self.y = y
        self.flag = flag
        self.group = group
        self.isvisited = isvisited
        self.color = color

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def scatter_plot(list):
  x = []
  y = []
  for i in list:
    x.append(i[0])
    y.append(i[1])

  plt.scatter(x,y)
  plt.show()


if __name__ == '__main__':
    points = []
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    screen.fill(color = '#FFFFFF')
    pygame.display.update()
    flag = True
    is_up = False
    c = 'red'
    ps = []
    start = time.time()
    while(flag):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                flag = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    is_up = True
                elif event.button == 3:
                    is_up = True
                    c = 'blue'
            if event.type == pygame.MOUSEBUTTONUP:
                is_up = False
            if (is_up):
                coord = event.pos
                points.append(Point(coord[0], -coord[1], 'black',0, False, c))
                pygame.draw.circle(screen, color = c, center=coord, radius=10)
                is_up = False
                time.sleep(0.2)
                is_up = True
            if event.type == pygame.KEYDOWN:
                if event.key == 13:
                    screen.fill(color = '#FF0000')
                pygame.display.update()
            end = time.time()
            if (end-start) > 10:
                pygame.display.update()
                x = []
                y = []
                for point in points:
                    new_arr = (point.x, point.y)
                    x.append(new_arr)
                    if point.color == 'red':
                        y.append(1)
                    elif point.color == 'blue':
                        y.append(2)

                X = np.array(x)
                Y = np.array(y)
                clf = svm.SVC(kernel='linear')
                clf.fit(X, Y)

                ax = plt.gca()
                DecisionBoundaryDisplay.from_estimator(
                    clf,
                    X,
                    cmap=plt.cm.Paired,
                    ax=ax,
                    response_method="predict",
                    plot_method="pcolormesh",
                    shading="auto",
                )

                plt.scatter(*zip(*X))
                plt.show()
                time.sleep(5)

                while(flag):
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            flag = False
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            if event.button == 1:
                                coord = event.pos
                                pygame.draw.circle(screen, color=c, center=coord, radius=10)
                                #pygame.display.update()
                                m = clf.predict([[coord[0], coord[1]]])
                                print(m)
                                if m[0] == 1:
                                    pygame.draw.circle(screen, color='red', center=coord, radius=10)
                                elif m[0] == 2:
                                    pygame.draw.circle(screen, color='blue', center=coord, radius=10)
                                pygame.display.update()
            pygame.display.update()







