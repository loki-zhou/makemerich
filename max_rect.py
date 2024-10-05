import numpy as np
import cv2 as cv
import largestinteriorrectangle as lir

# 生成简单的凸多边形数据
def generate_polygon(num_points=10, scale=1.0):
    angles = np.sort(np.random.rand(num_points) * 2 * np.pi)
    points = np.stack([np.cos(angles), np.sin(angles)], axis=1) * scale
    return points.astype(np.int32)
polygonIput = np.array([[[20, 15], [210, 10], [220, 100], [100, 150], [20, 100]]], np.int32)


polygons = np.array([generate_polygon(num_points=5, scale=100.0) for _ in range(4)])
# print(f"{polygons[:1].shape } polygons.shape")
rectangle = lir.lir(polygons[:1])

print(rectangle)

cv.rectangle(img, lir.pt1(rectangle), lir.pt2(rectangle), (255, 0, 0), 1)

cv.imshow('lir', img)
cv.waitKey(0)
cv.destroyAllWindows()