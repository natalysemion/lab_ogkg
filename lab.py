import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets


# Функція для визначення орієнтації трьох точок
def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # колінеарні
    elif val > 0:
        return 1  # по часовій стрілці
    else:
        return 2  # проти часової стрілки


# Функція для обчислення опуклої оболонки методом Грехема
def graham_scan(points):
    points_copy = points.copy()
    n = len(points_copy)
    if n < 3:
        return []

    # Знаходимо найнижчу точку (або найлівішу найнижчу точку)
    min_y = min(points_copy, key=lambda p: (p[1], p[0]))
    points_copy.remove(min_y)

    # Сортуємо точки за полярним кутом відносно найнижчої точки
    points_copy.sort(
        key=lambda p: (np.arctan2(p[1] - min_y[1], p[0] - min_y[0]), (p[0] - min_y[0]) ** 2 + (p[1] - min_y[1]) ** 2))

    # Стек для зберігання вершин опуклої оболонки
    hull = [min_y]

    for p in points_copy:
        while len(hull) > 1 and orientation(hull[-2], hull[-1], p) != 2:
            hull.pop()
        hull.append(p)

    return hull


# Функція для обчислення відстані від точки до лінії (відрізка)
def distance_point_to_segment(p, v, w):
    if np.array_equal(v, w):
        return np.linalg.norm(p - v)
    l2 = np.linalg.norm(w - v) ** 2
    t = max(0, min(1, np.dot(p - v, w - v) / l2))
    projection = v + t * (w - v)
    return np.linalg.norm(p - projection)


# Функція для обчислення мінімальної відстані від точки до всіх сторін многокутника
def min_distance_to_polygon(p, polygon):
    return min(distance_point_to_segment(p, polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon)))


# Функція для знаходження кола найбільшого радіусу, вписаного в опуклу оболонку
def max_inscribed_circle(hull_points, epsilon=1e-9):
    def ternary_search(f, left, right):
        while right - left > epsilon:
            left_third = left + (right - left) / 3
            right_third = right - (right - left) / 3
            if f(left_third) < f(right_third):
                left = left_third
            else:
                right = right_third
        return (left + right) / 2

    def radius_func(x, y):
        return min_distance_to_polygon(np.array([x, y]), hull_points)

    def find_best_y(x):
        return ternary_search(lambda y: radius_func(x, y), np.min(hull_points[:, 1]), np.max(hull_points[:, 1]))

    def find_best_x():
        return ternary_search(lambda x: radius_func(x, find_best_y(x)), np.min(hull_points[:, 0]),
                              np.max(hull_points[:, 0]))

    best_x = find_best_x()
    best_y = find_best_y(best_x)
    best_r = radius_func(best_x, best_y)

    return best_x, best_y, best_r


# Інтерфейс для взаємодії з користувачем
class ConvexHullApp:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.points = []
        self.hull_points = []
        self.circle_params = None
        self.initial_xlim = self.ax.get_xlim()
        self.initial_ylim = self.ax.get_ylim()

        # Налаштування інтерфейсу
        self.axclear = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.axrun = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.axnum = plt.axes([0.1, 0.05, 0.2, 0.075])

        self.bclear = widgets.Button(self.axclear, 'Очистити')
        self.bclear.on_clicked(self.clear)

        self.brun = widgets.Button(self.axrun, 'Запустити')
        self.brun.on_clicked(self.run_algorithm)

        self.text_box = widgets.TextBox(self.axnum, 'Кількість точок: ', initial="30")

        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

    def onclick(self, event):
        if event.inaxes == self.ax:
            self.points.append([event.xdata, event.ydata])
            self.ax.plot(event.xdata, event.ydata, 'bo')
            plt.draw()

    def clear(self, event):
        self.points = []
        self.hull_points = []
        self.circle_params = None
        self.ax.cla()
        self.ax.set_xlim(self.initial_xlim)
        self.ax.set_ylim(self.initial_ylim)
        plt.draw()

    def run_algorithm(self, event):
        n_points = int(self.text_box.text)
        if len(self.points) == 0:
            # Генерація випадкових точок
            self.points = np.random.rand(n_points, 2).tolist()
        self.hull_points = graham_scan(self.points)
        if len(self.hull_points) < 3:
            print("Недостатньо точок для побудови оболонки")
            return

        self.hull_points = np.array(self.hull_points)
        self.circle_params = max_inscribed_circle(self.hull_points)

        self.ax.cla()
        self.ax.plot(np.array(self.points)[:, 0], np.array(self.points)[:, 1], 'bo')
        self.ax.plot(np.append(self.hull_points[:, 0], self.hull_points[0, 0]),
                     np.append(self.hull_points[:, 1], self.hull_points[0, 1]), 'r--', lw=2)

        circle = plt.Circle((self.circle_params[0], self.circle_params[1]), self.circle_params[2], color='g',
                            fill=False)
        self.ax.add_artist(circle)

        # Забезпечуємо однаковий масштаб для осей тільки під час відображення кола
        self.ax.set_xlim(self.initial_xlim)
        self.ax.set_ylim(self.initial_ylim)
        self.ax.set_aspect('equal', 'box')

        plt.draw()


# Запуск додатку
app = ConvexHullApp()
