import random
import numpy as np
import matplotlib.pyplot as plt  # see http://matplotlib.org/users/pyplot_tutorial.html
import copy
from scipy import stats


def load_data_1b(fpath):
    data = []
    f = open(fpath, 'r')
    for line in f:
        words = line.split()
        data.append(words)
    f.close()
    arr = np.array(data, dtype=np.float64)
    return arr[:, 1:]  # what happened here?


if __name__ == "__main__":
    C2 = load_data_1b("C2.txt")

k = 4
length = len(C2)
new_center = [[] for i in range(k)]
cluster = [[] for i in range(k)]
d = [[] for i in range(k)]
p = [[] for i in range(k)]


def first_points(new_center, k):
    i = 0
    while (i < k):
        new_center[i] = C2[i]
        i = i + 1
    return new_center


def random_points(new_center, k):
    temp_center = random.sample(C2, k)
    for i in range(k):
        new_center[i] = temp_center[i]


def k_means(new_center, k):
    center = [0] * k
    counter = 0
    check = 0
    while (check == False):  # to examine whether the center has converged
        center = copy.copy(new_center)
        counter = counter + 1
        # empty the clusters
        print center
        print new_center
        for i in range(k):
            del cluster[i][:]
        for point in C2:
            index = 0
            min_r = np.linalg.norm(center[0] - point) ** 2
            for i in range(1, k):  # allocate the points to different cluster
                temp_r = np.linalg.norm(center[i] - point) ** 2
                if temp_r < min_r:
                    min_r = temp_r
                    index = i
            cluster[index].append(point)
        for i in range(k):
            new_center[i] = np.mean(cluster[i], axis=0)  # calculate the new centers
        check = np.allclose(center,new_center)  # (center == new_center).all() does not function & can not move up to while()
        print "iteration is %d" % counter
    return center


def k_means_plus_points(new_center, k):
    center = [0] * k
    j = 1
    count = 0
    index = 0
    center[0] = random.sample(C2, 1)[0]
    check = 0
    while j <= k:
        for point in C2:
            # here we get the j cluster
            for i in range(j):
                min_r = (np.linalg.norm(center[0] - point)) ** 2
                temp_r = (np.linalg.norm(center[i] - point)) ** 2
                if min_r > temp_r:
                    min_r = temp_r
                    index = i
                cluster[index].append(point)
        # here get the distance and probability
        if j <= k:
            distance(center[i], cluster[i], j)
        # here we get the center
        if j < k:
            k_plus_center(center, p, j)
        j = j + 1
    # update the center fot better solution
    new_center = copy.copy(center)
    k_means(new_center,k)
    plot_function(cluster,new_center)


def distance(center_i, cluster_i, j):
    print "J is %d" % j
    sum_r = 0
    for i in range(len(cluster_i)):
        d[j - 1].append((np.linalg.norm(center_i - cluster_i[i])) ** 2)
    sum_r = sum(d[j - 1])
    print "new sum is % d" % sum_r
    for i in range(len(cluster_i)):
        p[j - 1].append(d[j - 1][i] / sum_r)


def k_plus_center(center, p, j):
    prob = []
    point = []
    for i in range(j):
        prob.append(p[i])
        point.append(cluster[i])
    #  sum_p = np.sum(prob)
    # print "sum of probability is %d " % sum_p
    l = len(prob[0])
    distribution = stats.rv_discrete(values=(range(l), prob[0]))
    index = copy.copy(distribution.rvs(size=10))
    qq = index[0]
    center[j] = copy.copy(point[0][qq])


def compare(new_center, center):
    comp = [False] * k
    for i in range(k):
        temp_comp = np.linalg.norm(new_center[i] - center[i]) ** 2
        if temp_comp < 0.000001:
            comp[i] = True
    if (comp == [True] * k):
        return True
    else:
        return False


def gonzales(new_center, k):
    j = 1
    center = [0] * k
    center[0] = C2[0]
    while (j <= k):
        for i in range(k):
            del cluster[i][:]
        for point in C2:
            # here we get the j cluster
            min_r = abs(np.linalg.norm(center[0] - point))
            index = 0
            if j > 1:
                for i in range(j):
                    temp_r = abs(np.linalg.norm(new_center[i] - point))
                    if min_r > temp_r:
                        min_r = temp_r
                        index = i
                    cluster[index].append(point)
            else:
                cluster[index].append(point)
            # here we looking for new center
        if j < k:
            gonzales_center(center, cluster, j)
        # just print
        j = j + 1
   #update the center a little bit
    k_means(new_center,k)
    print j


def gonzales_center(center, cluster, j):
    max_r = 0
    check = [True]*j
    new_center[0] = center[0]
    for i in range(j):
        for point in cluster[i]:
            temp_r = abs(np.linalg.norm(new_center[i] - point))
            if temp_r > max_r:
                max_r = temp_r
                for i in range(j):
                    check[i] = np.allclose(new_center[i], point)
                if (check == [False]*j):
                    new_center[j] = point

def plot_function(cluster,new_center):
    x_listC = [x for [x, y] in C2]
    y_listC = [y for [x, y] in C2]
    plt.plot(x_listC, y_listC, 'ro')
    for i in range(k):
        x_list = [x for [x, y] in cluster[i]]
        y_list = [y for [x, y] in cluster[i]]
        plt.plot(x_list, y_list, 'o')
    for i in range(k):
        x_list = [x for [x, y] in new_center]
        y_list = [y for [x, y] in new_center]
        plt.plot(x_list, y_list, 'ys')
    # plt.axis([-40, 500, -45, 45])
    plt.show()

# first_points(new_center,k)
# random_points(new_center,k)
# k_means_plus_points(new_center,k)
# k_means(new_center, k)
gonzales(new_center, k)
print "my new center is :\n%s" %new_center

x_listC = [x for [x, y] in C2]
y_listC = [y for [x, y] in C2]
plt.plot(x_listC, y_listC, 'ro')
for i in range(k):
    x_list = [x for [x, y] in cluster[i]]
    y_list = [y for [x, y] in cluster[i]]
    plt.plot(x_list, y_list, 'o')
for i in range(k):
    x_list = [x for [x, y] in new_center]
    y_list = [y for [x, y] in new_center]
    plt.plot(x_list, y_list, 'ys')
# plt.axis([-40, 500, -45, 45])
plt.show()
