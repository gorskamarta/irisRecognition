import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


class irisPlot:

    def printIrisPlot(self, teachingSet, teachingLabels):

        xData = []

        for iris in teachingSet:
            a = float(iris[0])
            b = float(iris[1])
            dIris = [a, b]
            xData.append(dIris)

        X = np.array(xData)

        teachingLabelsNumber = []
        for irisClass in teachingLabels:
            if irisClass == 'Iris-virginica':
                teachingLabelsNumber.append(1)
            if irisClass == 'Iris-versicolor':
                teachingLabelsNumber.append(2)
            if irisClass == 'Iris-setosa':
                teachingLabelsNumber.append(3)

        y = np.array(teachingLabelsNumber)


        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

        plt.figure(2, figsize=(8, 6))
        plt.clf()

        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
                    edgecolor='k')
        plt.xlabel('Długość kielicha')
        plt.ylabel('Szerokość kielicha')

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())

        fig = plt.figure(1, figsize=(8, 6))
        ax = Axes3D(fig, elev=-150, azim=110)
        X_reduced = PCA(n_components=3).fit_transform(np.array(teachingSet))
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
                   cmap=plt.cm.Set1, edgecolor='k', s=40)
        ax.set_xlabel("Pierwszy współczynnik")
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel("Drugi współczynnik")
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel("Trzeci współczynnik")
        ax.w_zaxis.set_ticklabels([])

        plt.show()