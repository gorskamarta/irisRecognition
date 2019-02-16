import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import tree


data = []

# pobranie i oczyszczenie danych z strony internetowej
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
content = urllib.request.urlopen(url)
for line in content:
    strLine = str(line)
    strLine = strLine.replace("b\'", '')
    strLine = strLine.replace("\\n\'", '')
    iris = strLine.split(',')
    if (iris.__len__() == 5):
        data.append(iris)

# grupowanie po klasie
virginica_group = []
versicolor_group = []
setosa_group = []


for iris in data:
    if iris[4] == 'Iris-virginica':
        virginica_group.append(iris)
    if iris[4] == 'Iris-versicolor':
        versicolor_group.append(iris)
    if iris[4] == 'Iris-setosa':
        setosa_group.append(iris)

# obliczanie ilości danych w zbiorze uczącym (80/20)
quantitySetosaTeachSet = 0.8 * setosa_group.__len__()
quantityVersicolorTeachSet = 0.8 * versicolor_group.__len__()
quantityVirginicaTeachSet = 0.8 * virginica_group.__len__()


# podział danych dla grupy setosa
setosaTeachingSet = []
setosaTeachingLabels = []
setosaTestSet = []
setosaTestLabels = []

i = 0

for setosa in setosa_group:
    if i < quantitySetosaTeachSet:
        TeachSet = [setosa[0], setosa[1], setosa[2], setosa[3]]
        setosaTeachingSet.append(TeachSet)
        setosaTeachingLabels.append(setosa[4])

    else:
        TestSet = [setosa[0], setosa[1], setosa[2], setosa[3]]
        setosaTestSet.append(TestSet)
        setosaTestLabels.append(setosa[4])
    i = i + 1

# podział danych dla grupy virginica
virginicaTeachingSet = []
virginicaTeachingLabels = []
virginicaTestSet = []
virginicaTestLabels = []

i = 0

for virginica in virginica_group:
    if i < quantitySetosaTeachSet:
        TeachSet = [virginica[0], virginica[1], virginica[2], virginica[3]]
        virginicaTeachingSet.append(TeachSet)
        virginicaTeachingLabels.append(virginica[4])

    else:
        TestSet = [virginica[0], virginica[1], virginica[2], virginica[3]]
        virginicaTestSet.append(TestSet)
        virginicaTestLabels.append(virginica[4])
    i = i + 1

# podział danych dla grupy versicolor
versicolorTeachingSet = []
versicolorTeachingLabels = []
versicolorTestSet = []
versicolorTestLabels = []

i = 0

for versicolor in versicolor_group:
    if i < quantitySetosaTeachSet:
        TeachSet = [versicolor[0], versicolor[1], versicolor[2], versicolor[3]]
        versicolorTeachingSet.append(TeachSet)
        versicolorTeachingLabels.append(versicolor[4])

    else:
        TestSet = [versicolor[0], versicolor[1], versicolor[2], versicolor[3]]
        versicolorTestSet.append(TestSet)
        versicolorTestLabels.append(versicolor[4])
    i = i + 1

# tworzenie zbiorowego setu danych

teachingSet = versicolorTeachingSet + virginicaTeachingSet + setosaTeachingSet
teachingLabels = versicolorTeachingLabels + virginicaTeachingLabels + setosaTeachingLabels
testSet = versicolorTestSet + virginicaTestSet + setosaTestSet
# do porównania z wynikami uzyskanymi z wytrenowanego modelu
testLabels = versicolorTestLabels + virginicaTestLabels + setosaTestLabels

clf = tree.DecisionTreeClassifier()
clf.fit(teachingSet, teachingLabels)
result = clf.predict(testSet)


# ocena skuteczności
i = 0
success = 0
for x in result:
    if x == testLabels[i]:
        success = success + 1
    i = i + 1

effectiveness = success / i * 100
print("Skuteczność: " + str(effectiveness) + "%")

# ocena modelu
if effectiveness == 100:
    print('Model bezbłędny')
elif effectiveness > 80:
    print('Model skuteczny, ale nie bezbłędny')
elif effectiveness < 80:
    print('Model nieskuteczny')

#wykresy
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

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.xlabel('Długość kielicha')
plt.ylabel('Szerokość kielicha')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
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