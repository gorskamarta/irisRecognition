import csv
import pydotplus
import collections
import matplotlib.pyplot as plt
from sklearn import tree


data = []

# import danych z pliku
with open('C:\DEV\_Python\irisRecognition\iris.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        data.append(row)

#grupowanie po klasie
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

#obliczanie ilości danych w zbiorze uczącym (80/20)
quantitySetosaTeachSet = 0.8 * setosa_group.__len__()
quantityVersicolorTeachSet = 0.8 * versicolor_group.__len__()
quantityVirginicaTeachSet = 0.8 * virginica_group.__len__()


#podział danych dla grupy setosa
setosaTeachingSet = []
setosaTeachingLabels = []
setosaTestSet = []
setosaTestLabels = []

i=0

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

#podział danych dla grupy virginica
virginicaTeachingSet = []
virginicaTeachingLabels = []
virginicaTestSet = []
virginicaTestLabels = []

i=0

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

#podział danych dla grupy versicolor
versicolorTeachingSet = []
versicolorTeachingLabels = []
versicolorTestSet = []
versicolorTestLabels = []

i=0

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

#tworzenie zbiorowego setu danych

teachingSet = versicolorTeachingSet + virginicaTeachingSet + setosaTeachingSet
teachingLabels = versicolorTeachingLabels + virginicaTeachingLabels + setosaTeachingLabels
testSet = versicolorTestSet + virginicaTestSet + setosaTestSet
#do porównania z wynikami uzyskanymi z wytrenowanego modelu
testLabels = versicolorTestLabels + virginicaTestLabels + setosaTestLabels

clf = tree.DecisionTreeClassifier()
clf.fit(teachingSet, teachingLabels)
result = clf.predict(testSet)

#print("Wyniki")
#print(result)
#print("Oczekiwane wyniki")
#print(testLabels)

i = 0
success = 0
for x in result:
    if x == testLabels[i]:
        success = success + 1
    i = i + 1

print("Skuteczność: " + str(success / i * 100) + "%")