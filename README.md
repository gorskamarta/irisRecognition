# irisRecognition
Design for studies - Analysis and processing of data in Python

Klasyfikacji dokonujemy na trzy klasy: Iris setosa (kosaciec szczecinkowy), Iris versicolor (kosaciec różnobarwny) oraz Iris virginica.

Klasyfikujemy na podstawie czterech cech:

•	Długości kielicha w cm

•	Szerokości kielicha w cm

•	Długości płatka w cm

•	Szerokości płatka w cm

Źródło danych: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

Dane podlegają grupowaniu po określonej klasie, wybieranych z każdej klasy jest 80% danych jako dane uczące i 20% jako dane testowe.
Ostatecznie dane z trzech grup łączone są w zbiorcze grupy uczące i testowe i przy użyciu biblioteki sklearn są klasyfikowane.
Później zbiór testowy poddawany jest predykcji i porównywany z zapisanymi wynikami jakie były w pliku.
Ostatecznie określana jest skuteczność na podstawie ilości poprawnie sklasyfikowanych w stosunku do ilości wszystkich.
Przy założonym zbiorze danych poprawność rozpoznania wynosi 100%.
Program generuje również dwa wykresy obrazujące grupowanie irysów najpierw po dwóch później po trzech wartościach z czterech używanych do rozpoznawania.