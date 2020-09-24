#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:34:12 2020

@author: Robert_Hennings
"""
import pandas as pd

#SVM Algorithmus 
#Das DataSet reinladen
from sklearn import datasets
# Das Cancer DataSet importieren geht so:
cancer = datasets.load_breast_cancer()
#Cancer DataSet hat 13 Features bzw. Attribute und zwei Targets also Klassen: guter Krebs und schlehter Krebs

#Daten sollen nun veranschaulicht werden
#Alle 13 Features (Attribute) sollen angezeigt werden

print('Features:', cancer.feature_names)
#Ergebnis sind alle 13 Features des DataSets
#Zur Vollständigkeit die Labels bzw Targets ausgeben:
print('Labels:', cancer.target_names)

#Shape des DataSets:
cancer.data.shape
#Top 5 Records of the Cancer
print(cancer.data[0:5])
df = (cancer.data[0:5])
#Das Target Set ansehen:
print(cancer.target)
#Dies soll auch nochmal wie vorher als eigener Dataframe aufgefasst werden


#DAs Dataset soll nun in Trainingsdaten und in Testsdaten aufgespalten werden
#Funktion: train_test_split(), es müssen drei Angaben gemacht werden: 
# 1) features
# 2) target
# 3) test_set size
# Man nutzt random_state um die records zufällig auszuwählen

#Erst die Funktion importieren:

from sklearn.model_selection import train_test_split

#DataSet aufsplitten:
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) # 70% training and 30% test
#Nun soll das SVM Model importiert werden, creating the support vector classifier
#Danach wird das Model auf die Trainingsdaten gefittet und soll eine prediction ausführen

from sklearn import svm
#SVM Classifier erstellen
clf = svm.SVC(kernel='linear')  #Lineare Kernelfunktion
#Model wird trainiert auf Training Set
clf.fit(X_train, y_train)

#Als Test mal etwas predicten:
y_pred = clf.predict(X_test)

#Genauigkeit der Vorhersage soll beurteilt werden indem die vorhergesagten Werte mit den tatsächlichen verglichen werden:
#Metrics modul aus sklearn
from sklearn import metrics
#Genauigkeit beurteilen
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
