#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:18:26 2020

@author: Robert_Hennings
"""


import pandas as pd
import numpy as np
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
df = pd.DataFrame(data=cancer.data)
 #Im Dataframe sind nun alle Daten also zu jedem der 30 Attribute die einzelnen Werte als einzelne Reihen
dff = cancer.feature_names

#Nun sollen die einzelnen Attribute als Zeilen vorliegend als einzelne Spaltennamen auftreten!

df.rename(columns={0: dff[0], 1: dff[1]}, inplace=True)

for i in range (30):
    
    df.rename(columns={i: dff[i], i: dff[i]}, inplace=True)
    i+1

#Nun soll noch eine weitere Spalte hinzugefügt werden mit den 0en und 1en die für die schlechte und gute Ausprägung des Krebs stehen also als Ergebnis so gesehen

dft = cancer.target

df['Ergebnis'] = dft


df.loc[df['Ergebnis'] ==0, 'Befund'] = 'malignant'
df.loc[df['Ergebnis'] >0, 'Befund'] = 'benign'

    
    