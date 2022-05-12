#%matplotlib inline
import math as mth
import numpy as np
import collections
import sys
import random
#import pandas as pd
#import matplotlib.pyplot as plt

lista = []
slownik_SredniaWspolrzednychPunktow = {}
slownik_SredniaWspolrzednychPunktow[0.0] = []
slownik_SredniaWspolrzednychPunktow[1.0] = []
plik = open("australian.dat", "r+", encoding = 'utf-8')
for line in plik:
    tmp = line.split()
    tmp = list(map(lambda e: float(e), tmp))
    lista.append(tmp)

plik.close()
def Euklides(lista1, lista2):
    wynik = 0
    for i in range(len(lista[0]) - 1):
        wynik += (lista1[i]-lista2[i])**2
    return wynik

def Euklides2(lista1, lista2):
    wynik = 0
    for i in range(len(lista[0]) - 1):
        v1 = np.array(lista1[i])
        v2 = np.array(lista2[i])
        wynik += (v1-v2)**2
    return wynik

def Odleglosc_do_listy(lista1, lista2):
    return Euklides2(lista1, lista2)

def Klucze_wartosci_do_slownika(slownik_odleglosc, lista, nr_listy, odleglosc):
    if lista[nr_listy][len(lista[nr_listy])-1] in slownik_odleglosc:
        slownik_odleglosc[lista[nr_listy][len(lista[nr_listy])-1]].append([nr_listy, odleglosc])
    else:
        slownik_odleglosc[lista[nr_listy][len(lista[nr_listy])-1]] = []
        slownik_odleglosc[lista[nr_listy][len(lista[nr_listy]) - 1]].append([nr_listy, odleglosc])
    return slownik_odleglosc

def JedenKontraReszta(lista):
    wynik = 0
    slownik_odleglosc = {}
    for i in range(len(lista)):
        for j in range(len(lista)):
            if lista[i] != lista[j]:
                wynik += Odleglosc_do_listy(lista[i], lista[j])
        Klucze_wartosci_do_slownika(slownik_odleglosc, lista, i, wynik)
        wynik = 0
    return slownik_odleglosc

def JedenKontraResztaKoloru(lista):
    wynik=0
    slownik_odleglosc = {}
    for i in range(len(lista)):
        for j in range(len(lista)):
            if lista[i][len(lista[i])-1] == lista[j][len(lista[j])-1] and lista[i] != lista[j]:
                wynik += Odleglosc_do_listy(lista[i], lista[j])
        Klucze_wartosci_do_slownika(slownik_odleglosc, lista, i, wynik)
        wynik = 0
    return slownik_odleglosc

def SortujSlownik(slownik):
    i = 0
    slownik = collections.OrderedDict(sorted(slownik.items()))
    for item in slownik.values():
        key = list(slownik.keys())[i]
        slownik[key] = sorted(item, key=lambda x: x[1], reverse=False)
        print(key,":",slownik[key])
        i += 1
    return slownik
        #slownik_sort = sorted(item, key=lambda x: x[1], reverse=True)
    #slownik_sort = sorted(slownik.items(), key=lambda x: x[1], reverse=False)
    #return slownik_sort

def SredniaWspolrzednychPunktow(lista):
    i=0
    suma0=0
    suma1=0
    srednia0=0
    srednia1=0
    i_sum0=0
    i_sum1=0
    while i != len(lista[i])-1:
        for j in range(len(lista)):
            if lista[j][len(lista[j]) - 1] == 0.0:
                suma0 += lista[j][i]
                i_sum0+=1
            if lista[j][len(lista[j]) - 1] == 1.0:
                suma1 += lista[j][i]
                i_sum1+=1
        srednia0 = suma0/i_sum0
        srednia1 = suma1/i_sum1
        if 0.0 in slownik_SredniaWspolrzednychPunktow.keys():
            slownik_SredniaWspolrzednychPunktow[0.0].append(srednia0)
        if 1.0 in slownik_SredniaWspolrzednychPunktow.keys():
            slownik_SredniaWspolrzednychPunktow[1.0].append(srednia1)
        suma0=0
        suma1=0
        srednia0=0
        srednia1=0
        i_sum0=0
        i_sum1=0
        i+=1
    return slownik_SredniaWspolrzednychPunktow

def waga_i_zamianaKoloru(slownik, slownik_koloru):
    #iteracja=0
    nr_obiektu = 0
    slownik_wag = {}
    min = slownik[0][10][1]
    for kolor, item_kolor in slownik_koloru.items():
        slownik_wag[item_kolor[0][0]] = []
        slownik_wag[item_kolor[0][0]].append(item_kolor[0][1])
        slownik_wag[item_kolor[0][0]].append(kolor)
        #for i in range(len(item)):
        #    suma += item[i][1]
        #print("suma:")
        #print(suma/len(item))
        #print(list(slownik.keys())[iteracja])
        #slownik_wag[list(slownik.keys())[iteracja]] = suma/len(item)
        #iteracja+=1
    for item in slownik.values():
        #suma=0.
        #print(item[0][1])
        if item[0][1] < min:
            min = item[0][1]
            nr_obiektu = item[0][0]
    slownik_wag[nr_obiektu] = min
    return slownik_wag

def Zmien_kolor(slownik_waga_kolor):
    wynik = pow(10000, 10000)
    for i in range(len(lista)):
        for j in range(len(slownik_waga_kolor)-1):
        #for j in waga_zamianaKoloru.keys()-1:
            #if lista[i][len(lista[i]) - 1] == lista[j][len(lista[j]) - 1] and lista[i] != lista[j]:
            if Odleglosc_do_listy(lista[i], lista[list(slownik_waga_kolor)[j]]) < wynik:
                wynik = Odleglosc_do_listy(lista[i], lista[list(slownik_waga_kolor)[j]])
                lista[i][len(lista[i]) - 1] = list(slownik_waga_kolor.values())[j][1]
        wynik = pow(10000, 10000)
    return lista


#print(lista)
#print(lista[1][len(lista[1])-1])
#print(len(lista))

def Wykonaj_knn(ilosc_wykonan):
    for i in range(ilosc_wykonan):
        print("jeden kontra reszta",i+1,":")
        slownik_odleglosci = JedenKontraReszta(lista)
        slownik_odleglosci_kolor = JedenKontraResztaKoloru(lista)
        print("odleglosc do wszystkich:",slownik_odleglosci)
        print("odleglosc do tego samego koloru:",slownik_odleglosci_kolor)
        print("sortowanie:")
        slownik_odleglosci = SortujSlownik(slownik_odleglosci)
        print(slownik_odleglosci)
        print("slownik odleglosci do koloru posortowany:")
        slownik_odleglosci_kolor = SortujSlownik(slownik_odleglosci_kolor)
        print("slownik najmniejszych odleglosci:")
        print(slownik_odleglosci_kolor)
    #print(slownik_odleglosc[0][0][1])
        waga_zamianaKoloru = waga_i_zamianaKoloru(slownik_odleglosci, slownik_odleglosci_kolor)
        print(waga_zamianaKoloru)
    #print(waga_zamianaKoloru.values())
    #print(waga_zamianaKoloru.get())
    #print(list(waga_zamianaKoloru)[0])
    #print(list(waga_zamianaKoloru.values())[0][1])
    #print(len(waga_zamianaKoloru))
        Zmien_kolor(waga_zamianaKoloru)
    print("Srednia arytmetyczna: ",Srednia_arytmetyczna(slownik_odleglosci))
    print("Srednia arytmetyczna koloru: ",Srednia_arytmetyczna_koloru(slownik_odleglosci))
    print("Odchylenie standardowe", Odchylenie_standardowe(slownik_odleglosci))
    print("Wariacja", "{:.2f}".format(Wariancja(slownik_odleglosci)))

"""print("jeden kontra reszta 2:")
slownik_odleglosci = JedenKontraReszta(lista)
slownik_odleglosci_kolor = JedenKontraResztaKoloru(lista)
print(slownik_odleglosci)
print(slownik_odleglosci_kolor)
print("sortowanie 2:")
slownik_odleglosci = SortujSlownik(slownik_odleglosci)
print(slownik_odleglosci)
print("slownik odleglosci posortowany 2:")
slownik_odleglosci_kolor = SortujSlownik(slownik_odleglosci_kolor)
print("slownik odleglosci kolorow posortowany 2:")
print(slownik_odleglosci_kolor)
#print(slownik_odleglosc[0][0][1])
waga_zamianaKoloru = waga_i_zamianaKoloru(slownik_odleglosci, slownik_odleglosci_kolor)
print(waga_zamianaKoloru)
Zmien_kolor(waga_zamianaKoloru)

print("Srednia wspolrzednych punktow: ", SredniaWspolrzednychPunktow(lista))"""


def Wektor_razy_skalar(wektor, skalar):
    return wektor*skalar

def Wektor_razy_wektor(wektor1, wektor2):
    return [i * j for i, j in zip(wektor1, wektor2)]

def Macierz_razy_macierz(macierz1, macierz2):
    if (macierz1.shape or macierz2.shape) != (1, 1):
        return np.dot(macierz1, macierz2)
    else:
        if macierz1.shape == (1, 1):
            macierz1.item()
        else:
            macierz2.item()
        return macierz1*macierz2

def Macierz_transponowana(macierz):
    macierz = np.array([macierz])
    macierz = macierz.T
    return macierz

def Odejmowanie_Macierzy(macierz1, macierz2):
    return np.subtract(macierz1, macierz2)

def Dodawanie_Macierzy(macierz1, macierz2):
    return np.add(macierz1, macierz2)
"""
skalar=3
wektor = [1,1,1,2]
wektor1 = [2, 2, 2, 2]
a = [[1, 0], [0, 1]]
b = [[4, 1], [2, 2]]
C = np.array([[2.2, 0, 0], [0, 2.2, 0], [0, 0, 2.2]])
D = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
E = np.array([[1, 0], [1, 1], [0, 1]])
#b = b.transpose()
print(Wektor_razy_wektor(wektor1,wektor))
print(Macierz_razy_macierz(a,b))
print(b)
print(D)
print(Macierz_razy_macierz(C, D))
macierz = Macierz_razy_macierz(a,b)
print(Wektor_razy_skalar(wektor,3))
print(Macierz_razy_macierz(wektor1,skalar))
print(Macierz_razy_macierz(wektor1, wektor))
print("bezTranspozycji", D)
print("transponowana:",Macierz_transponowana(D))
print(Dodawanie_Macierzy(E,E))
print(Odejmowanie_Macierzy(E,E))

def macierz(n,m):
   A = np.random.randint(0,10,(m,n))
   B = np.empty([n,m], dtype=int)

   print(A)

   for i in range(0,n):
      for j in range(0,m):
         B[i][j]=A[j][i]

   print(B)

   A=A.transpose()
   return A

#print(macierz(5,8))
"""
def Srednia_arytmetyczna_koloru(slownik):
    i = 0
    slownik = collections.OrderedDict(sorted(slownik.items()))
    srednia = 0.0
    wynik = []
    for item in slownik.values():
        for j in range(len(item)):
            srednia += item[j][1]
        key = list(slownik.keys())[i]
        suma = round(srednia/len(item),2)
        i += 1
        #srednia.append(srednia, [key, suma])
        wynik.append([key, suma])
    return wynik,2

def Srednia_arytmetyczna(slownik):
    slownik = collections.OrderedDict(sorted(slownik.items()))
    srednia = 0.0
    for item in slownik.values():
        for i in range(len(item)):
            srednia += item[i][1]
        #print(item)
        #print(item[::][1])
        #print(item, x: x[1])
        #srednia += sum(item, key=lambda x: x[1])
    return round(srednia/len(slownik),2)

def Wariancja(slownik):
    slownik = collections.OrderedDict(sorted(slownik.items()))
    sigma = 0.0
    for item in slownik.values():
        for j in range(len(item)):
            sigma += (item[j][1] - Srednia_arytmetyczna(slownik))**2
    return sigma / len(slownik)

def Odchylenie_standardowe(slownik):
    wariacja = Wariancja(slownik)
    return round(mth.sqrt(wariacja),2)

def RegresjaLiniowa():

    return 0

def Column(matrix, i):
    kol = [row[i] for row in matrix]
    kol = np.array([kol])
    return kol.T

def ObliczanieQR(macierzA):
    num_rows, num_cols = macierzA.shape
    Q = []
    for i in range(num_cols):
        if i==0:
            u = Column(macierzA, i)
            print("MacierzA", macierzA)
            uBezwzgledne = mth.sqrt(Macierz_razy_macierz(u.T, u))
            e = Macierz_razy_macierz(u, 1/uBezwzgledne)
            #print(u)
            #print(uBezwzgledne)
            #print(e)
            #e = np.array([e])
            Q = e
            """
            print(e.T)
            print(Macierz_transponowana(e))
            Q = Macierz_transponowana(e)
            """
            print("Q:",Q)
        else:
            suma = 0
            v = Column(macierzA, i)
            for j in range(i):
                ui = Column(macierzA, j)
                proj = Macierz_razy_macierz((Macierz_razy_macierz(v.T, ui) / Macierz_razy_macierz(ui.T, ui)), ui)
                suma = Dodawanie_Macierzy(suma, proj)
            u = Odejmowanie_Macierzy(v, suma)
            uBezwzgledne = mth.sqrt(Macierz_razy_macierz(u.T, u))
            e = Macierz_razy_macierz(u, 1/uBezwzgledne)
            #e = np.array([e])
            Q = np.hstack((Q, e))
            #e = np.array([e])
            #Q = [Q, e.T]
            #Q = Macierz_transponowana(Q)
            print(u)
            print(uBezwzgledne)
            print(e)
    print("Q ostateczne:",Q)
    R = Macierz_razy_macierz(Q.T, macierzA)
    print("R: ", R)
    QR = Macierz_razy_macierz(Q, R)
    A = QR
    print("QR: ", np.matrix.round(QR))
    #QT = Q.T
    #print(QT)
    #A1 = Macierz_razy_macierz(QT, A)
    #A1 = Macierz_razy_macierz(Macierz_razy_macierz(Q.T, A), Q)
    #print("A1: ", np.matrix.round(A1))

    #return Column(macierzA, 0)

#print(ObliczanieQR(E))

def calkowanieMonteCarlo(l_punktow):
    N = l_punktow  # liczba punktów losowych
    print("Podaj poczatek przedzialu calkowania:")
    x_poczatkowe = float(input())
    print("Podaj koniec przedzialu calkowania:")
    x_koncowe = float(input())
    s = 0.
    dx = x_koncowe - x_poczatkowe
    for i in range(N):
        x = x_poczatkowe + random.uniform(0, dx)
        s += x * x + 2 * x
    s *= dx / N
    print("Wartosc calki wynosi : %8.3f" % s)

def calkowanieMetodaProstokatow(l_punktow):
    N = l_punktow

    print("Podaj poczatek przedzialu calkowania:")
    x_poczatkowe = float(input())
    print("Podaj koniec przedzialu calkowania:")
    x_koncowe = float(input())
    dl = (x_koncowe-x_poczatkowe)/N
    s = 0.
    z = x_poczatkowe
    for i in range(1,N+1):
        z += dl
        s += np.arctan(z) * dl
    print("Wartosc calki wynosi : %8.3f" % s)


def main():
    try:
        #print(lista)
        #Wykonaj_knn(2)
        #print(lista)
        E = np.array([[1, 0], [1, 1], [0, 1]])
        ObliczanieQR(E)
        #print(calkowanieMonteCarlo(1000))
        #print(calkowanieMetodaProstokatow(1000))

    except ValueError as ve:
        return str(ve)

if __name__ == "__main__":
    sys.exit(main())