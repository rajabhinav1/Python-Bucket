# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
string = input("Please enter your own String : ")
if string==1:
    longgame=0

longgame=(len(string))
count = 0
countA=0
count1 = 0
count2= 0
count3= 0
count4= 0
count5=0
count6=0
count7=0
count8=0
count9=0
count10=0
count11=0
count12=0
count13=0
count14=0
count15=0
count16=0
count17=0
count18=0
count19=0
count20=0
count21=0
count22=0
count23=0
count24=0
count25=0
A='A'
E='E'
I='I'
O='O'
U='U'
B='B'
C='C'
D='D'
E='E'
F='F'
G='G'
H='H'
J='J'
K='K'
L='L'
M='M'
N='N'
P='P'
Q='Q'
R='R'
S='S'
T='T'
V='V'
W='W'
X='X'
Y='Y'
Z='Z'
for i in range(len(string)):
    if(string[i] == 0):
        count =0
for i in range(len(string)):
    if (string[i] == A):
        countA = countA + 1
for i in range(len(string)):
    if (string[i] == E):
        count1= count1 + 1

for i in range(len(string)):
    if (string[i] == I):
        count2 = count2 + 1

for i in range(len(string)):
    if (string[i] == O):
        count3 = count3 + 1

for i in range(len(string)):
    if (string[i] == U):
        count4 = count4 + 1
for i in range(len(string)):
    if (string[i] == B):
        count5 = count5 + 1
for i in range(len(string)):
    if (string[i] == C):
        count6 = count6 + 1
for i in range(len(string)):
    if (string[i] == D):
        count7 = count7 + 1
for i in range(len(string)):
    if (string[i] == F):
        count8 = count8 + 1
for i in range(len(string)):
    if (string[i] == G):
        count9 = count9 + 1
for i in range(len(string)):
    if (string[i] == H):
        count10 = count10 + 1
for i in range(len(string)):
    if (string[i] == J):
        count11 = count11 + 1
for i in range(len(string)):
    if (string[i] == K):
        count12 = count12 + 1
for i in range(len(string)):
    if (string[i] == L):
        count13 = count13 + 1
for i in range(len(string)):
    if (string[i] == M):
        count14 = count14 + 1

for i in range(len(string)):
    if (string[i] == N):
        count15 = count15 + 1
for i in range(len(string)):
    if (string[i] == P):
        count16 = count16 + 1
for i in range(len(string)):
    if (string[i] == Q):
        count17 = count17 + 1
for i in range(len(string)):
    if (string[i] == R):
        count18= count18 + 1
for i in range(len(string)):
    if (string[i] == S):
        count19 = count19 + 1
for i in range(len(string)):
    if (string[i] == T):
        count20 = count20 + 1
for i in range(len(string)):
    if (string[i] == V):
        count21 = count21 + 1
for i in range(len(string)):
    if (string[i] == W):
        count22 = count22 + 1
for i in range(len(string)):
    if (string[i] == X):
        count23 = count23 + 1
for i in range(len(string)):
    if (string[i] == Y):
        count24 = count24 + 1
for i in range(len(string)):
    if (string[i] == Z):
        count25 = count25 + 1


vowels = (countA+count1+count2+count3+count4)
onso=(count5+count6+count7+count8+count9+count10+count11+count12+count13+count14+count15+count16+count17+count18
       +count19+count20+count21+count22+count23+count24+count25)
vx=max(countA,count1,count2,count3,count4)
VC=max(count5,count6,count7,count8,count9,count10,count11,count12,count13,count14,count15,count16,count17,count18
       ,count19,count20,count21,count22,count23,count24,count25)

print(onso)
print(VC)
print(vowels)
t=(onso-2)
n=(vowels-2)
s=(onso-1)
m=(vowels-1)
osi=4
nu=1

#CONDITIONAL LOOPING TREES
#if longgame==1:
    #print ("0")

if longgame==vowels and longgame==vx:
    print("0")
elif longgame==onso and longgame==VC:
    print("0")
elif VC==onso and vowels==onso:
    print (vowels)
elif vx==vowels and vowels==onso:
    print (onso)
elif vx==vowels and vowels>=onso:
    print (onso)
#debug1
elif vx >=osi and vx==m and longgame>=osi and onso==nu:
    digit=(((vowels-vx)*2)+onso)
    print(digit)
elif VC>=osi and VC==s and longgame>=osi and vowels==nu:
    piz=(((onso-VC)*2)+vowels)
    print (piz)

elif onso >= vowels and onso==VC:
    print(vowels)

#1ST
elif vowels>=onso and vowels >=vx:
    a=(vowels + ((onso-VC)*2))
    print (a)
elif vowels>=onso and vowels==vx:
    print(onso)

#2ND

elif onso >= vowels and onso>=VC:
    b = (onso + ((vowels-vx)*2))
    print(b)
#DEBUG repeated above
elif onso >= vowels and onso==VC:
    print(vowels)

#3RD
#elif onso==vowels and :

elif onso==vowels and VC==vowels:
    print (len(vowels))

elif onso==vowels and vx==onso:
    print (len(onso))



elif onso==vowels and VC>=vx:
    d = (vowels + ((onso-VC)*2))
    print(d)

#4TH

elif onso==vowels and vx>=VC:
    e = (onso + ((vowels-vx)*2))
    print(e)


















