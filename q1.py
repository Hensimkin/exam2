import math
from random import choice

import sympy as sp
from sympy import solve
from sympy.utilities.lambdify import lambdify

def polynom(x):
    return (x**4)+(x**3)-(3*(x**2))


def Bisection_Method(polynom,start_point,end_point,eps):
    z=polynom
    bigarray=[]
    size=(abs(start_point)+abs(end_point))/0.1
    num=start_point
    for i in range(int(size)+1):
        smallarray = []
        smallarray.append(num)
        smallarray.append(round(polynom(num),4))
        bigarray.append(smallarray)
        num = num + 0.1
        num=round(num,3)

    for i in range(1,int(size),1):
        if (bigarray[i][1]*bigarray[i-1][1])<0:
            print("I", "   ", "A", "   ", "B", "   ", "C", "   ", "F(a)", "   ", "F(b)", "   ", "F(c)")
            Bisection(polynom, bigarray[i-1][0], bigarray[i][0], eps)
    x = sp.symbols('x')
    f2 = diff(polynom)
    f3 = lambdify(x, f2)
    for i in range(int(size)+1):
        num=round(bigarray[i][0],6)
        f4=round(f3(num),6)
        bigarray[i].append(f4)

    for i in range(1, int(size)+1, 1):
        if (bigarray[i][2] * bigarray[i - 1][2]) < 0:
            print("I", "   ", "A", "   ", "B", "   ", "C", "   ", "F'(a)", "   ", "F'(b)", "   ", "F'(c)")
            Bisection(f3, bigarray[i - 1][0], bigarray[i][0], eps)


def diff(polynom):
    x = sp.symbols('x')
    my_f = polynom(x)
    return sp.diff(my_f, x)




def Bisection(polynom,a,b,eps):
    c=0
    fc=0
    i=1
    counter=0
    temp=0
    while(abs(a-b) >eps):
        c=round((a+b)/2,6)
        fc=round(polynom(c),6)
        if fc<0 and polynom(a)<0:
            temp=round(a,6)
            counter=1
            a=c
            a=round(a,6)
        if fc<0 and polynom(b)<0:
            temp=round(b,6)
            counter=2
            b=c
            b = round(b, 6)
        if fc>0 and polynom(a)>0:
            temp=round(a,6)
            counter=1
            a=c
            a = round(a, 6)
        if fc>0 and polynom(b)>0:
            temp = round(b, 6)
            counter=2
            b=c
            b = round(b, 6)
        if counter==1:
            print(i, " ", temp, " ", b, " ", c, " ", round(polynom(temp),6), " ", round(polynom(b),6), " ", round(fc,6))
        if counter==2:
            print(i, " ", a, " ", temp, " ", c, " ", round(polynom(a),6), " ", round(polynom(temp),6), " ", round(fc,6))
        i+=1
    print("The point is: ",c)





def Newton_Raphson(f,start,end,eps):
    x = sp.symbols('x')
    f_prime=diff(f)
    f_prime = lambdify(x, f_prime)
    xr=round((start+end)/2,6)
    result1=round(f(xr),6)
    result2=round(f_prime(xr),6)
    i=1
    print("f'(x)         f(x)         xr      iteration num     ")
    print(result2,"      ",result1,"      ",xr,"      ",i,"     x0=",xr)
    xr_next=round(xr-(result1/result2),6)
    while(abs(xr-xr_next)>eps):
        i+=1
        xr=round(xr_next,6)
        result1 = round(f(xr),6)
        result2 = round(f_prime(xr),6)
        xr_next = round(xr - (result1 / result2),6)
        print(result2, "      ", result1, "      ", xr, "      ", i)
    print("the result of the function in that range is: ",xr_next)


def simpson(table,n):
    h=(table[len(table)-1][0]-table[0][0])/n
    total=table[0][1]
    for i in range(1,len(table)-1):
        if i%2==0:
            total = total + (2*table[i][1])
        else:
            total = total + (4 * table[i][1])
    total=total+ table[len(table)-1][1]
    return (1/3)*h*total


def numofparts(func,originalfunc,points):
    arr=[]
    x = sp.symbols('x')
    func = lambdify(x, func)
    for i in range(len(points)):
        if func(points[i])<0:
            arr.append(points[i])
    originalfunc = lambdify(x, originalfunc)
    max=originalfunc(points[0])
    for i in range(len(arr)):
        if originalfunc(points[i])>max:
            max=originalfunc(points[i])
    print(max)
    return max



def derive(f):
    x = sp.symbols('x')
    f_prime = f.diff(x)
    #y = lambdify(x, f)
    #f_prime = lambdify(x, f_prime)
    return f_prime


def findanswer(f):
    z=solve(f)
    return z

def calculateder4(func,a,b):
    z=func
    z=derive(z)#1
    arr=solve(z)
    for i in range(3):
        z=derive(z)#4
    max=numofparts(func, z, arr)
    x = sp.symbols('x')
    func = lambdify(x, func)
    return func(max)


def calcall(h,n,a,b,max):
    #total=((h**5)/90)*((b-a)/h)*(max/2)
    total=((1/180)*(h**4))*(b-a)*max
    print(total)

def func(x):
    return (sp.cos(2*(math.e**(-2*x))))/(2*x**3+5*x**2-6)








def main2():
    table = [[-0.4, 0.0484], [-0.2 , 0.169799324529847], [0,0.0693578060911904], [0.2 , -0.0394415009080748], [0.4, -0.122764331442294]]
    n = 4
    h = (table[len(table) - 1][0] - table[0][0]) / n
    print(simpson(table, n))
    x = sp.symbols('x')
    func = (sp.cos(2*(math.e**(-2*x))))/(2*x**3+5*x**2-6)
    max = calculateder4(func, table[0][0], table[len(table) - 1][1])
    calcall(h, n, table[0][0], table[len(table) - 1][0], max)


def polynom(table, point):
    if len(table) < 3:
        print("cant calculate")
    else:
        i = 0
        mat = [[1], [1], [1]]
        vec = [[], [], []]
        """""
        while table[i][0] < point and table[i + 1][0] < point and table[i + 2][0] < point:
            i += 1
        mat[0].append(table[i][0])
        mat[0].append((table[i][0]) ** 2)
        mat[1].append(table[i + 1][0])
        mat[1].append((table[i + 1][0]) ** 2)
        mat[2].append(table[i + 2][0])
        mat[2].append((table[i + 2][0]) ** 2)
        vec[0].append(table[i][1])
        vec[1].append(table[i + 1][1])
        vec[2].append(table[i + 2][1])
        """""
        a=choice(table)
        if a==table[len(table)-2] or a==table[len(table)-1]:
            a=table[len(table)-3]
        mat[0].append(a[0])
        mat[0].append(a[0]**2)
        vec[0].append(a[1])
        b=a
        while b==a or (b[0]<a[0]):
            b=choice(table)
        if b==table[len(table)-1]:
            b=table[len(table)-2]
        mat[1].append(b[0])
        mat[1].append(b[0] ** 2)
        vec[1].append(b[1])

        c=a
        while (c==a or c==b) or c[0]<b[0]:
            c=choice(table)
        mat[2].append(c[0])
        mat[2].append(c[0] ** 2)
        vec[2].append(c[1])

        print(mat)
        print(vec)
        """""
        #mat=[[1,1,1],[1,2,4],[1,3,9]]
        vec=[[0.8415],[0.9093],[0.1411]]
        mat=getMatrixInverse(mat)
    
        mat = [[1, 2, 4], [1, 3, 9], [1, 4, 16]]
        vec = [[0.9093], [0.1411],[-0.7568]]
        """""
        mat = getMatrixInverse(mat)
        g = mult(mat, vec)
        sum = g[0][0] + (g[1][0] * point) + (g[2][0] * (point ** 2))
        print("the point of", point, "is", round(sum, 5))


def nevile(table, point):
    myarr=[[0 for col in range(len(table))]for row in range(len(table))]
    for i in range(len(table)):
        if i==0:
            for j in range(len(table)):
                myarr[j][j]=round(table[j][1],4)
        else:
            for k in range(len(table)):
                if k+i<len(table):
                 myarr[k][k+i]=round((((point-table[k][0])*myarr[k+1][k+i])-((point-table[k+i][0])*myarr[k][k+i-1]))/(table[k+i][0]-table[k][0]),4)
        print(myarr)
    print("The value of f(", point, ") is :",round(myarr[0][len(table)-1],4) )



def transposeMatrix(m):
    a= list(map(list,zip(*m)))
    return a

def getMatrixMinor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

def getMatrixDeternminant(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c))
    return determinant

def getMatrixInverse(m):
    determinant = round(getMatrixDeternminant(m),6)
    if len(m) == 2:
        return [[round(m[1][1]/determinant,6), round(-1*(m[0][1]/determinant),6)],
                [round(-1*(m[1][0]/determinant),6), round(m[0][0]/determinant,6)]]
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m,r,c)
            cofactorRow.append(((-1)**(r+c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transposeMatrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
    return cofactors

def mult(matrix1,matrix2):
    res=[[0 for x in range(len(matrix2[0]))] for y in range(len(matrix1))]
    size=len(matrix1)
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                res[i][j] += matrix1[i][k] * matrix2[k][j]
    return res


def multiply(v, G):
    result = []
    for i in range(len(G[0])):
        total = 0
        for j in range(len(v)):
            a=round((G[j][i])*(v[j]),6)
            total = total+a
        result.append(total)
    return result



def multiply2(v, G):
    result = []
    for i in range(len(G[0])):
        total = 0
        for j in range(len(v)):
            a=round((G[i][j])*(v[j]),6)
            total = round(total+a,5)
        result.append(total)
    return result





def transposeMatrix(m):
    a= list(map(list,zip(*m)))
    return a

def getMatrixMinor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

def getMatrixDeternminant(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c))
    return determinant


print("Part I")
print("\n")
print("\n")
print("Bisection method")
Bisection_Method(func,-1.1,2,0.0001)
print("\n")
print("Newton Raphson method")
Newton_Raphson(func,-1.1,2,0.0001)
print("\n")
print("Simpson method")
print("\n")
#main2()
print("Part II")
print("\n")
print("\n")
table=[[2.2,1.50],[3.3 ,1.69],[4.4,1.90],[3.5,2.12],[2.6,2.37]]
print("polynom")
print("\n")
print("\n")
polynom(table, 4.5)
print("nevile")
print("\n")
print("\n")
nevile(table, 4.5)






