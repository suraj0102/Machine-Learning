import numpy as np
a = np.array([1, 2, 3])   
print(type(a))            
print(a.shape)            
print(a[0], a[1], a[2])
b = np.array([[1,2,3],[4,5,6]])    
print(b)
print(b.shape)                     
print(b[0, 0], b[0, 1], b[1, 0]) 

a = np.zeros((2,2))   
print(a)              
                      
b = np.ones((1,2))    
print(b)              

c = np.full((2,2), 7)  
print(c)               
                       

d = np.eye(2)         
print(d)              
                      

e = np.random.random((2,2))  
print(e) 
import numpy as np

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)
print(x + y)
print(np.add(x, y))

print(x - y)
print(np.subtract(x, y))

print(x * y)
print(np.multiply(x, y))

print(x / y)
print(np.divide(x, y))

print(np.sqrt(x))

import numpy as np

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

print(v.dot(w))
print(np.dot(v, w))
print(x.dot(v))
print(np.dot(x, v))

print(x.dot(y))
print(np.dot(x, y))
import numpy as np

x = np.array([[1,2], [3,4]])
print(x)    
print(x.T)  
            
v = np.array([1,2,3])
print(v)    
print(v.T)  
import numpy as np 
import matplotlib.pyplot as plt 

x = np.arange(0, 10, 0.1)
y = 1 + (x * 2) + (np.random.normal(0, 1, len(x)) * 5)

mx = x.mean()
my = y.mean()
temp = x - mx
c1 = np.sum(temp * (y - my)) / np.sum(temp ** 2)
c0 = my - c1 * mx

x2 = [0,10]
y2 = [c0 + c1*0, c0 + c1*10]

my_dpi = 96

plt.scatter(x,y, color='b', s=20)
plt.plot(x2, y2, color='r', linewidth=3)
plt.axis([0,10,-5,30])

plt.show()
