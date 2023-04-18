import numpy as np
import matplotlib.pyplot as plt

y = np.random.uniform(low = -100, high = 100, size = [50])
x = np.random.uniform(low = -100, high = 100, size = [50])

for i in range(50): #round
    x[i] = round(x[i], 1)
    y[i] = round(y[i], 1)

print(x.tolist())  # the array
print(y.tolist())

classed = []

for s in range(len(x)):
    if abs(x[s]) < 2.5 or abs(y[s]) < 2.5:
        classed.append(4) #neither
    elif x[s] > 0: #right
        if y[s] > 0: #top
            classed.append(1)
        else: #bottom
            classed.append(3)
    else: #left
        if y[s] > 0: #top
            classed.append(0)
        else: #bottom
            classed.append(2)
    	
print(classed) #the classes of each point

plt.scatter(x, y, c=classed, s=10)
plt.ylabel("Y")
plt.xlabel("X")
plt.show()

