#this is just a demonstartion of KNN with pictures, 
#going to have 3 categories that are triangle, circle, square

#going to go with a non linear model most likely, this is more 

from turtle import distance
from types import new_class
import matplotlib.pyplot as plt
import math

x = [0, 100, 80, 20, 0, 75, 25, 75, 10, 90, 32, 86, 13, 65, 34, 80]
y = [100, 0, 20, 80, 75, 0, 75, 25, 90, 10, 86, 32, 65, 13, 80, 34]
classes = [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

#new point, this should go to class 0
new_x = 8
new_y = 80

#3 identifiable objects: seal, denomination and person for both currancies
#going to focus on seal and denomination for the time being
seal_x = [0, 100, 80, 20, 0, 75, 25, 75, 10, 90, 32, 86, 13, 65, 34, 80]
seal_y = [100, 0, 20, 80, 75, 0, 75, 25, 90, 10, 86, 32, 65, 13, 80, 34]
seal_class = [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

denom_x = [0, 100, 80, 20, 0, 75, 25, 75, 10, 90, 32, 86, 13, 65, 34, 80]
denom_y = [100, 0, 20, 80, 75, 0, 75, 25, 90, 10, 86, 32, 65, 13, 80, 34]
denom_class = [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

#new point will be given to us via image and also we will know where to place it
# ASSUMED it was the correct orientation
image = "" #numpy array in the future, but image for now
type = "D"  # D: denomination, S: seal

#calculate percentage match
new_x = 35 #need to calculate this
new_y = 89

#find the closet point of all the classes, whichever is closer in distance it is said class
distance_points_denom = []
distance_points_seal = []
new_class = -1 

if type == "S":
    for i in range(len(seal_x)):  # go through all know points to calculate the distance for the seal
        dist_x = new_x - seal_x[i]
        dist_y = new_y - seal_y[i]
        total = dist_x**2 + dist_y**2
        total = math.sqrt(total)
        distance_points_denom.append(total)

    index_smallest_path = distance_points_denom.index(min(distance_points_denom))
    new_class = classes[index_smallest_path]

    
if type == "D":
    for i in range(len(denom_x)):  # go through all know points to calculate the distance for the denomination
        dist_x = new_x - denom_x[i]
        dist_y = new_y - denom_y[i]
        total = dist_x**2 + dist_y**2
        total = math.sqrt(total)
        distance_points_seal.append(total)

    index_smallest_path = distance_points_seal.index(min(distance_points_seal))
    new_class = classes[index_smallest_path]

# need to check to see if its too far out from the perfect image
# line 
dist = abs(new_x - new_y)

if dist <= 10:
    new_class = 2 #too close to tell


#add to the lists
if type == "S":
    seal_x.append(new_x)
    seal_y.append(new_y)
    classes.append(new_class)
if type == "D":
    denom_x.append(new_x)
    denom_y.append(new_y)
    classes.append(new_class)

#print("Image is ")

#display all points and prediction
x.append(new_x)
y.append(new_y)
plt.scatter(x, y, c=classes)
plt.text(x=new_x, y=new_y, s=f"new point, class: {new_class}")
plt.show()

