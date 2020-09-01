from sklearn.datasets 
import load_digits 
from sklearn.cluster 
import KMeans 
from sklearn.preprocessing 
import StandardScaler digits = load_digits() dataset = digits.data print(dataset[0])
img = cv2.imread('/home/suraj/Desktop/digit2.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(img_gray)

%matplotlib inline
plt.imshow(img_gray, cmap='gray')

