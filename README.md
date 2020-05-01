# A2 : Warping, Matching, Stitching, Blending
## Part1
### How it works:
It follows the basic idea of k-means clustering by grouping pictures that are closer to each other(shares more distinct feature points).
#####Test code:
```python
python part1.py 2 bigben_6.jpg bigben_8.jpg eiffel_18.jpg eiffel_19.jpg part1_output.txt
```
### Assumptions:
Feature points can be detected efficiently with cv2.ORB_create() before further processing.
Pictures of the same category should share significantly more distinct features than pictures of different categories. 
Problems faced:
The number of shared distinct feature is not largely affected by if two picture belongs to the same category or not, and in some cases picture from different categories can share more distinct feature than picture in the same category.
### Simplifications:
To shorten the time of the whole process, I decided to compute a distance matrix of all possible pairs of data points in the data and use the index as reference when doing clustering in K-Medoids.
### Design decisions:
Tried to use K-Means as the method but ran into problem creating new centroids that minimize the distance from other in-group feature vectors. So, I turned to K-Medoids which uses existing points in data as the new centroids. Which is not so time efficient but also not too bad, and it overcomes the difficulty facing by using K-Means.

