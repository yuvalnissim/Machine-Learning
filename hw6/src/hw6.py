import numpy as np

def get_random_centroids(X, k):

    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    
    centroids = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    indices = X.shape[0]
    print("indices:", indices)
    random_k_indices =np.random.choice(indices ,k , replace=False)
    centroids = X[random_k_indices, :]
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # make sure you return a numpy array
    
    return np.asarray(centroids).astype(np.float) 



def lp_distance(X, centroids, p=2):

    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    
    k = len(centroids)
    distances = []

    number_of_instances = X.shape[0]
    distances = np.zeros((centroids.shape[0], number_of_instances))
    
    for a, b in enumerate(centroids):
        distances[a] = np.sum(np.abs(X - b)**p,axis=1)**(1/p)
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return distances

def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    centroids = get_random_centroids(X, k)
    prev_centroids = centroids.copy()

    for a in range(max_iter):
        distances = lp_distance(X, centroids, p)
        classes = np.argmin(distances, axis=0)
        centroids_temp = []

        for lbl in range(k):
            c = np.mean(X[classes == lbl], axis=0)
            centroids_temp.append(c)

        centroids = np.array(centroids_temp)

        if np.array_equal(centroids, prev_centroids):break
        else:
            prev_centroids = centroids.copy()
            
            
    return centroids, classes
        
            
            

def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = np.zeros((k,3))
    rnd = np.random.randint(0, X.shape[0])
    centroids[0,:] = X[rnd, :]
    Xc = X.copy()
    Xc = np.delete(Xc, rnd, axis=0)
    for i in range(1,k):
        dis = lp_distance(Xc, centroids, p)
        disMat = dis.min(axis=0) ** 2
        disTotal =disMat.sum()
        wDis = disMat / disTotal
        rnd = np.random.choice(Xc.shape[0], size=1,replace=False, p=wDis)
        centroids[i,:] = Xc[rnd,:]
        Xc = np.delete(X, rnd, axis=0)
        
            
    prev_centroids = centroids.copy()

    for a in range(max_iter):
        distances = lp_distance(X, centroids, p)
        classes = np.argmin(distances, axis=0)
        centroids_temp = []

        for lbl in range(k):
            c = np.mean(X[classes == lbl], axis=0)
            centroids_temp.append(c)

        centroids = np.array(centroids_temp)

        if np.array_equal(centroids, prev_centroids):break
        else:
            prev_centroids = centroids.copy()
            
            
    # K: number of centroid is between 1 to m.
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes
