## load dependency - third party
import numpy as np

## euclidean distance calculation
def euclidean_dist(a, b, d):
    
    """ 
    calculates the euclidean distance between observations for data 
    containing only numeric / continuous features, returns float value
    """
    
    ## create list to store distances
    dist = [None] * d
    
    ## loop through columns to calculate euclidean 
    ## distance for numeric / continuous features
    for i in range(d):
        
        ## the squared difference of values in
        ## vectors a and b of equal length 
        dist[i] = (a.iloc[i] - b.iloc[i]) ** 2
        
    ## sum all the squared differences and take the square root
    dist = np.sqrt(sum(dist))
    
    ## return distance list
    return dist

## heom distance calculation
def heom_dist(a_num, b_num, d_num, ranges_num, a_nom, b_nom, d_nom):
    
    """ 
    calculates the heterogenous euclidean overlap (heom) distance between 
    observations for data containing both numeric / continuous and nominal  
    / categorical features, returns float value
    
    ref:
        
    Wilson, D., Martinez, T. (1997). 
    Improved Heterogeneous Distance Functions.
    Journal of Artificial Intelligence Research, 6:1-34.
    https://arxiv.org/pdf/cs/9701101.pdf.
    """
    
    ## load dependency
    # import numpy as np
    
    ## create list to store distances
    dist = [None] * d_num
    
    ## specify epsilon
    eps = 1e-30
    
    ## loop through columns to calculate euclidean 
    ## distance for numeric / continuous features
    for i in range(d_num):
        
        ## epsilon utilized to avoid division by zero
        if ranges_num[i] > eps:
        
            ## the absolute value of the differences between values in
            ## vectors a and b of equal length, divided by their range, squared
            ## (division by range conducted for normalization)
            dist[i] = (abs(a_num.iloc[i] - b_num.iloc[i]) / ranges_num[i]) ** 2
    
    ## loop through columns to calculate hamming
    ## distance for nominal / categorical features
    for i in range(d_nom):
        
        ## distance equals 0 for values that are equal
        ## in two vectors a and b of equal length
        if a_nom.iloc[i] == b_nom.iloc[i]:
            dist[i] = 0
        
        ## distance equals 1 for values that are not equal
        else:
            dist[i] = 1
        
        ## theoretically, hamming differences are squared when utilized
        ## within heom distance, however, procedurally not required, 
        ## as squaring [0,1] returns same result
    
    ## sum all the squared differences and take the square root
    dist = np.sqrt(sum(dist))
    
    ## return distance list
    return dist

## hamming distance calculation
def overlap_dist(a, b, d):
    
    """ 
    calculates the hamming (overlap) distance between observations for data 
    containing only nominal / categorical features, returns float value
    """
    
    ## create list to store distances
    dist = [None] * d
    
    ## loop through columns to calculate hamming
    ## distance for nominal / categorical features
    for i in range(d):
        
        ## distance equals 0 for values that are equal
        ## in two vectors a and b of equal length
        if a.iloc[i] == b.iloc[i]:
            dist[i] = 0
        
        ## distance equals 1 for values that are not equal
        else:
            dist[i] = 1
    
    ## sum all the differences   
    dist = sum(dist)
    
    ## return distance list
    return dist
