#!/usr/bin/python
import numpy as np

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    errors = np.absolute(predictions-net_worths)

    print errors.shape
    cleaned_data = np.hstack([ages, net_worths, errors])
    print cleaned_data.shape
    cleaned_data = [(x[0], x[1], x[2]) for x in cleaned_data]

    cleaned_data = sorted(cleaned_data, key=lambda x: x[2], reverse=False)[0:int(0.9*len(cleaned_data))]

    return cleaned_data

