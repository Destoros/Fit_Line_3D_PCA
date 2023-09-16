from collections import deque
import numpy as np
import logging
from typing import Callable



"""
This class uses SVD and the first principal component to determine the best
fitting line to some given measurements. This is similar to linear regression
for 2D case but can be extended to more than 2 dimensions. The sources used for
this are listed below: 
- https://stackoverflow.com/questions/2298390/fitting-a-line-in-3d 
- https://en.wikipedia.org/wiki/Singular_value_decomposition 
- https://www.youtube.com/watch?v=CpD9XlTu3ys&pp=ygUoc2luZ3VsYXIgdmFsdWUgZGVjb21wb3NpdGlvbiAzYmx1ZTFicm93bg%3D%3D

The class takes care of storing the measurements, fitting the line to it AND
projecting the newest sample onto the fitted line. This projection onto the
fitted line is done because the moving car is expected to move along a line and
by projecting the newest point onto the line, outlier which vary orthogonal to
the line have almost no impact. With this said, outlier in the same direction as
the fitted line do not benefit from this property.

TODO: We will have to see how this behaves if time is also used as input for the
fitted line.
"""


#TODO: Use the FilterInterface as base class 


# create a logger object which prints to the serial interface
logger = logging.getLogger(__name__)

class FitAndProjectLine():

    def __init__(self, n_samples: int, min_samples: int = 3) -> None:
        """Constructor class
            n_samples: int; Amount of samples used for the line fitting
            min_samples: int; Amount of samples required until the first time a
                               line will be fitted
            
            If the amount of samples saved within this class is bigger than
            min_samples, the class will start to project the latest measurement
            onto the fitted line. This parameter was implemented to reduce the
            time until the first estimated position is displayed to the user.
            For example if n_samples = 20 and min_samples = 20, the first
            estimated position will be displayed to the user after 20*0.3s = 6s
            give the sample_period = 0.3s, which is is way to long. Setting
            min_samples = 3 reduces this time to 0.9s which is still not perfect
            but acceptable plus the fitted line gets better and better over time
            and uses 20 samples in the end.
             
            NOTE: Test which amount of samples at which sample rate delivers the
            best user expierence. It should not happen, that a vehicle which
            already charged and aborted, can not start charging again, because
            the position measurements deliver an invalid positon.
            """


        # if the algortihm stores less samples than `min_samples`, it will never
        # estimate a position and would be useless
        if n_samples < min_samples:
            raise Exception(
                f"PROJLINE: The amount of samples used for estimation must be\
                 greater or equal to min_samples = {self.min_samples},\
                 but it was set to {n_samples}")
        
        #save the parameters inside the instance
        self.n_samples = n_samples
        self.min_samples = min_samples  

        # create an empty queue object which can store up to n_samples
        self.positions = deque([], maxlen=n_samples)  
        # maxlen assures that when adding more then maxlen entries, the oldest
        # entry will be deleted

        #create a counter for how many measurements were added in total
        self.n_measurements_total: int = 0

    def add_measurement(self, position: np.ndarray) -> bool:
        """
        This method adds a position measurement to the queue.
        """
        # Flatten the input to a vector
        position = position.flatten()

        # if len(self.positions) > self.min_samples and self.detect_outlier(position):
        #     print("Outlier detected")
        #     return False

        # append it to matrix/queue holding all positions
        self.positions.append(position)

        # a measurement was added -> increase the couter
        self.n_measurements_total += 1
        logger.debug(
            f"PROJLINE: Measurement Nr. {self.n_measurements_total} with n_samples = {self.n_samples}")

        return True
    
    def fit_line_to_data(self) -> Callable:
        """
        This method determines the first principal component using SVD (Singular
        Vector Decomposition) and returns a lambda function with the scalar
        paramter t which returning a point on the line. This procedure is
        equivalent to fitting a function in the Least-Squares sense.
        """

        # convert the queue into a matrix, where each row is one sample, and
        # each column is dimension/feature
        position_matrix = np.array(self.positions)

        # source for the following code: 
        # https://stackoverflow.com/questions/2298390/fitting-a-line-in-3d

        # Calculate the mean of the points, i.e. the 'center' of the cloud
        position_matrix_mean = position_matrix.mean(axis=0)
        # This will be the origin of the fitted line

        
        # Do an SVD =Singular Value Decomposition) on the mean-centered data.
        _, _, vv = np.linalg.svd(position_matrix - position_matrix_mean, full_matrices=False) 
        #use full_matrices to reduce the required memory
        
        # Now vv[0] contains the first principal component, i.e. the direction
        # vector of the 'best fit' line in the least squares sense.

        # Create a lambda function which takes in the scalar parameter t and
        # returns a point on the fitted line
        fitted_line = lambda t: (vv[0] * t + position_matrix_mean).reshape(-1,1)

        return fitted_line


    def project_point_onto_line(self, p: np.ndarray, line_function: Callable) -> np.ndarray:
        """
        Given a function describing a line, calculate the orthogonal projection
        of the point p onto this line. 
        Source: https://math.stackexchange.com/questions/62633/orthogonal-projection-of-a-point-onto-a-line
        """

        # Make sure p is a column vector
        p = p.reshape(-1,1)

        # Get the origin of the fitted line
        p0 = line_function(0)
        
        # Get a vector pointing along the fitted line and normalize it to 1
        v = line_function(1) - p0 # Spitze minus Schaft
        v /= np.linalg.norm(v) # normalize to 1

        # Calculate an intermediate term
        projection_matrix = np.dot(v, v.T) 
        # The division term displayed on the website is not needed, because v
        # has a length of 1

        # Project the point onto the line
        p_projected = projection_matrix.dot(p) - projection_matrix.dot(p0) + p0 

        return p_projected

    def estimate_position(self) -> np.ndarray:
        """
        Estimates the positon of the latest entered measurement by projecting it
        onto a fitted line. If less than min_samples were added yet, it simply
        returns the latest added positon.
        """

        # Guard clause; If less than min_samples were added yet return the
        # latest position
        if len(self.positions) < self.min_samples:
            return self.positions[-1]  
        
        # Fit a line to all given measurements (this includes the latest one,
        # which will be project later)
        fitted_line_function = self.fit_line_to_data()
        
        # Project the latest measurement onto the fitted line
        estimated_position = self.project_point_onto_line(self.positions[-1] , fitted_line_function)

        logger.debug(
            f"PROJLINE: Projected point {self.positions[-1]} onto the fitted line, new coordinates = {estimated_position}")

        #return the numpy array 
        return estimated_position
    


def main():
    n_samples = 10
    x = np.mgrid[-2:5:n_samples*1j]
    y = np.mgrid[1:9:n_samples*1j]
    z = np.mgrid[-5:3:n_samples*1j]

    data = np.concatenate((x[:, np.newaxis], 
                        y[:, np.newaxis], 
                        z[:, np.newaxis]), 
                        axis=1)
    
    # Perturb with some Gaussian noise
    noise = np.random.normal(size=data.shape) * 0.4
    data += noise

    # Instantiate the class

    estimater = FitAndProjectLine(n_samples)

    # Verify that everything looks right.
    import matplotlib.pyplot as plt


    # Feed several positions to the class
    for i in range(n_samples):
        estimater.add_measurement(data[i])

        # Fit a line
        fitted_line = estimater.fit_line_to_data()

        # Create two points along this line for plotting
        linepts = np.hstack([fitted_line(-7), fitted_line(7)])

        if i >= 2:

            # Plot it
            ax = plt.axes(projection='3d')
            ax.scatter3D(*np.array(estimater.positions).T)
            ax.plot3D(*linepts)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_xlim((x[0], x[-1]))
            ax.set_ylim((y[0], y[-1]))
            ax.set_zlim((z[0], z[-1]))

            fitted_point = estimater.estimate_position()
            ax.scatter3D(*fitted_point, color="red")    

            plt.show()



def time_estimate_position():
    import time

    """" This function times the estimate_position function, since it does some
    heavy math. The result is, that the 
    """

    n_samples = 10000 #the SDA works with matrices, which increases with n_samples
    #Therefore increasing n_samples also increases the execution time of the SVA
    #algorithm

    #How often the execution of estiamte_position() should be repeated, the
    #final time gets divided by this amount to get an average
    n_repetition = 10000 #10_000 is more than enough 

    # Do not touch below!
    #===============================
    x = np.mgrid[-2:5:n_samples*1j]
    y = np.mgrid[1:9:n_samples*1j]
    z = np.mgrid[-5:3:n_samples*1j]

    data = np.concatenate((x[:, np.newaxis], 
                        y[:, np.newaxis], 
                        z[:, np.newaxis]), 
                        axis=1)
    
    # Perturb with some Gaussian noise
    noise = np.random.normal(size=data.shape) * 0.4
    data += noise

    # Instantiate the class
    estimater = FitAndProjectLine(n_samples)

    # Feed n_samples positions to the class to get a full matrix
    for i in range(n_samples):
        estimater.add_measurement(data[i])


    #time the function
    start = time.time()    

    for i in range(n_repetition):
        estimater.estimate_position()

    end = time.time()

    print(f"Average elapsed time for estimate_positon {((end - start)*10**6)/n_repetition:.4f} µs using {n_samples = }")


if __name__ == "__main__":
    # main()
    time_estimate_position()