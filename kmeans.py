from math import dist
from random import randrange

# Local
from data import DataPoint


class _Centroid(DataPoint):
    def __init__(self, x: float, y: float, n: int):
        super().__init__(x, y)

        self.id = n
        self.convergence = 0.0

    def update(self, datapoints: list):
        x = self.x
        y = self.y

        self.x = sum(dp.x for dp in datapoints) / len(datapoints)
        self.y = sum(dp.y for dp in datapoints) / len(datapoints)

        # Euclidean distance from the previous location
        self.convergence = dist((self.x, self.y), (x, y))


class KMeansDataPoint(DataPoint):
    def __init__(self, x: float, y: float, centroid: _Centroid):
        super().__init__(x, y)

        self.centroid = centroid


# Return a KMeansDatapoint list classified using K-Means
# Arguments: DataPoint list, number of centroids and convergence threshold
def kmeans(datapoints: list, k: int, threshold: float = 0.0001) -> list:
    max_x = max(dp.x for dp in datapoints)
    max_y = max(dp.y for dp in datapoints)

    # Initialize K centroids at random points
    centroids = [
        _Centroid(randrange(max_x), randrange(max_y), i) for i in range(1, k + 1)
    ]

    # Initialize K-Means data points identified by the first centroid
    km_datapoints = [
        KMeansDataPoint(dp.x, dp.y, centroids[0]) for dp in datapoints
    ]

    # Initialize average convergence as a number higher than the threshold
    avg_convergence = threshold + 1

    while avg_convergence > threshold:
        # Define to which centroid each data point belongs by calculating their
        # Euclidean distance to each of the centroids and selecting the closest.
        for dp in km_datapoints:
            dc = dist((dp.centroid.x, dp.centroid.y), (dp.x, dp.y))

            for c in centroids[1:]:
                if (d := dist((c.x, c.y), (dp.x, dp.y))) < dc:
                    dc = d
                    dp.centroid = c

        # Update centroids
        for c in centroids:
            c.update([dp for dp in km_datapoints if dp.centroid == c])

        # Update average convergence
        avg_convergence = sum(c.convergence for c in centroids) / len(centroids)

    # Return KMDataPoint list
    return km_datapoints
