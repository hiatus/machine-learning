from math import dist

# Local
from data import DataPoint


class KNNDataPoint(DataPoint):
    def __init__(self, x: float, y: float, c: int):
        super().__init__(x, y)

        # Almost killed myself for not being able to use 'class'
        self.category = c


# The optimal value for K in KNN is generally accepted as the square root of the
# dataset's length, but it's not a rule. Smaller values result in noise having a
# higher influence on the result, whereas larger values are more computationally
# expensive.
def optimal_k(n: int) -> int:
    return int(n ** 0.5)

# Return a KNNDataPoint classified using KNN. The 'datapoints' argument expects
# a KNNDataPoint list which will be used to classify 'datapoint'.
def knn(datapoints: list, datapoint: DataPoint, k: int) -> KNNDataPoint:
    # Sort data points by their Euclidean distance to the target data point and
    # truncate the sorted list the it's K first elements.
    k_nearest = sorted(
        datapoints,
        key = lambda dp: dist([dp.x, dp.y], [datapoint.x, datapoint.y])
    )[:k]

    # Only the category attribute is needed
    k_nearest_categories = [kn.category for kn in k_nearest]

    # The data point's class is the class with the highest incidence
    knn_category = k_nearest_categories[0]
    knn_category_incidence = k_nearest_categories.count(k_nearest_categories[0])

    for knc in k_nearest_categories[1:]:
        if (incidence := k_nearest_categories.count(knc)) > knn_category_incidence:
            knn_category = knc
            knn_category_incidence = incidence

    # Return KNNDataPoint
    return KNNDataPoint(datapoint.x, datapoint.y, knn_category)
