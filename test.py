from random import randrange
from matplotlib import pyplot

# Local
import knn
from kmeans import kmeans
from data import DataPoint


class Config:
    MAX_X = 100
    MAX_Y = 100
    LEN_DATA = 250

    # KNN categories along with the matplotlib color identifiers for the plot.
    # This dictionary's length equals the number of KNN categories.
    KNN_CATEGORY_COLOR = {1:'r', 2:'c', 3:'m'}

    # KMEANS Ks (centroids) along with the matplotlib color identifiers for the
    # plot. This dictionary's length equals the number of KMEANS's K.
    KMEANS_K_COLOR = {1:'r', 2:'c', 3:'m'}

    KMEANS_THRESHOLD = 0.0001


if __name__ == '__main__':
    # Initialize 200 random datapoints
    datapoints = [
        DataPoint(
            randrange(Config.MAX_X), randrange(Config.MAX_Y)
        ) for _ in range(Config.LEN_DATA)
    ]

    ## KNN
    print('KNN\n---\n')

    # Initialize 200 KNNDataPoint objects with random classes ranging from 1 to
    # Config.KNN_CLASSES
    knn_datapoints = [
        knn.KNNDataPoint(
            datapoints[i].x,
            datapoints[i].y,
            c = randrange(1, len(Config.KNN_CATEGORY_COLOR) + 1)
        ) for i in range(Config.LEN_DATA)
    ]

    # Initialize a random data point
    knn_datapoint = DataPoint(randrange(Config.MAX_Y), randrange(Config.MAX_Y))

    # Classify it using KNN
    knn_datapoint = knn.knn(
        knn_datapoints, knn_datapoint, knn.optimal_k(Config.LEN_DATA)
    )

    # Plot KNN data points emphasizing the one which was classified
    for knn_category in set(dp.category for dp in knn_datapoints):
        x = []
        y = []
        color = Config.KNN_CATEGORY_COLOR[knn_category]

        if knn_datapoint.category == knn_category:
            pyplot.scatter(
                knn_datapoint.x, knn_datapoint.y, c = color, s = 200, marker = '+'
            )

        for dp in [dp for dp in knn_datapoints if dp.category == knn_category]:
            x.append(dp.x)
            y.append(dp.y)

        pyplot.scatter(x, y, c = color, marker = 'o')

    print(f'KNN category: {knn_datapoint.category}\n\n')
    pyplot.show()
    
    ## K-Means
    print('K-Means\n-------\n')

    # Cluster data using K-Means with a K of Config.KMEANS_K and convergence
    # threshold of Config.KMEANS_THRESHOLD
    km_datapoints = kmeans(
        datapoints, len(Config.KMEANS_K_COLOR), Config.KMEANS_THRESHOLD
    )

    # Plot centroids and their corresponding data points
    for centroid in set(dp.centroid for dp in km_datapoints):
        x = []
        y = []
        color = Config.KMEANS_K_COLOR[centroid.id]

        pyplot.scatter(centroid.x, centroid.y, c = color, s = 200, marker = '+')

        for dp in [dp for dp in km_datapoints if dp.centroid == centroid]:
            x.append(dp.x)
            y.append(dp.y)

        pyplot.scatter(x, y, c = color, marker = 'o')

        print((
            f'Centroid {centroid.id} ({centroid.x:.1f}, {centroid.y:.1f}): '
            f'{len(x)} data points'
        ))

    pyplot.show()
