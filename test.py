from random import randrange
from matplotlib import pyplot

# Local
from kmeans import kmeans
from data import DataPoint


class Config:
    MAX_X = 100
    MAX_Y = 100
    LEN_DATA = 250

    KMEANS_K = 3
    KMEANS_THRESHOLD = 0.0001


if __name__ == '__main__':
    # Initialize 200 random datapoints
    datapoints = [
        DataPoint(
            randrange(Config.MAX_Y), randrange(Config.MAX_Y)
        ) for _ in range(Config.LEN_DATA)
    ]

    ## K-Means
    print('K-Means\n-------\n')

    # Cluster data using K-Means with a K of Config.KMEANS_K and convergence
    # threshold of Config.KMEANS_THRESHOLD
    km_datapoints = kmeans(datapoints, Config.KMEANS_K, Config.KMEANS_THRESHOLD)

    # Plot centroids and their corresponding data points
    for centroid, color in zip(set(dp.centroid for dp in km_datapoints), 'rcm'):
        x = []
        y = []

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
