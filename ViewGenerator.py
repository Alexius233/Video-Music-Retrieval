import numpy as np
np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, Transforms, Transforms_fine, n_views=2):
        self.transform = Transforms
        self.transform_fine = Transforms_fine
        self.n_views = n_views

    def __call__(self, x):

        if self.n_views == 2:
            y = x[0::8]
            return self.transform(x), self.transform_fine(y)

        if self.n_views == 1:
            return self.transform_fine(x)