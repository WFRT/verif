import sys
import inspect


class MetricType(object):
    def __eq__(self, other):
        return self.__class__ == other.__class__

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def name(cls):
        name = cls.__name__
        return name


class Deterministic(MetricType):
    description = "Deterministic"
    pass


class Probabilistic(MetricType):
    description = "Probabilistic"
    pass


class Threshold(MetricType):
    description = "Threshold"
    pass


class Diagram(MetricType):
    description = "Special diagrams"
    pass
