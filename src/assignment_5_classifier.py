"""Abstract data type definitions for a basic classifier."""

from __future__ import annotations
import math

from assignment_5_classifier_model import classifier
from assignment_5_classifier_model import *

__author__ = "Daniel Patterson, Griffin Brown, Mike Ryu"
__copyright__ = "Copyright 2023, Westmont College, Mike Ryu"
__credits__ = ["Mike Ryu"]
__license__ = "MIT"
__email__ = "dapatterson@westmont.edu, gbrown@westmont.edu, mryu@westmont.edu"


class SubjectivityFeature(Feature):
    """Feature used classification of an object.

    Attributes:
        _name (str): human-readable name of the feature (e.g., "over 65 years old")
        _value (str): machine-readable value of the feature (e.g., True)
    """


class SubjectivityFeatureSet(FeatureSet):
    """A set of features that represent a single object. Optionally includes the known class of the object.

    Attributes:
        _feat (set[Feature]): a set of features that define this object for the purposes of a classifier
        _clas (str | None): optional attribute set as the pre-defined classification of this object
    """

    @classmethod
    def build(cls, source_object: Any, known_clas=None, **kwargs) -> FeatureSet:
        stopwords = kwargs.get('stopwords', [])
        features = set()

        # This allows us to iterate over the split() doc and place each individual term into a dictionary
        for word in source_object:
            # Assuming each tuple has two elements: (feature_name, feature_value)
            if word not in stopwords:
                feature = Feature(name=word, value=True)
                features.add(feature)

        return cls(features=features, known_clas=known_clas)


class SubjectivityClassifier(AbstractClassifier):
    def __init__(self, feature_probabilities: dict):
        self.feature_probabilities = feature_probabilities
        #self.return_values = return_values
        #self.probability_list = probability_list

    def gamma(self, a_feature_set: FeatureSet) -> str:
        """Given a single feature set representing an object to be classified,
        returns the most probable class for the object based on the training
        this classifier received (via a call to `train` class method).

        :param a_feature_set: a single feature set representing an object to be classified
        :return: name of the class with the highest probability for the object
        """
        # Get the features from the feature set
        predicted_class = ''

        subj_prob = 0
        obj_prob = 0
        for feature in a_feature_set.feat:
            if feature not in self.feature_probabilities:
                pass
            elif feature in self.feature_probabilities:
                subj_accumulator = self.feature_probabilities[feature][0]
                subj_prob += math.log(subj_accumulator + 1)
                obj_accumulator = self.feature_probabilities[feature][1]
                obj_prob += math.log(obj_accumulator + 1)

        if subj_prob >= obj_prob:
            predicted_class = 'subj'
        elif obj_prob >= subj_prob:
            predicted_class = 'obj'

        return predicted_class

    def calc_present_features(self, top_n: int = 1) -> str:
        """Prints `top_n` feature(s) used by this classifier in the descending order of informativeness of the
        feature in determining a class for any object. Informativeness of a feature is a quantity that represents
        how "good" a feature is in determining the class for an object.

        :param top_n: how many of the top features to print; must be 1 or greater
        """
        probability_list = []

        for feature, probs in self.feature_probabilities.items():
            subj_prob = self.feature_probabilities[feature][0]
            obj_prob = self.feature_probabilities[feature][1]
            if subj_prob == 0 or obj_prob == 0:
                continue
            if subj_prob >= obj_prob:
                subj_ratio = subj_prob / obj_prob
                ratio = ("subj:obj", subj_ratio)
            elif obj_prob >= subj_prob:
                obj_ratio = obj_prob / subj_prob
                ratio = ("obj:subj", obj_ratio)
            probability_list.append((feature, ratio))

        sorted_features = sorted(probability_list, key=lambda item: item[1][1], reverse=True)

        top_results = sorted_features[:top_n]
        return_values = ''

        for feature, ratio in top_results:
            ratio_name, value = ratio
            return_values += f"{feature}: {ratio_name} = {value:.2f}:1\n"

        return return_values

    def present_features(self, top_n: int = 1) -> None:
        """For the sake of tests, we need to return an item in the function above.
        This function prints the function above"""
        print(self.calc_present_features(top_n))

    @classmethod
    def train(cls, training_set: Iterable[FeatureSet]) -> AbstractClassifier:
        # Initialize dictionaries to store counts
        # feature_counts is a nested dictionary where the outer keys are feature names,
        # and the values are counts for subj, and obj appearances

        feature_counts = {}
        return_values = []

        subj_count = 0
        obj_count = 0

        for feature_set in training_set:
            if feature_set.clas == 'subj':
                subj_count += 1
            elif feature_set.clas == 'obj':
                obj_count += 1

            for feature in feature_set.feat:
                if feature_counts.get(feature, 0) == 0:
                    feature_counts[feature] = [0, 0]  # First value is subjective, 2nd is objective
                if feature_set.clas == 'subj':
                    feature_counts[feature][0] += 1
                if feature_set.clas == 'obj':
                    feature_counts[feature][1] += 1

        if subj_count == 0:
            subj_count += .00001
        if obj_count == 0:
            obj_count += .00001

        for item in feature_counts.keys():
            feature_counts[item][0] /= subj_count
            feature_counts[item][1] /= obj_count

        return cls(feature_probabilities=feature_counts)
