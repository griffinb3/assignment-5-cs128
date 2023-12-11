

import random

from assignment_5_classifier import SubjectivityClassifier, SubjectivityFeatureSet


# Your existing classes and definitions go here

def main():
    # Load subjectivity corpus sentences
    sentences = list(subjectivity.sents(categories='subj'))
    sentence2 = list(subjectivity.sents(categories='obj'))
    feature_sets = [SubjectivityFeatureSet.build(sentence, 'subj', stopwords=('--', 'i', "'")) for sentence in sentences]
    feature_sets2 = [SubjectivityFeatureSet.build(sentence, 'obj', stopwords=('--', 'i', "'")) for sentence in sentence2]

    all_sets = feature_sets2 + feature_sets

    # Shuffle the sentences for randomness
    random.shuffle(all_sets)

    # Convert sentences to FeatureSet instances

    # Split the feature sets into training and testing sets
    training_set = all_sets[5000:]
    testing_set = all_sets[:1000]

    # Train the classifier
    classifier = SubjectivityClassifier.train(training_set)

    # Accuracy calculation
    correct_predictions = 0

    for feature_set in testing_set:
        result = classifier.gamma(feature_set)
        predicted_class = result.split(',')[0]
        if predicted_class == feature_set.clas:
            correct_predictions += 1

    total_predictions = len(testing_set)
    accuracy = (correct_predictions / total_predictions * 100)

    print(f"Accuracy: {accuracy:.2f}%")

    classifier.present_features(30)


if __name__ == "__main__":
    main()
