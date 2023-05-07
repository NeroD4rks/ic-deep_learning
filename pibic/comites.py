from image_handler import OneNN


def instanceElection(votes, distances):
    result_dic = {}
    majority_dic = {}  # stores class:votes
    weighted_dic = {}  # stores class:weighted_sum
    for v_index in range(len(votes)):
        vote = votes[v_index]  # vote under consideration
        majority_dic[vote] = majority_dic[vote] + 1 if (vote in majority_dic) else 1  # adds one to the class
        weighted_dic[vote] = (weighted_dic[vote] + (1 / distances[v_index])) if (vote in weighted_dic) else (
                    1 / distances[v_index])  # sums 1/distance to the total of the class
    result_dic["Majority Voting"] = max(majority_dic, key=majority_dic.get)  # gets the class with more votes
    result_dic["Weighted Voting"] = max(weighted_dic, key=weighted_dic.get)  # gets the class with heighest value
    result_dic["Distance Voting"] = votes[distances.index(min(distances))]  # gets the class with the smallest distance
    return result_dic


def scoresPerDataset(CODECs, X_test, y_test, X_train, y_train, Dataset):
    numberOfTestsInstances = len(y_test)
    finalResults = {}
    finalResults["Dataset"] = Dataset
    finalResults["Majority Voting"] = 0  # total hits of the majority ensemble
    finalResults["Weighted Voting"] = 0  # total hits of the weighted ensemble
    finalResults["Distance Voting"] = 0  # total hits of the distance ensemble
    for codec in CODECs:
        finalResults[codec] = 0  # total hits for each codec

    for instance_index in range(len(X_test)):  # Iterates over each test_datasets instance
        testInstance = X_test[instance_index]  # image path of the test_datasets instance
        testInstanceClass = y_test[instance_index]  # class of the image
        votes = []  # the votes of each classifier for the testInstance
        distances = []  # the min_distance returned by each classifier
        for codec in CODECs:  # Iterates over each CODEC
            print(codec, instance_index, numberOfTestsInstances, Dataset)
            codec_result = OneNN(X_train, y_train, testInstance, codec)
            codec_prediction = codec_result[0]  # classe obtida pelo classsificador
            codec_min_distance = codec_result[1]  # distancia da classe obtida da imagem de teste
            votes.append(codec_prediction)  # builds the vote list for the ensembles
            distances.append(codec_min_distance)  # builds distance list for the ensembles
            if codec_prediction == testInstanceClass:  # checks if the codec prediction is a hit
                finalResults[codec] += 1
        enseble_results = instanceElection(votes, distances)  # gets the ensembles' prediction
        for result in enseble_results:  # checks the hits for the ensembles
            if enseble_results[result] == testInstanceClass:
                finalResults[result] += 1
    for clf in finalResults:  # computes the accuracy for each classifier
        if clf != "Dataset":
            finalResults[clf] = finalResults[clf] / numberOfTestsInstances
    print(finalResults)
    return finalResults
