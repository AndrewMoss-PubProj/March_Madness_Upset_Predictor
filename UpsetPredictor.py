from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
import pandas as pd
import math
from sklearn.metrics import precision_score
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
from itertools import repeat
import matplotlib.pyplot as plt
import seaborn as sns




path = input("Enter file path: ")

def extract(path):

    df = pd.read_csv(path)

    return df

def LOOCV(df):
    NBprecisions = []
    KNNprecisions =[[]for i in repeat(None,50)]
    TreePrecisions =[]
    ForestPrecisions = []
    KNNavg = []

    NBconf = []
    Treeconf = []
    Forestconf = []
    KNNconf =[[]for i in repeat(None,50)]
    KNNavgconf =[]

    df = shuffle(df)
    labs = df.iloc[:,22].values.tolist()

    num_folds = df.shape[0]
    subset_size = math.floor(len(df)/num_folds)
    data = df.values.tolist()
    for x in range(1):
        for i in range(num_folds):
            testing_this_round = data[i * subset_size:][:subset_size]
            training_this_round = data[:i * subset_size] + data[(i + 1) * subset_size:]
            dfTrain = pd.DataFrame(training_this_round)
            dfTest= pd.DataFrame(testing_this_round)
            train_x = dfTrain.iloc[:, 2:22]
            train_y = dfTrain.iloc[:, 22]
            test_x = dfTest.iloc[:, 2:22]
            test_y = dfTest.iloc[:, 22]


            nbVal, nbBool,NBpred = nb(train_x, train_y, test_x, test_y)
            NBconf.append(NBpred)
            if nbBool == True:
                NBprecisions.append(nbVal)
            treeVal, treeBool,Treepred = DecisionTree(train_x, train_y, test_x, test_y)
            Treeconf.append(Treepred)
            if treeBool == True:
                TreePrecisions.append(treeVal)
            for k in range(1,51):
                kVal, kBool,KNNpred = knn(train_x, train_y, test_x, test_y,k)
                KNNconf[k-1].append(KNNpred)
                if kBool == True:
                    KNNprecisions[k-1].append(kVal)
            forestVal, forestBool,Forestpred = randForest(train_x, train_y, test_x, test_y)
            Forestconf.append(Forestpred)
            if forestBool == True:
                ForestPrecisions.append(forestVal)

    for i in range (0, len(KNNprecisions)):
        KNNavg.append(np.mean(KNNprecisions[i]))
    xlist = list(range(0,50))
    makeKNNgraph(xlist,KNNavg,NBprecisions,TreePrecisions,ForestPrecisions)

    bestVal = max(KNNavg)
    KNNavgconf = KNNconf[int(bestVal)]
    bestK = KNNavg.index(max(KNNavg)) + 1

    dfConf = makeConfidenceMap(NBconf, Treeconf, Forestconf, KNNavgconf)

    print("\nIn reality, there were 53 upsets out of 207 total games \n")
    print("Naive Bayes Precision: " + str(round(np.mean(NBprecisions)*100,2))+"%" + " " + "on " +  str(len(NBprecisions))+" predicted upsets")
    print("Decision Tree Precision: " + str(round(np.mean(TreePrecisions)*100,2)) + "%" + " " + "on " + str(len(TreePrecisions)) + " predicted upsets")
    print("KNN Precision: " + str(round(np.mean(bestVal)*100,2)) + "% for k = " + str(bestK)+ " " + "on " + str(len(KNNprecisions))+ " predicted upsets")
    print("Random Forest Precision: " + str(round(np.mean(ForestPrecisions)*100,2)) + "%"+ " " + "on " + str(len(ForestPrecisions))+ " predicted upsets")
    print()

    results = ensemble(dfConf,labs)

    print(str(round(np.mean(results[4])*100,2))+ "%  of the games that 4 models agreed on were upsets. There were " + str(len(results[4])) + " of this type")
    print(str(round(np.mean(results[3])*100,2))+ "%  of the games that 3 models agreed on were upsets. There were " + str(len(results[3])) + " of this type")
    print(str(round(np.mean(results[2])*100,2))+ "%  of the games that 2 models agreed on were upsets. There were " + str(len(results[2])) + " of this type")
    print(str(round(np.mean(results[1])*100,2))+ "%  of the games that 1 models agreed on were upsets. There were " + str(len(results[1])) + " of this type")
    print("There were " + str(len(results[0])) + " games  that were not predicted as an upset by any model")


def makeKNNgraph(xlist,KNNavg,NBprecisions,TreePrecisions,ForestPrecisions):
    plt.style.use('ggplot')
    sns.lineplot(x=xlist, y=KNNavg)
    plt.axhline(y=np.mean(NBprecisions), color = 'r')
    plt.axhline(y=np.mean(TreePrecisions), color = 'g')
    plt.axhline(y=np.mean(ForestPrecisions), color = 'b')
    plt.legend(['KNN','Naive Bayes','Decision Tree', 'Random Forest'])

    plt.xlabel('k value')
    plt.ylabel('Precision')
    plt.title('KNN precision varied by K value')
    plt.xlim(0, 50)

    #plt.show()
    plt.savefig("KNNPlot.png")
    plt.clf()


def makeConfidenceMap(NBconf, Treeconf, Forestconf, KNNavgconf):
    lengthAll = len(NBconf)
    if lengthAll != len(Treeconf) or lengthAll != len(Forestconf) or lengthAll != len(KNNavgconf):
        print("Lengths for lists are not equal. There must be an error.")
        return None
    confArr = []
    for i in range(lengthAll):
        confArr.append(NBconf[i] + Treeconf[i] + Forestconf[i] + KNNavgconf[i])
    unsorted_conf = []
    for x in range(len(confArr)):
        unsorted_conf.append(confArr[x])
    confArr.sort(reverse=True)
    xlist =list(range(0,lengthAll))
    sns.barplot(x=xlist,y=confArr)
    plt.title("Distribution of Model Agreement")
    plt.ylabel("Number of models in agreement")
    plt.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)
    plt.savefig("confidence_chart.png")
    #plt.show()

    return unsorted_conf

def nb(train_x, train_y, test_x, test_y):
    gnb = GaussianNB(priors=[.75,.25])
    gnb.fit(train_x, train_y)
    if (gnb.predict(test_x) == 0):
        return 0, False,0
    score = precision_score(test_y, gnb.predict(test_x), pos_label=1)
    return score, True, 1

def DecisionTree(train_x, train_y, test_x, test_y):
    tree = DecisionTreeClassifier()
    tree.fit(train_x, train_y)
    if (tree.predict(test_x) == 0):
        return 0, False,0
    score = precision_score(test_y, tree.predict(test_x), pos_label=1)
    return score, True,1

def knn(train_x, train_y, test_x, test_y,k):
    nn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    nn.fit(train_x, train_y)
    if(nn.predict(test_x) == 0):
        return 0, False,0
    score = precision_score(test_y, nn.predict(test_x), pos_label=1,)
    return score, True, 1

def randForest(train_x, train_y, test_x, test_y):
    randF = RandomForestClassifier(n_estimators=100)
    randF.fit(train_x, train_y)
    if (randF.predict(test_x) == 0):
        return 0, False,0
    score = precision_score(test_y, randF.predict(test_x), pos_label=1)
    return score, True, 1
def ensemble(dfconf,labs):
    ensemble_results = [[]for i in repeat(None,5)]
    x = 0
    for entry in dfconf:
        if entry == 0:
            ensemble_results[0].append(0)
        elif entry == 1:
            if labs[x] == 1:
                ensemble_results[1].append(1)
            elif labs[x] == 0:
                ensemble_results[1].append(0)
        elif entry == 2:
            if labs[x] == 1:
                ensemble_results[2].append(1)
            elif labs[x] == 0:
                ensemble_results[2].append(0)
        elif entry == 3:
            if labs[x] == 1:
                ensemble_results[3].append(1)
            elif labs[x] == 0:
                ensemble_results[3].append(0)
        elif entry == 4:
            if labs[x] == 1:
                ensemble_results[4].append(1)
            elif labs[x] == 0:
                ensemble_results[4].append(0)
        x=x+1

    return ensemble_results


warnings.simplefilter("ignore")
LOOCV(extract(path))

