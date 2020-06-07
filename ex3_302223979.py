import torch
import torch.nn as nn
from sklearn.model_selection import cross_val_score
import pandas as pd
import nltk
from nltk.corpus import stopwords
import datetime
from collections import defaultdict
import time
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import pickle

"""
Tomer Shahar - 302223979
The third assignment in NLP - deciding if a given tweet was written by Trump or not.
In this assignment I will use several ML classifiers to achieve this. 
"""

stopWords = set(stopwords.words('english'))

# Found on https://medium.com/swlh/analyzing-trumps-tweets-5368528d2c90
trump_top_words = {'great', 'trump', 'thank', 'just', 'people', 'obama', 'new', 'now', 'thanks', 'big', 'get', 'america'
    , 'time', 'make', 'good', 'country', 'can', 'many', 'president', 'today', 'one', 'never', 'going',
                   'hillary', 'barackobama', 'must', 'much', 'news', 'back', 'last', 'really', 'bad', 'think', 'china',
                   'see', 'job', 'tonight', 'want', 'even', 'new', 'crooked', 'fake', 'failing'}


def process_tweet_file(tsv_file_path, is_test=False):
    """
    receives a file path for a .tsv file containing tweets.
    :param is_test: Flag that indicates if we're parsing the training data or test data. Matters for classification
    :param tsv_file_path: The path to the .tsv file
    :return: a data frame made from the file
    """

    result = pd.DataFrame(columns=['tweet_id', 'handle', 'text', 'time_stamp', 'device'])
    with open(tsv_file_path, 'r') as tsv_file:
        for line in tsv_file:
            line_list = line.strip('\n').split('\t')
            if is_test:
                result = result.append(
                    {'tweet_id': 0, 'handle': line_list[0], 'text': line_list[1], 'time_stamp': line_list[2],
                     'device': None}, ignore_index=True)
            else:
                result = result.append(
                    {'tweet_id': line_list[0], 'handle': line_list[1], 'text': line_list[2], 'time_stamp': line_list[3],
                     'device': line_list[4]}, ignore_index=True)

    if not is_test:
        authorship = assign_authorship(result)
        result['class'] = authorship
        group_by_months(result)

    return result


def group_by_months(data):

    dates = {}

    for idx, row in data.iterrows():
        if row['device'] == 'android' or row['device'] == 'iphone':
            tweet_date = datetime.datetime.strptime(row['time_stamp'], '%Y-%m-%d %H:%M:%S')
            month = tweet_date.month
            year = tweet_date.year
            date_device = f'{month}-{year}'
            if date_device not in dates:
                dates[date_device] = {'android': 0, 'iphone': 0}
            dates[date_device][row['device']] += 1

    df = pd.DataFrame(columns=['Date', 'Android', 'Iphone'])
    for date, posts in dates.items():
        df = df.append({'Date': date, 'Android': posts['android'], 'Iphone': posts['iphone']}, ignore_index=True)

    df.to_csv('device_data.csv', index=False)


def assign_authorship(result):
    """
    Assigns whether each line is from Trump or not. If it was written through an iphone, it's definitely not Trump.
    If it was written with an android, we must dive further and see if it really is him or not.
    :param result: A dataframe of the tweets
    :return: A list containing 0 if it's trump, 1 if it isn't.
    """
    authorship = []
    trump = 0
    staff = 1
    # The day Trump became president and the POTUS account is associated with him
    inauguration_date = datetime.datetime.strptime('2017-01-20', '%Y-%m-%d')
    for idx, row in result.iterrows():
        tweet_date = datetime.datetime.strptime(row['time_stamp'], '%Y-%m-%d %H:%M:%S')
        if tweet_date < inauguration_date:  # Tweet was posted before he came the POTUS
            if row['handle'] != 'realDonaldTrump':  # Not his account, has to be staff
                authorship.append(staff)
            elif row['device'] == 'iphone':  # an iphone, can't be him
                authorship.append(staff)
            else:
                authorship.append(trump)
        else:  # He already became president
            if row['device'] != 'android':  # Not an android, can't be him
                authorship.append(staff)
            else:
                authorship.append(trump)
    return authorship


def vectorize_data(data):
    """
    A function that gets as input a dataframe of tweets and outputs
    :param data: a dataframe containing text
    :return: a dataframe where each row is a tweet and the columns are the vectors parameters.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    tf_idf_vectorizer = TfidfVectorizer(max_features=150)  # Play with this
    tweets = data['text'].tolist()
    bow = tf_idf_vectorizer.fit_transform(tweets)  # calculate the tf-idf matrix
    bow = bow.todense()
    words = tf_idf_vectorizer.get_feature_names()
    vectors = pd.DataFrame(bow, columns=words)

    return vectors


def encode_handle(data):
    """
    Assign each label a unique integer
    :param data: A single handle
    :return: An integer
    """
    if data == 'POTUS':
        return 0
    elif data == 'realDonaldTrump':
        return 1
    elif data == 'PressSec':
        return 2
    else:
        return 3


def encode_devices(device):
    """
    A function that converts the device to an integer
    :param device: the device data
    :return: an integer
    """
    if device == 'android':
        return 0
    elif device == 'iphone':
        return 1
    elif device == '<a href="http://twitter.com" rel="nofollow">Twitter Web Client</a>':
        return 2
    elif device == '<a href="http://instagram.com" rel="nofollow">Instagram</a>':
        return 3
    elif device == '<a href="http://www.twitter.com" rel="nofollow">Twitter for BlackBerry</a>':
        return 4
    elif device == '<a href="http://twitter.com/#!/download/ipad" rel="nofollow">Twitter for iPad</a>':
        return 5
    elif device == '<a href="https://about.twitter.com/products/tweetdeck" rel="nofollow">TweetDeck</a>':
        return 6
    elif device == '<a href="https://periscope.tv" rel="nofollow">Periscope.TV</a>':
        return 7
    elif device == '<a href="http://www.facebook.com/twitter" rel="nofollow">Facebook</a>':
        return 8
    else:
        return 9


def extract_time_data(raw_data):
    """
    parses the tweet time stamps in order to extract information from it. Returns a dataframe containing features, e.g.
    what day it was posted on, what time it was posted (if a tweet was posted in the middle of the night, it's usually
    Trump)
    :param raw_data: the input
    :return: A dataframe where each column is a feature about the time
    """

    hours = []
    is_weekend = []
    after_work = []
    result = pd.DataFrame(columns=['hour', 'is_weekend', 'after_work'])

    for idx, row in raw_data.iterrows():
        timestamp = datetime.datetime.strptime(row['time_stamp'], '%Y-%m-%d %H:%M:%S')
        hours.append(timestamp.hour)
        is_weekend.append(1 if timestamp.weekday() > 4 else 0)
        after_work.append(1 if timestamp.hour >= 22 or timestamp.hour <= 6 else 0)

    result['hour'] = hours
    result['is_weekend'] = is_weekend
    result['after_work'] = after_work
    return result


def get_text_data(text):
    """
    Receives the text of the tweet and counts various interesting parameters about the tweet.
    :param text: The text of the tweet
    :return: A dictionary mapping features to integers.
    """
    feature_dict = {'caps': 0, 'hashtags': 0, 'links': 0, 'exclamation': 0, 'top_words': 0}
    tweet_text = nltk.word_tokenize(text)
    feature_dict['quotation'] = 1 if text[0] == '\"' else 0  # tweet begins with quotes
    feature_dict['exclamation_end'] = 1 if text.split()[-1][-1] == '!' else 0  # tweet ends with exclamation

    for word in tweet_text:
        if word.lower() in stopWords:  # Skip over stopwords
            continue
        if word.isupper():
            feature_dict['caps'] += 1
        if word[0] == '#':
            feature_dict['hashtags'] += 1
        if word[0] == '!':
            feature_dict['exclamation'] += 1
        if 'https' in word:
            feature_dict['links'] += 1
        if word.lower() in trump_top_words:
            feature_dict['top_words'] += 1

    return feature_dict


def extract_text_data(raw_data):
    """
    Perhaps the most complex function here. This function extracts facts about the tweets text and converts it to
    integers. This uses information such as Trump's most used words, if the sentence ends in an exclamation mark, if it
    begins with a quotation mark etc.
    :param raw_data:
    :return:
    """
    result = pd.DataFrame()
    final_dict = defaultdict(list)
    for idx, row in raw_data.iterrows():
        text_dict = get_text_data(row['text'])
        for key, val in text_dict.items():
            final_dict[key].append(val)

    for key, final_val in final_dict.items():
        result[key] = final_val

    return result


def extract_data_features(raw_data):
    """
    This function receives the data and extracts the (interesting) features we choose, such us the first character in
    the tweet, the time, who posted it, if it ended with an exclamation mark etc.
    :param raw_data: A dataframe containing tweets
    :return: A new dataframe where each row corresponds to the same row in the input, but the columns are features.
    """
    handles = pd.DataFrame()
    handles['handle'] = raw_data['handle'].apply(encode_handle)
    time_data = extract_time_data(raw_data)
    language_data = extract_text_data(raw_data)
    result = pd.concat([handles, time_data, language_data], axis=1)

    return result


def create_feature_data_frame(data):
    """
    Converts the given dataframe into a dataframe that can be used by a ML algorithm.
    :param data: a dataframe where each row is a tweet that contains text, handle and time.
    :return: a dataframe containing features.
    """

    tf_idf_model = vectorize_data(data)
    meta_features = extract_data_features(data)
    final_clean_data = pd.concat([tf_idf_model, meta_features], axis=1)
    return final_clean_data


def normalize_dataframe(df):
    """
    Normalizes each column to hold values between 0 and 1.
    :param df: A dataframe that is not normalized
    :return: A normalized dataframe.
    """
    from sklearn import preprocessing

    min_max_scaler = preprocessing.MinMaxScaler()
    x = df.values
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=df.columns)
    return df


def logistic_regression_classifier(training_data, is_train=True):
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(solver='liblinear', random_state=0, penalty='l1')

    if is_train:
        x = training_data.drop(['class'], axis=1)
        y = training_data['class']
        start = time.time()
        scores = cross_val_score(lr, x, y, cv=10)
        end = time.time() - start
        lr.fit(x, y)
        pred = lr.predict(x)
        fpr, tpr, threshold = metrics.roc_curve(y, pred, pos_label=1)
        print(f'Logistic Regression Results - Accuracy: {np.mean(scores)}, AUC:{metrics.auc(fpr, tpr)} Time: {end}')

    return lr


def svc_classifier(training_data, linear=True):
    from sklearn.svm import SVC
    from sklearn.svm import LinearSVC

    if linear:
        svm = LinearSVC(max_iter=2000, random_state=0)
    else:
        svm = SVC(gamma='auto', random_state=0, max_iter=-1)

    x = training_data.drop(['class'], axis=1)
    if linear:
        x = normalize_dataframe(x)  # Otherwise we'll get convergence warnings
    y = training_data['class']
    start = time.time()
    scores = cross_val_score(svm, x, y, cv=10)

    end = time.time() - start
    mode = 'Linear' if linear else 'Non Linear'
    svm.fit(x, y)
    pred = svm.predict(x)
    fpr, tpr, threshold = metrics.roc_curve(y, pred, pos_label=1)
    print(f'{mode} SVM Results - Accuracy: {np.mean(scores)}, AUC:{metrics.auc(fpr, tpr)} Time: {end}')

    return svm


def neural_net_classifier(training_data):
    """
    A regular feed-forward neural network using the pytorch library.
    Uses k-cross validation
    :param training_data: A dataframe containing the training data.
    :return: the model
    """

    scores = []
    epoch = 100
    start = time.time()
    net = None
    net, auc = nn_k_fold(epoch, net, scores, training_data)
    end = time.time() - start

    print(f'Neural Net Results - Accuracy: {np.mean(scores)}, AUC:{auc} Time: {end}')

    return net


def random_forest_classifier(training_data):
    """
    Uses the popular ensemble decision tree classifier random forest.
    :param training_data: The training data
    :return: The fitted classifier
    """

    from sklearn.ensemble import RandomForestClassifier

    x = training_data.drop(['class'], axis=1)
    y = training_data['class']
    clf = RandomForestClassifier(max_depth=25, random_state=0, n_estimators=75, min_samples_leaf=1,
                                 min_samples_split=5, criterion='gini')
    start = time.time()
    scores = cross_val_score(clf, x, y, cv=10)
    end = time.time() - start

    clf.fit(x, y)
    pred = clf.predict(x)
    fpr, tpr, threshold = metrics.roc_curve(y, pred, pos_label=1)
    print(f'Random Forest Results - Accuracy: {np.mean(scores)}, AUC:{metrics.auc(fpr, tpr)} Time: {end}')

    return clf


def nn_k_fold(epoch, net, scores, training_data):
    """
    Performs k-cross validation on the neural net we will build to train and evaluate it. Currently using k=10
    :param epoch: Number of training iterations
    :param net: The Neural Net object
    :param scores: A list of scores from each single fold
    :param training_data: A dataframe of the training data/
    :return: The trained Neural Net classifier.
    """

    from sklearn.model_selection import KFold
    k_folds = KFold(n_splits=10)
    hidden = 5
    output = 1
    learning_rate = 0.01
    x_train = training_data.drop(['class'], axis=1)
    y_train = training_data['class']
    for train_idx, val_idx in k_folds.split(x_train):
        x_t_temp, x_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
        input_dimensions = x_t_temp.shape[1]
        net = nn.Sequential(nn.Linear(input_dimensions, hidden), nn.ELU(), nn.Linear(hidden, output), nn.Sigmoid())

        y_train_split, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        y_train_temp = torch.FloatTensor(y_train_split.values).reshape(-1, 1)
        train_nn(epoch, net, learning_rate, x_t_temp, y_train_temp)

        X_val_t = torch.FloatTensor(x_val.values)
        y_hat_test = net(X_val_t)
        y_hat_test_class = np.where(y_hat_test.detach().numpy() < 0.5, 0, 1)
        val_accuracy = np.sum(y_val.values.reshape(-1, 1) == y_hat_test_class) / len(y_val)
        scores.append(val_accuracy)

    x_tensor = torch.FloatTensor(x_train.values)
    y_pred = net(x_tensor)
    pred = np.where(y_pred.detach().numpy() < 0.5, 0, 1)

    fpr, tpr, threshold = metrics.roc_curve(y_train, pred, pos_label=1)
    return net, metrics.auc(fpr, tpr)


def train_nn(epoch, net, learning_rate, x_train, y_train):
    """
    Function for training our neural network.
    :param epoch: How many training cycles we will do
    :param net: Neural net object
    :param learning_rate: Optimizing function. Uses the Adam algorithm.
    :param x_train: A Dataframe of tweets. Used as training
    :param y_train: A Dataframe of their classifications.
    :return: Updates the net object
    """
    learning_rate = torch.optim.Adam(net.parameters(), lr=learning_rate)

    for i in range(epoch):
        x_train_temp = torch.FloatTensor(x_train.values)
        loss_func = nn.BCELoss()
        y_hat = net(x_train_temp)
        loss = loss_func(y_hat, y_train)
        loss.backward()
        learning_rate.step()
        learning_rate.zero_grad()


def save_model(model):

    with open('best_model.pkl', 'wb') as model_file:
        pickle.dump(obj=model, file=model_file)


def load_best_model():
    """
    Returns the best model
    :return: The best model found in the training phase: Random Forest
    """
    with open('best_model.pkl', 'rb') as model_file:
        m = pickle.load(file=model_file)
        return m


def train_best_model():
    """
    Trains the classifier from scratch
    :return: The trained model
    """

    train_path = './trump_train.tsv'
    train = process_tweet_file(train_path, is_test=False)
    train_features = create_feature_data_frame(train)
    train_features['class'] = train['class']
    x = train_features.drop(['class'], axis=1)
    y = train_features['class']
    clf = RandomForestClassifier(max_depth=25, random_state=0, n_estimators=75, min_samples_leaf=1,
                                 min_samples_split=5, criterion='gini')
    clf.fit(x, y)

    return clf


def predict(m, fn):
    """
    Uses a trained model to predict based on the data given in fn.
    :param m: The trained model
    :param fn: full path to a file
    :return: A list of 1s and 0s where a 0 will be places if the tweet was classified as Trump, 0 otherwise.
    """

    test_tweets = process_tweet_file(fn, is_test=True)
    test_features = create_feature_data_frame(test_tweets)

    res = m.predict(test_features)
    return res.tolist()


def save_prediction(prediction):
    """
    Saves to a txt file the prediction given
    :param prediction: A list of 0s and 1s (ints)
    :return: Writes to a file the results.
    """
    for idx, val in enumerate(prediction):
        prediction[idx] = str(val)

    with open('./302223979.txt', 'w') as res_file:
        text = ' '.join(prediction)
        res_file.write(text)
