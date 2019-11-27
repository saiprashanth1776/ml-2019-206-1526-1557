from warnings import filterwarnings
from pandas import read_csv, cut, DataFrame
from numpy import array, unique, log2, inf, append, where, square, random, concatenate
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import time
from random import randrange
from sklearn.metrics import classification_report

filterwarnings("ignore")

file_path = '/home/pramod/5th_sem/ML/MiniProject1_SectionE_G/catalog1/cat1.csv'
df = read_csv(file_path)
df.head()
y = array(df['class'])
# df = df.rename(columns={"class": "label"})
if 'cat2.csv' in file_path:
    df.drop("Unnamed: 0.1", axis=1, inplace=True)
df.drop(["Unnamed: 0", "galex_objid", "sdss_objid", "pred"], axis=1, inplace=True)


def bucketize(dataframe, col_headers, bucket_size):
    assert len(col_headers) == len(bucket_size)
    no_of_columns = len(col_headers)
    for col in range(no_of_columns):
        labels = array([(x + 1) for x in range(bucket_size[col])])
        temp = cut(dataframe[col_headers[col]], bucket_size[col], labels=labels)
        dataframe.drop(col_headers[col], inplace=True, axis=1)
        dataframe[col_headers[col]] = temp
    return dataframe

y = df['class']
redshift = array(df['spectrometric_redshift'])
X = df.drop(['class', 'spectrometric_redshift'], axis=1)
df_new = bucketize(X, X.columns, [10 for x in range(len(X.columns))])
df_new['class'] = y.values


train, test, redshift_train, redshift_test = train_test_split(df_new, redshift, test_size=0.3, random_state=7)


def over_sampling(df):
    df_0 = df[df['class'] == 0]
    df_1 = df[df['class'] == 1]
    
    length_1 = len(df_1)
    length_0 = len(df_0)
    for i in range(length_1):
        index = randrange(length_0)
        df_1 = df_1.append(df_0.iloc[index])
    return df_1

train = over_sampling(train)


# extracting class and bucketizing the train data
y_train = array(train['class'])
train.drop('class', inplace=True, axis=1)
X_train = array(train)

# extracting class and bucketizing the test data
y_test = array(test['class'])
test.drop('class', inplace=True, axis=1)
X_test = array(test)


class DecisionTree:
    
    def __init__(self, max_depth=5, min_samples=2):
        self.counter = 0
        self.max_depth = max_depth
        self.min_samples = min_samples
        
    def check_purity(self, y):
        return True if len(unique(y)) == 1 else False
        
    def classify(self, y):
        unique_classes, counts_unique_classes = unique(y, return_counts=True)

        index = where(counts_unique_classes == max(counts_unique_classes))[0][0]
        
        return unique_classes[index]
    
    def get_potential_splits(self, X):
    
        potential_splits = {}
        n_columns = len(X[0])
        for column_index in range(n_columns):  
            potential_splits[column_index] = set()
            values = X[:, column_index]
            unique_values = unique(values)

            for index in range(1, len(unique_values)):
                current_value = unique_values[index]
                previous_value = unique_values[index - 1]
                potential_split = (current_value + previous_value) / 2

                potential_splits[column_index].add(potential_split)

        return potential_splits
    
    def split_data(self, X, y, split_column, split_value):
    
        no_of_columns = len(X[0]) + 1
        split_column_values = X[:, split_column]
        data = append(X, y.reshape(len(X), 1), axis=1)
        data_below = data[data[:, split_column] <= split_value]
        data_above = data[data[:, split_column] > split_value]

        return data_below, data_above
    
    def gini(self, label_column):

        labels, counts = unique(label_column, return_counts=True)
        probabilities = counts / sum(counts)
        return 1 - sum(square(probabilities))
    
    def overall_gini(self, data_below, data_above):
    
        p = len(data_below) / (len(data_below) + len(data_above))

        overall_entropy =  (p * self.gini(data_below[:, -1]) 
                          + (1 - p) * self.gini(data_above[:, -1]))

        return overall_entropy
    
    def determine_best_split(self, X, y, potential_splits):    
        overall_gini = inf
        for column_index in potential_splits:
            for value in potential_splits[column_index]:
                data_below, data_above = self.split_data(X, y, split_column=column_index, split_value=value)
                current_overall_gini = self.overall_gini(data_below, data_above)

                if current_overall_gini <= overall_gini:
                    overall_gini = current_overall_gini
                    best_split_column = column_index
                    best_split_value = value

        return best_split_column, best_split_value
    
    def build_tree(self, X, y):
        if (self.check_purity(y)) or (len(X) < self.min_samples) or (self.counter == self.max_depth):
            return self.classify(y)

        else:    
            self.counter += 1

            potential_splits = self.get_potential_splits(X)
            split_column, split_value = self.determine_best_split(X, y, potential_splits)
            data_below, data_above = self.split_data(X, y, split_column, split_value)

            question_at_node = "column_{} <= {}".format(split_column, split_value)
            sub_tree = {question_at_node: []}

            true = self.build_tree(data_below[:, :-1], data_below[:, -1])
            false = self.build_tree(data_above[:, :-1], data_above[:, -1])

            if true == false:
                sub_tree = true
            else:
                sub_tree[question_at_node].append(true)
                sub_tree[question_at_node].append(false)

            return sub_tree
    
    def fit(self, X, y):
        
        start_time = time.time()
        tree = self.build_tree(X, y)
        end_time = time.time()
        print("Time taken to construct the decision tree =", end_time - start_time)
        return tree
    
    def classify_example(self, example, tree):
        question_at_node = list(tree.keys())[0]
        feature_name, comparison_operator, value = question_at_node.split(" ")
        x = int(feature_name.split("_")[1])
        
        current_answer = tree[question_at_node][0] if (example[x] <= float(value)) else tree[question_at_node][1]
        
        if type(current_answer) == dict:
            residual_tree = current_answer
            return self.classify_example(example, residual_tree)

        else:
            return current_answer
    
    def predict(self, X_test, tree):
        predictions = array([])
        for example in X_test:
            predictions = append(predictions, self.classify_example(example, tree))

        return predictions
    
    def comp_with_redshift(self, redshift, y_test, y_pred):
        correct_count = 0
        length = len(y_test)
        z_1_test = z_2_test = z_3_test = array([])
        z_1_pred = z_2_pred = z_3_pred = array([])
        for index in range(length):
            if redshift[index] <= 0.0033:
                z_1_test = append(z_1_test, y_test[index])
                z_1_pred = append(z_1_pred, y_pred[index])
            elif redshift[index] >= 0.0033 and redshift[index] <= 0.004:
                z_2_test = append(z_2_test, y_test[index])
                z_2_pred = append(z_2_pred, y_pred[index])
            else:
                z_3_test = append(z_3_test, y_test[index])
                z_3_pred = append(z_3_pred, y_pred[index])

        print("Accuracy of range 1:", accuracy_score(z_1_test, z_1_pred), " len = ", len(z_1_test))
        print("Accuracy of range 2:", accuracy_score(z_2_test, z_2_pred), " len = ", len(z_2_test))
        print("Accuracy of range 3:", accuracy_score(z_3_test, z_3_pred), " len = ", len(z_3_test))


tree = DecisionTree(max_depth=5)

decision_tree = tree.fit(X_train, y_train)
predictions = tree.predict(X_test, decision_tree)

print(classification_report(y_test, predictions))
tree.comp_with_redshift(redshift_test, y_test, predictions)