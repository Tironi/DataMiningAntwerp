
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
  
def importdata():
    balance_data = pd.read_csv('existing-customers.csv', sep= ';')
    balance_data = balance_data.dropna()
    return balance_data

def metrics(y_test, y_pred):
    accuracy_val = int(accuracy_score(y_test, y_pred)*100)
    precision_val = int(precision_score(y_test, y_pred, pos_label=">50K")*100)
    recall_val = int(recall_score(y_test, y_pred, pos_label=">50K")*100)
    f1_val = int(f1_score(y_test, y_pred, pos_label=">50K")*100)
    print("Accuracy: " + str(accuracy_val))
    print("Precision: " + str(precision_val))
    print("Recall: " + str(recall_val))
    print("F1: " + str(f1_val))
    print(confusion_matrix(y_test, y_pred, labels=[">50K", "<=50K"]))
    print("\n")

def train_using_gini(X_train, y_train):
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, min_samples_leaf=5)
    clf_gini.fit(X_train, y_train)
    return clf_gini
      
def prediction(X, clf_object):
    y_pred = clf_object.predict(X)
    return y_pred

def encode(df, encoder, columns_to_encode):
    encoded_data = encoder.transform(df[columns_to_encode])
    encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(columns_to_encode))
    df.drop(columns=columns_to_encode, inplace=True)
    df.reset_index(drop=True, inplace=True)
    encoded_df.reset_index(drop=True, inplace=True)
    df_encoded = pd.concat([df, encoded_df], axis=1)

    columns_to_encode = ['sex']
    encoder = LabelEncoder()
    for col in columns_to_encode:
        df_encoded[col] = encoder.fit_transform(df_encoded[col])

    return df_encoded

def testing(clf_gini, encoder):
    balance_data = pd.read_csv('potential-customers.csv', sep= ';')
    balance_data = balance_data.dropna()
    balance_data = balance_data.drop(['RowID', 'education'], axis=1)

    columns_to_encode = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
    df_encoded = encode(balance_data, encoder, columns_to_encode)
    X = df_encoded

    y_pred_gini = prediction(X, clf_gini)
    balance_data['class'] = y_pred_gini
    balance_data.to_csv('output_dt.csv', index=False)

    return balance_data
  
def main():
    #IMPORT DATA
    data = importdata()
    data = data.drop(['RowID', 'education'], axis=1)
    df = pd.DataFrame(data)

    #ENCODE DATA
    encoder = OneHotEncoder()
    columns_to_encode = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
    encoder.fit(df[columns_to_encode])
    df_encoded = encode(df, encoder, columns_to_encode)

    #DIVIDE X and Y - train and test
    X = df_encoded.drop(['class'], axis=1)
    y = df_encoded['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    #TRAINING
    clf_gini = train_using_gini(X_train, y_train)

    #VALIDATION
    y_pred_val = prediction(X_val, clf_gini)
    print("Validation")
    metrics(y_val, y_pred_val)
    print("\n")

    #TESTING
    y_pred_test = prediction(X_test, clf_gini)
    print("Testing")
    metrics(y_test, y_pred_test)
    precision_test = int(precision_score(y_test, y_pred_test, pos_label=">50K")*100)
    recall_test = int(recall_score(y_test, y_pred_test, pos_label=">50K")*100)

    #CALCULATION
    out = testing(clf_gini, encoder)
    out_high = out[out['class'].str.contains('>50K')]
    out_low = out[out['class'].str.contains('<=50K')]
    
    num_predict_high = len(out_high.index)
    num_predict_low = len(out_low.index)
    print("Number of predicted customer over 50k: " + str(num_predict_high))
    print("Number of predicted customer under 50k: " + str(num_predict_low))

    true_high = num_predict_high * precision_test / 100
    false_high = num_predict_high * (100-precision_test) / 100
    
    acp_customers_high = true_high * 10 / 100
    acp_customers_low = false_high * 5 / 100
    
    total_revenue = acp_customers_high * 980
    print("REVENUE: " + str(int(total_revenue)) + "€")
    print("CUSTOMER UNDER 50k COST: " + str(int(acp_customers_low * 310)) + "€")
    print("PROFIT: " + str(int(total_revenue - acp_customers_low * 310))+ "€")
    print("\n")
    print("MAILING COST: " + str(int(10 * num_predict_high)) + "€")
    total_cost = acp_customers_low * 310 + 10 * num_predict_high

    expected = total_revenue - total_cost
    print("Total expected: " + str(int(expected)) + " €")

if __name__=="__main__":
    main()