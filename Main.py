# kode untuk menghubungkan dengan gdrive jika menggunakan google colab
from google.colab import drive
drive.mount('/content/drive')

# Berikut kode program untuk pengolahan data terlebih dahulu
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import random
%matplotlib inline

# model imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# processing imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

#Kode dibawah untuk membaca dataset
data_training20 = '/content/drive/MyDrive/Dataset/KDDTrain+_20Percent.txt'
data_training = '/content/drive/MyDrive/Dataset/KDDTrain+.txt'
data_test = '/content/drive/MyDrive/Dataset/KDDTest+.txt' 

#df = pd.read_csv(file_path_20_percent)
df = pd.read_csv(data_training)
test_df = pd.read_csv(data_test)

#Kode dibawah untuk pelabelan dataset
columns = (['duration'
,'protocol_type'
,'service'
,'flag'
,'src_bytes'
,'dst_bytes'
,'land'
,'wrong_fragment'
,'urgent'
,'hot'
,'num_failed_logins'
,'logged_in'
,'num_compromised'
,'root_shell'
,'su_attempted'
,'num_root'
,'num_file_creations'
,'num_shells'
,'num_access_files'
,'num_outbound_cmds'
,'is_host_login'
,'is_guest_login'
,'count'
,'srv_count'
,'serror_rate'
,'srv_serror_rate'
,'rerror_rate'
,'srv_rerror_rate'
,'same_srv_rate'
,'diff_srv_rate'
,'srv_diff_host_rate'
,'dst_host_count'
,'dst_host_srv_count'
,'dst_host_same_srv_rate'
,'dst_host_diff_srv_rate'
,'dst_host_same_src_port_rate'
,'dst_host_srv_diff_host_rate'
,'dst_host_serror_rate'
,'dst_host_srv_serror_rate'
,'dst_host_rerror_rate'
,'dst_host_srv_rerror_rate'
,'attack'
,'level'])

df.columns = columns
test_df.columns = columns

# sanity check
df.head()

# Membuat target 
# map normal to 0, all attacks to 1
is_attack = df.attack.map(lambda a: 0 if a == 'normal' else 1)
test_attack = test_df.attack.map(lambda a: 0 if a == 'normal' else 1)

#data_with_attack = df.join(is_attack, rsuffix='_flag')
df['attack_flag'] = is_attack
test_df['attack_flag'] = test_attack

# view the result
df.head()


# Melakukan set pelabelan
np.shape(df)
set(df['protocol_type'])
set(df['attack'])
set(df['service'])


#Membuat datfar mana yang serangan ddos
# lists to hold our attack classifications
dos_attacks = ['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm']
probe_attacks = ['ipsweep','mscan','nmap','portsweep','saint','satan']
privilege_attacks = ['buffer_overflow','loadmdoule','perl','ps','rootkit','sqlattack','xterm']
access_attacks = ['ftp_write','guess_passwd','http_tunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xclock','xsnoop']

# we will use these for plotting below
attack_labels = ['Normal','DoS','Probe','Privilege','Access']

# helper function to pass to data frame mapping
def map_attack(attack):
    if attack in dos_attacks:
        # dos_attacks map to 1
        attack_type = 1
    elif attack in probe_attacks:
        # probe_attacks mapt to 2
        attack_type = 2
    elif attack in privilege_attacks:
        # privilege escalation attacks map to 3
        attack_type = 3
    elif attack in access_attacks:
        # remote access attacks map to 4
        attack_type = 4
    else:
        # normal maps to 0
        attack_type = 0
        
    return attack_type

# map the data and join to the data set
attack_map = df.attack.apply(map_attack)
df['attack_map'] = attack_map

test_attack_map = test_df.attack.apply(map_attack)
test_df['attack_map'] = test_attack_map

# view the result
df.head()

#melakukan set lagi
set(df['attack_map'])

#Membuat model tampilan hasil output setelah dimasukkan model machine learning
comparePrecision=[]
compareRecall=[]
compareAccuracy=[]
compareF1=[]

# get the intial set of encoded features and encode them
features_to_encode = ['protocol_type', 'service', 'flag']
encoded = pd.get_dummies(df[features_to_encode])
test_encoded_base = pd.get_dummies(test_df[features_to_encode])

# not all of the features are in the test set, so we need to account for diffs
test_index = np.arange(len(test_df.index))
column_diffs = list(set(encoded.columns.values)-set(test_encoded_base.columns.values))

diff_df = pd.DataFrame(0, index=test_index, columns=column_diffs)

# we'll also need to reorder the columns to match, so let's get those
column_order = encoded.columns.to_list()

# append the new columns
test_encoded_temp = test_encoded_base.join(diff_df)

# reorder the columns
test_final = test_encoded_temp[column_order].fillna(0)

# get numeric features, we won't worry about encoding these at this point
numeric_features = ['duration', 'src_bytes', 'dst_bytes']

# model to fit/test
to_fit = encoded.join(df[numeric_features])
test_set = test_final.join(test_df[numeric_features])



#Membuat target clasifikasi
# create our target classifications
binary_y = df['attack_flag']
multi_y = df['attack_map']

test_binary_y = test_df['attack_flag']
test_multi_y = test_df['attack_map']

# build the training sets
binary_train_X, binary_val_X, binary_train_y, binary_val_y = train_test_split(to_fit, binary_y, test_size=0.6)
multi_train_X, multi_val_X, multi_train_y, multi_val_y = train_test_split(to_fit, multi_y, test_size = 0.6)


#Melihat info
binary_train_X.info()
binary_train_X.sample(5)

# Tampilan model
#Displaying Model Metrics 
def Report(model,y_pred):
    print('Summary Metrics')
    print('Precision score: {:.4f}'.format(precision_score(binary_val_y,y_pred)))
    comparePrecision.append(precision_score(binary_val_y,y_pred))
    print('Recall score: {:.4f}'.format(recall_score(binary_val_y,y_pred)))
    compareRecall.append(recall_score(binary_val_y,y_pred))
    print('Accuracy score: {:.4f}'.format(accuracy_score(binary_val_y,y_pred)))
    compareAccuracy.append(accuracy_score(binary_val_y,y_pred))
    print('F1 score: {:.8f}'.format(f1_score(binary_val_y,y_pred)))
    compareF1.append(f1_score(binary_val_y,y_pred))
    print('Classification Report')
    print(classification_report(binary_val_y, y_pred))
    print('Train score: ', model.score(binary_train_X, binary_train_y))
    print('Test score: ', model.score(binary_val_X, binary_val_y))
    print('Best Parameters:',model.best_params_)
    


# Model Machine Learning

# Model Machine Learning dengan algoritma random forest
# Buat model Random Forest
random_forest = RandomForestClassifier()
param_random = {'n_estimators': [3, 4], 'max_depth': [3, 6, None], 'bootstrap': [True, False]}

# Gunakan GridSearchCV untuk mencari parameter terbaik
binary_model = GridSearchCV(random_forest, param_random, cv=3, scoring='accuracy')
binary_model.fit(binary_train_X, binary_train_y)

# Prediksi pada data validasi menggunakan model terbaik
binary_predictions = binary_model.predict(binary_val_X)

# Tampilkan laporan klasifikasi
print(classification_report(binary_val_y, binary_predictions))

# Hitung akurasi pada data training dan data validasi
train_acc = accuracy_score(binary_train_y, binary_model.predict(binary_train_X))
test_acc = accuracy_score(binary_val_y, binary_predictions)

# Tampilkan akurasi pada data training dan data validasi
print('Training Accuracy:', train_acc)
print('Test Accuracy:', test_acc)
# Hitung akurasi pada keseluruhan data
overall_acc = accuracy_score(np.concatenate((binary_train_y, binary_val_y)),
                             np.concatenate((binary_model.predict(binary_train_X), binary_predictions)))

# Tampilkan akurasi keseluruhan
print('Accuracy:', overall_acc)





# Model Machine Learning dengan algoritma MLP (Multi Layer Peceptron)
mlp = MLPClassifier(solver='lbfgs', random_state=1)
params_mlp = {
    'alpha': [1e-5, 1e-3, 1e-1],
    'hidden_layer_sizes': [(5, 2), (10, 5), (20, 10)],
}

# Define GridSearchCV with 7-fold cross validation and accuracy as the scoring metric
grid_mlp = GridSearchCV(mlp, params_mlp, cv=7, scoring='accuracy')

# Fit the GridSearchCV object to the training data
grid_mlp.fit(binary_train_X, binary_train_y)

# Get the predicted labels for the test data using the best estimator found by GridSearchCV
y_pred_mlp = grid_mlp.best_estimator_.predict(binary_val_X)

# Generate a classification report for the MLPClassifier model
Report(grid_mlp, y_pred_mlp)




# Model Machine Learning dengan algoritma assemble Learning 
# define the list of models that we want to test
models = [
    RandomForestClassifier(),
    LogisticRegression(max_iter=250),
    KNeighborsClassifier(),
    MLPClassifier(solver='lbfgs', random_state=1),
]
grid_model = GridSearchCV(mlp, params_mlp, cv=7, scoring='accuracy')

# Fit the GridSearchCV object to the training data
grid_model.fit(binary_train_X, binary_train_y)

model_comps = []

for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, binary_train_X, binary_train_y, scoring='accuracy')
    for count, accuracy in enumerate(accuracies):
        model_comps.append((model_name, count, accuracy))

# calculate and display our base accuracty
Report(grid_model, binary_predictions)


# Model Machine Learning dengan algoritma SVM (Support vector Machine)
# Buat model SVM
svm_model = SVC()

# Definisikan parameter SVM yang akan dioptimasi
param_svm = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 'scale']}

# Gunakan GridSearchCV untuk mencari parameter terbaik
svm_grid = GridSearchCV(svm_model, param_svm, cv=3, scoring='accuracy')
svm_grid.fit(binary_train_X, binary_train_y)

# Hitung train score dan test score
train_score = svm_grid.best_estimator_.score(binary_train_X, binary_train_y)
test_score = svm_grid.best_estimator_.score(binary_val_X, binary_val_y)

# Prediksi pada data validasi menggunakan model terbaik
svm_predictions = svm_grid.best_estimator_.predict(binary_val_X)

# Tampilkan laporan klasifikasi
report = classification_report(binary_val_y, svm_predictions, output_dict=True)
accuracy = round(report['accuracy'], 4)

print(f'Train Score: {train_score:.4f}')
print(f'Test Score: {test_score:.4f}')
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(binary_val_y, svm_predictions))




# Model Machine Learning dengan algoritma KNN (K-Nearest Neighbor)
# Buat model KNN
knn_model = KNeighborsClassifier()

# Definisikan parameter KNN yang akan dioptimasi
param_knn = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'p': [1, 2]}

# Gunakan GridSearchCV untuk mencari parameter terbaik
knn_grid = GridSearchCV(knn_model, param_knn, cv=3, scoring='accuracy')
knn_grid.fit(binary_train_X, binary_train_y)

# Hitung train score, test score, dan akurasi
train_score = knn_grid.best_score_
test_score = knn_grid.score(binary_val_X, binary_val_y)
accuracy = (knn_grid.predict(binary_val_X) == binary_val_y).mean()

# Tampilkan laporan klasifikasi
print("Train score: {:.2f}".format(train_score))
print("Test score: {:.2f}".format(test_score))
print("Akurasi: {:.2f}".format(accuracy))
print(classification_report(binary_val_y, knn_predictions))



