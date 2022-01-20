import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

col = ['Tile','Target']
fully_annotated = pd.read_csv('.\\data\\train_input\\train_tile_annotations.csv', names = col, skiprows=1)
fully_annotated['ID'] = fully_annotated['Tile'].apply(lambda x:x[:16])

def clean_key(tile):
    my_string = tile.split('.')[0].split('_')
    del my_string[4]
    return '_'.join(my_string)
fully_annotated['myIndex'] = fully_annotated['Tile'].apply(lambda x:clean_key(x))
images_fully_annotated = fully_annotated.ID.unique().tolist()
images_fully_annotated = ["{}{}".format(i,'.npy') for i in images_fully_annotated]

dataList = []
cols = ['x', 'y', 'z'] + list(range(2048))
for x in os.listdir('./train_input/resnet_features'):

    if x in images_fully_annotated:
        # Prints only text file present in My Folder

        ResNet50_features = np.load('./train_input/resnet_features/' + x)
        ResNet50_features = pd.DataFrame(ResNet50_features)
        ResNet50_features.columns = cols
        #         ResNet50_features['Index'] = ResNet50_features.apply(lambda x: )
        ResNet50_features['ID'] = x[:16] + '_tile'
        ResNet50_features['x'] = ResNet50_features['x'].astype(int).astype(str)
        ResNet50_features['y'] = ResNet50_features['y'].astype(int).astype(str)
        ResNet50_features['z'] = ResNet50_features['z'].astype(int).astype(str)
        ResNet50_features['TheIndex'] = ResNet50_features[['ID', 'x', 'y', 'z']].agg('_'.join, axis=1)
        dataList.append(ResNet50_features)


final_data = pd.concat(dataList)
final_data = final_data.iloc[:,3:]

Final = pd.merge(final_data,fully_annotated,  right_on='myIndex', left_on='TheIndex')
Final = Final.drop(['myIndex','TheIndex','ID_y','ID_x','Tile'],axis=1)

X=Final.drop(['Target'],axis = 1) # Features
Y=Final['Target']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
          max_depth=50, max_features='auto', max_leaf_nodes=None,
          min_impurity_decrease=0.0, min_impurity_split=None,
          min_samples_leaf=4, min_samples_split=5,
          min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,
          oob_score=False, random_state=100, verbose=0, warm_start=False)
RF.fit(X_train,y_train)


def plot_feature_importances(clf, X_train, y_train=None,
                             top_n=10, figsize=(8, 8), print_table=False, title="Feature Importances"):
    __name__ = "plot_feature_importances"

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from xgboost.core import XGBoostError
    from lightgbm.sklearn import LightGBMError

    try:
        if not hasattr(clf, 'feature_importances_'):
            clf.fit(X_train.values, y_train.values.ravel())

            if not hasattr(clf, 'feature_importances_'):
                raise AttributeError("{} does not have feature_importances_ attribute".
                                     format(clf.__class__.__name__))

    except (XGBoostError, LightGBMError, ValueError):
        clf.fit(X_train.values, y_train.values.ravel())

    feat_imp = pd.DataFrame({'importance': clf.feature_importances_})
    feat_imp['feature'] = X_train.columns
    feat_imp.sort_values(by='importance', ascending=False, inplace=True)
    feat_imp = feat_imp.iloc[:top_n]

    feat_imp.sort_values(by='importance', inplace=True)
    feat_imp = feat_imp.set_index('feature', drop=True)
    feat_imp.plot.barh(title=title, figsize=figsize)
    plt.xlabel('Feature Importance Score')
    plt.show()

    if print_table:
        from IPython.display import display
        print("Top {} features in descending order of importance".format(top_n))
        display(feat_imp.sort_values(by='importance', ascending=False))

    return feat_imp
plot_feature_importances(RF, X_train, y_train,
                             top_n=30, figsize=(6,6), print_table=False, title="Feature Importances")
feat_imp = pd.DataFrame({'importance':RF.feature_importances_})
feat_imp['feature'] = X_train.columns
feat_imp.sort_values(by='importance', ascending=False, inplace=True)
feat_imp = feat_imp.iloc[:100]
myindexes = feat_imp.feature.unique().tolist()


with open('Important_features.pkl', 'wb') as f:
    pickle.dump(myindexes, f)