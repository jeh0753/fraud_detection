from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import Imputer
import cPickle as pickle


class CustomMixin(TransformerMixin):
    def get_params(self, **kwargs):
        return dict()

    def set_params(self, **kwargs):
        for key in self.get_params():
            setattr(self, key, kwargs[key])


class FilterColumns(CustomMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        self.keep_columns = ['user_age', 'delivery_method', 'body_length', 'num_payouts',\
         'gts', 'sale_duration', 'org_facebook', 'org_twitter']
        return X.ix[:, self.keep_columns]


class toNaN(CustomMixin):
    def fit(self, X, y):
        return self
    def transform(self, X):
        X.replace('', np.nan, inplace = True)
        X.replace('NaN', np.nan, inplace = True)
        X.replace(' ', np.nan, inplace = True)
        return np.array(X)


class Old_Age(CustomMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X['age_77_less'] = 0
        X['age_77_less'][X['user_age']<=77] = 1
        return X


class Delivery(CustomMixin):
    def fit(self, X, y):
        return self
    def transform(self, X):
        X['delivery_is_0'] = 0
        X['delivery_is_0'][X['delivery_method']==0.0] = 1
        return X


class Long_Body(CustomMixin):
    def fit(self, X, y):
        return self
    def transform(self, X):
        X['body_length_890_less'] = 0
        X['body_length_890_less'][X['body_length']<=890] = 1
        return X


class NumPayouts(CustomMixin):
    def fit(self, X, y):
        return self
    def transform(self, X):
        X['num_order_6_less'] = 0
        X['num_order_6_less'][X['num_payouts']<=6] = 1
        return X


class Gts(CustomMixin):
    def fit(self, X, y):
        return self
    def transform(self, X):
        X['gts_4_less'] = 0
        X['gts_4_less'][X['gts']<=6] = 1
        return X


class SaleDuration(CustomMixin):
    def fit(self, X, y):
        return self
    def transform(self, X):
        X['sale_duration_0'] = 0
        X['sale_duration_0'][X['sale_duration']<=6] = 1
        return X


class OrgFacebook(CustomMixin):
    def fit(self, X, y):
        return self
    def transform(self, X):
        X['org_facebook_0'] = 0
        X['org_facebook_0'][X['org_facebook']<=6] = 1
        return X


class OrgTwitter(CustomMixin):
    def fit(self, X, y):
        return self
    def transform(self, X):
        X['org_twitter_0'] = 0
        X['org_twitter_0'][X['org_twitter']<=6] = 1
        return X


class FilterColumns2(CustomMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.ix[:, ['age_77_less', 'delivery_is_0', 'body_length_890_less', \
        'num_order_6_less', 'gts_4_less', 'sale_duration_0', 'org_facebook_0', 'org_twitter_0']]



def make_dependent(y):
    return np.array((y=='fraudster') | (y =='fraudster_att') | (y=='fraudster_event')).astype(int)


if __name__ == '__main__':
    path = 'files/data.json'
    df = pd.read_json(path)
    y = make_dependent(df['acct_type'])
    p = Pipeline([
        ('filter', FilterColumns()),
        ('old', Old_Age()),
        ('delivery', Delivery()),
        ('body_len', Long_Body()),
        ('num_payout', NumPayouts()),
        ('gts', Gts()),
        ('sale_duratn', SaleDuration()),
        ('org_fb', OrgFacebook()),
        ('org_twt', OrgTwitter()),
        ('filter2', FilterColumns2()),
        ('nanify', toNaN()),
        ('impute', Imputer(missing_values = "NaN", strategy = "median")),
        ('resample', SMOTE()),
        ('rf',GradientBoostingClassifier(learning_rate=0.001))
    ])

    params = {'rf__n_estimators':[50,100,150]}
    cross_val = StratifiedKFold(5)
    gscv = GridSearchCV(p, params, scoring='f1', cv=cross_val,verbose=3)

    clf = gscv.fit(df, y)

    print 'Best parameters: %s' % clf.best_params_
    print 'Best F1 Score: %s' % clf.best_score_

    with open('model.pkl', 'w') as f:
        pickle.dump(clf, f)
