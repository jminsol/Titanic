
from entity import Entity
import pandas as pd
import numpy as np


from sklearn.tree import DecisionTreeClassifier #dtree
from sklearn.ensemble import RandomForestClassifier #rforest
from sklearn.naive_bayes import GaussianNB #nb
from sklearn.neighbors import KNeighborsClassifier #knn
from sklearn.svm import SVC #svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold   # k =count
from sklearn.model_selection import cross_val_score

"""
#### PassengerId  고객ID,
#### Survived 생존여부,  --> Answer!!
Pclass: 1 = First class, 2 = Second class, 3 = Third class,
Name,
Sex,
Age,
SibSp Siblings, sisters, partners,
Parch Parents, Children,
#### Ticket,
Fare,
#### Cabin,
Embarked ports C = 쉐브루, Q = 퀸즈타운, S = 사우스햄튼
"""

class Service:
    def __init__(self):
        self.entity = Entity()
        pass

    
    def new_model(self, payload) -> object:
        this = self.entity
        this.fname = payload
        return pd.read_csv(this.context + this.fname) # p.139  df = tensor

    @staticmethod
    def create_train(this) -> object:
        return this.train.drop('Survived', axis=1) # train 은 답이 제거된 데이터셋이다.

    @staticmethod
    def create_label(this) -> object:
        return this.train['Survived'] # label 은 곧 답이 된다.

    @staticmethod
    def drop_feature(this, feature) -> object:
        # axis =0 for all ROWs in each COLUMN
        # axis =1 for all COLUMNS in each ROW
        this.train = this.train.drop([feature], axis = 1)
        this.test = this.test.drop([feature], axis = 1) # p.149 에 보면 훈련, 테스트 세트로 나눈다
        return this


    @staticmethod
    def pclass_ordinal(this) -> object:
        return this

    @staticmethod
    def sex_norminal(this) -> object:
        # male = 0, female = 1
        combine = [this.train, this.test] # train과 test 가 묶입니다.
        sex_mapping = {'male':0, 'female':1}
        for dataset in combine:
            dataset['Sex'] = dataset['Sex'].map(sex_mapping)
        # this.train = this.train # overriding
        # this.test = this.test //실험
        return this

    @staticmethod
    def age_ordinal(this) -> object:
        train = this.train
        test = this.test
        train['Age'] = train['Age'].fillna(-0.5)
        test['Age'] = test['Age'].fillna(-0.5)
    
        '''
        reason for setting -.5 : Since the weight of Age for survival is high, we need more precise, detailed approash.
        Thus, treat passengers whose ages are unknown as unknown values in order to decrease errors of actual values, process their age to -.5 as a boundary value.

        '''
        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf] # 이 파트는 범위
         # -1 이상 0 미만....60이상 기타 ...
         # [] 에 있으니 이것은 변수명
        labels = ['Unknown', 'Baby', 'Child', 'Teenager','Student','Young Adult', 'Adult', 'Senior']
        # [] 은 변수명으로 선언
        train['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
        test['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
        
        age_mapping = {
            'Unknown': 0,
            'Baby': 1,
            'Child': 2,
            'Teenager': 3,
            'Student': 4,
            'Young Adult': 5,
            'Adult': 6,
            'Senior': 7
        }
        train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
        test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
        this.train = train
        this.test = test
        return this

    @staticmethod
    def sibsp_numeric(this) -> object:
        return this

    @staticmethod
    def parch_numeric(this) -> object:
        return this

    @staticmethod
    def fare_ordinal(this) -> object:
        this.train['FareBand'] = pd.qcut(this['Fare'], 4, labels={1,2,3,4})
        this.test['FareBand'] = pd.qcut(this['Fare'], 4, labels={1,2,3,4})
        return this


    @staticmethod
    def fareBand_nominal(this) -> object:  # 요금이 다양하니 클러스터링을 하기위한 준비
        this.train = this.train.fillna({'FareBand' : 1})  # FareBand 는 없는 변수인데 추가함
        this.test = this.test.fillna({'FareBand' : 1})
        return this

    @staticmethod
    def embarked_norminal(this) -> object:
        this.train = this.train.fillna({'Embarked': 'S'}) # S가 가장 많아서 빈곳에 채움
        this.test = this.test.fillna({'Embarked': 'S'}) # 교과서 144
        # Encoding class labels in *integers* 
        # 교과서 146 문자 blue = 0, green = 1, red = 2 로 치환 -> mapping 합니다.
        this.train['Embarked'] = this.train['Embarked'].map({'S': 1, 'C' : 2, 'Q' : 3}) # This is not 'ordinal' 
        this.test['Embarked'] = this.test['Embarked'].map({'S': 1, 'C' : 2, 'Q' : 3})
        return this

    @staticmethod
    def title_norminal(this) -> object:
        combine = [this.train, this.test]
        for dataset in combine:
            dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
            # expand = true: print in coulums; expand = false: print in labels
            # Created a new column 'Title' and saved data Starting with capital and ends with lower_case stop at '.' 
        for dataset in combine:
            dataset['Title'] = dataset['Title'].replace(['Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona', 'Mme'], 'Rare')
            dataset['Title'] = dataset['Title'].replace(['Countess','Lady','Sir'], 'Royal')
            dataset['Title'] = dataset['Title'].replace('Ms','Miss')
            dataset['Title'] = dataset['Title'].replace('Mlle','Mr')
        title_mapping = {'Mr':1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Royal': 5, 'Rare': 6}
        for dataset in combine:
            dataset['Title'] = dataset['Title'].map(title_mapping)
            dataset['Title'] = dataset['Title'].fillna(0) # Unknown
        this.train = this.train
        this.test = this.test
        return this
        
    @staticmethod
    def create_k_fold():
        return KFold(n_splits=10, shuffle = True, random_state=0)
    
    
    def accuracy_by_dtree(self,this):
        dtree = DecisionTreeClassifier()
        score = cross_val_score(dtree, this.train, this.label, cv=Service.create_k_fold(), n_jobs=1, scoring='accuracy')
        return round(np.mean(score)*100,2)
        
    def accuracy_by_rforest(self,this):
        rforest = RandomForestClassifier()
        score = cross_val_score(rforest, this.train, this.label, cv=Service.create_k_fold(), n_jobs=1, scoring='accuracy')
        return round(np.mean(score)*100,2)
        
    def accuracy_by_nb(self,this):
        nb = GaussianNB()
        score = cross_val_score(nb, this.train, this.label, cv=Service.create_k_fold(), n_jobs=1, scoring='accuracy')
        return round(np.mean(score)*100,2)
        
    def accuracy_by_knn(self, this):
        knn = KNeighborsClassifier()
        score = cross_val_score(knn, this.train, this.label, cv=Service.create_k_fold(), n_jobs=1, scoring='accuracy')
        return round(np.mean(score)*100,2)
        
    def accuracy_by_svm(self, this):
        svm = SVC()
        score = cross_val_score(svm, this.train, this.label, cv=Service.create_k_fold(), n_jobs=1, scoring='accuracy')
        return round(np.mean(score)*100,2)
        
