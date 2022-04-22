{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "64e5e0b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn import model_selection\n",
    "from sklearn import linear_model\n",
    "from sklearn import metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "4bfa1854",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('562_Dataset.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "366171d8",
   "metadata": {},
   "outputs": [],
   "source": [
    "county_name = list(set(df['county_name']))\n",
    "county_id = {}\n",
    "for i in range(len(county_name)):\n",
    "    county_id[county_name[i]] = i\n",
    "#county_id\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "7df54257",
   "metadata": {},
   "outputs": [],
   "source": [
    "state_name = list(set(df['state']))\n",
    "state_id = {}\n",
    "for i in range(len(state_name)):\n",
    "    state_id[state_name[i]] = i\n",
    "#state_id\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "9a19086c",
   "metadata": {},
   "outputs": [],
   "source": [
    "region_name = list(set(df['region']))\n",
    "region_id = {}\n",
    "for i in range(len(region_name)):\n",
    "    region_id[region_name[i]] = i\n",
    "#region_id\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "1667c2ef",
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"county_name\"].replace(county_id, inplace=True)\n",
    "df[\"state\"].replace(state_id, inplace=True)  \n",
    "df[\"region\"].replace(region_id, inplace=True) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "53cc5c46",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.dropna(axis=0,inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "b77f9115",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "caseid           False\n",
       "race             False\n",
       "vote2016         False\n",
       "faminc           False\n",
       "fips             False\n",
       "wait             False\n",
       "land_area        False\n",
       "county_name      False\n",
       "state            False\n",
       "county_pop       False\n",
       "income_county    False\n",
       "region           False\n",
       "dtype: bool"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().any()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "2e7535a6",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df[df.vote2016 != 5] #delete I don't recall candidate\n",
    "df = df[df.vote2016 != 4]  # delete i did not cast a vote, meaningless\n",
    "df = df[df.vote2016 != 3] #delete someone else - increases accuracy\n",
    "df = df[df.wait != 6]  #delete i don't remember wait time\n",
    "df = df.drop(['caseid', 'fips', 'region','county_name'], axis=1) \n",
    "#caseid is useless, fips is useless according to documentation, \n",
    "#region is redundent we have states, county_name is confusing -> different states could have same county name"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "da39fc6e",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.loc[:, df.columns != 'vote2016']\n",
    "y = df.loc[:,df.columns == 'vote2016']\n",
    "X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size= 0.25, random_state = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "4389e0d7",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/sklearn/utils/validation.py:993: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().\n",
      "  y = column_or_1d(y, warn=True)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:478: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:327: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:478: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:327: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:478: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:327: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:478: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:327: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:478: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:327: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:478: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:327: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:478: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:327: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:478: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:327: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:478: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:327: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:478: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:327: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:478: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:327: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:478: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:327: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:478: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/scipy/optimize/linesearch.py:327: LineSearchWarning: The line search algorithm did not converge\n",
      "  warn('The line search algorithm did not converge', LineSearchWarning)\n",
      "/Users/katemao/opt/anaconda3/lib/python3.8/site-packages/sklearn/utils/optimize.py:210: ConvergenceWarning: newton-cg failed to converge. Increase the number of iterations.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "LogisticRegression(multi_class='ovr', solver='newton-cg')"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lm_2 = linear_model.LogisticRegression(multi_class='ovr', solver='newton-cg')\n",
    "lm_2.fit(X_train, y_train)\n",
    "#.         accuracy.     f1\n",
    "#newton-cg: 0.61.       0.61\n",
    "#sag:        0.53.      0.11\n",
    "#liblinear:  0.59.      0.54\n",
    "#saga:      0.52.       0.05\n",
    "#default:   0.59.       0.55"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "41995f31",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.6055443704506737"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lm_2.score(X_test, y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "0fc1f16f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "         1.0       0.58      0.64      0.61      3107\n",
      "         2.0       0.63      0.57      0.60      3350\n",
      "\n",
      "    accuracy                           0.61      6457\n",
      "   macro avg       0.61      0.61      0.61      6457\n",
      "weighted avg       0.61      0.61      0.61      6457\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(metrics.classification_report(y_test, lm_2.predict(X_test)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "810bd6e4",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
