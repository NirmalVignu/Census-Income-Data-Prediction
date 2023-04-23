## End to End ML Project on Census Income Data Prediction(Classification Probelm)

### created an environment
```
conda create -p venv python==3.8

conda activate venv/
```
### Install all the neccessary libraries
```
pip install -r requirements.txt

```

### Data Set Information:

Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

**Prediction task is to determine whether a person makes over 50K a year.**

### Attribute Information:

Listing of attributes:

\>50K, <=50K.

* age: continuous.
* workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
* fnlwgt: continuous.
* education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
* education-num: continuous.
* marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
* occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
* relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
* race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
* sex: Female, Male.
* capital-gain: continuous.
* capital-loss: continuous.
* hours-per-week: continuous.
* native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

### Model Report

```
Model Name: LogisticRegression
Accuracy:	0.8226984235310175
ROC AUC SCORE:	0.6925511177259119
Confusion Matrix:	[[10494 615] [ 1983 1561]]
Precision:	0.7173713235294118
Recall:	0.4404627539503386
F1 Score:	0.5458041958041958
```
##### Classification Report
![image](https://user-images.githubusercontent.com/70325804/233851754-66169317-e42f-45fe-828a-8d98f379dafe.png)


```
As our data set is imbalanced i.e,
 % of values are <=50K = 76.07182343065395
 % of values are >50K = 23.928176569346054

 tried giving class weight parameter and balancing using SMOTE both are not performing better than the normal approach so after careful observation from the different implementations logistic regression without balancing taken as best model and as it is imbalanced we have to look at the following statistics to determine model performance 
 ROC AUC SCORE, Confusion Matrix, Precision,Recall, F1 Score for which our model gives better results than the Base Line Score which i taken into COnsiderattion for evaluation and below is the comparision between the base line score and the acquired results
```
### Base Line Score 
```                    
Precision: 0.248
F1 score: 0.397
ROC AUC Score: 0.50
```
### Acquired Results
```
Precision:	0.7173713235294118
F1 Score:	0.5458041958041958
ROC AUC SCORE:	0.6925511177259119
```


![image](https://user-images.githubusercontent.com/70325804/233851691-3b95abd3-a67c-4bb6-9280-9410637eb347.png)

