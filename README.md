## 개인 프로젝트
통합공공임대주택 정책 대상 가구(청년, 신혼, 중장년, 고령 가구)별로 선택하여 공공임대주택 입주의향을 예측하는 데 큰 영향을 미치는 입주의향 결정 요인을 찾고자 하였습니다.

shap value를 cross validation하여 shap importance를 구한 뒤, shap importance를 기준으로 feature selection을 진행했습니다.

## Stack
```
Python
SHAP(SHapley Additive exPlanations)
```

# 코드 실행
## 1. Data preprocessing
아래 Jupyter Notebook으로 기술되어 있음. 
https://github.com/HyunJae0/SHAP-CV-Feature-Selection/blob/main/preprocessing.ipynb

사용한 데이터는 MDIS에서 제공한 2018, 2019, 2020, 2021년 주거실태조사 결과입니다. 

예측 대상이 '통합공공임대주택 정책 대상자'이기 때문에 서민층 및 중산층 무주택 가구를 기준으로, 각 대상 유형(청년, 신혼, 중장년, 고령)의 기준에 맞는 데이터만 추출했습니다. 이때, 이미 공공임대주택에 거주하고 있는 가구는 제외하였습니다. 단 한 번도 공공임대주택에 거주하지 않은 가구만 예측하기 위함입니다.

그리고 몇 가지 파생 변수를 만들었습니다. 예를 들어, 소득 대비 주택 임대로 비율은 정책 대상자들은 모두 무주택 가구이기 때문에, 무주택 가구의 소득 대비 주택 임대료에 대한 부담 정도에 따라 입주의향 여부가 나뉘는지 확인하기 위함입니다. 

이외에도 가구의 재무 건전성을 확인하기 위해 총 소득 대비 생활비의 비중, 총 소득 대비 월평균 주거관리비의 비중, 중/장기부채부담지표, 총 자산 대비 금융/기타 자산의 비중 등 다양한 파생 변수를 추가하였습니다.

## 2. SHAP
SHAP은 머신러닝 모델의 예측 결과에 대한 특성 중요도(feature importance)를 해석하고 시각화하기 위한 것으로, Shapley value를 기반으로 각 특성이 모델 예측에 기여하는 영향을 추정합니다.

### 2.2 combine one-hot
범주형 변수를 원-핫 인코딩하여 사용할 경우, 개별 카테고리에 대한 shap value가 계산됩니다. 범주형 변수 자체가 모델에 미치는 영향을 확인하기 위해 원-핫 인코딩된 개별 카테고리의 shap value를 다시 하나의 범주형 변수로 결합합니다. 

이를 위한 코드는 https://github.com/shap/shap/issues/1654에 게시된 코드를 사용하였습니다.
이 코드는 여러 개의 원-핫 인코딩된 카테고리들의 SHAP값을 통합해서 하나의 SHAP Explanation으로 반환하는 함수입니다. mask = True인 카테고리 컬럼들의 SHAP 정보를 모아 sv_name이라는 객체를 만듭니다. 그리고 sv_name.values.sum(axis=1)로 원-핫 인코딩된 개별 카테고리 컬럼에 분산되어 있던 shap value를 하나의 값으로 합칩니다. 예를 들어 '가구주 성별'이라는 두 개의 카테고리(남성, 여성)를 가지는 범주형 변수가 원-핫 인코딩되어 가구주 성별_남성, 가구주 성별_여성으로 나뉘어 각각 shap value를 가졌었다면, combine_one_hot 함수를 통해 두 개의 shap value가 합쳐진 값이 '가구주 성별'의 shap value가 됩니다.

### 2.3 Nested Stratified K-Fold Cross-Validation
데이터를 4개의 그룹(청년, 신혼, 중장년, 고령)으로 나눴을 때, 신혼 그룹의 데이터 개수는 6,119개 였습니다. 

적은 데이터로 모델 성능의 일반화를 확인하기 위해 cross validation을 사용했습니다. 이때 일반적인 cross validation이 아닌 nestesd cross validation을 사용했습니다.

예를 들어, 일반적인 k-fold cross validation은 k개의 fold를 만들면, 하나의 fold를 test set으로 지정하고 남은 k-1개의 fold를 train set으로 지정해 학습을 진행합니다. 만약, k = 5라면 이 과정을 5번 반복합니다.
그리고 일반적으로 train set과 valid set에 대한 교차 검증은 unseen data에 모델을 적용하기 전, 성능 검증이나 하이퍼파라미터를 탐색하기 위해서, train set과 test set에 대한 교차 검증은 다양한 test data에 대해 model의 일반화 성능을 평가하기 위해서입니다.
nested cross validation은 이 두 종류의 교차 검증을 수행하여, 두 효과를 모두 이용하는 방법입니다. 

이 방법은 train set과 test set을 k개의 fold로 나눈 뒤, 각 fold의 train set을 다시 n개의 train-valid fold로 분할합니다. 즉, 총 k*n개의 모델이 생성되므로 모델 튜닝과 동시에 일반화 성능까지 확인할 수 있습니다. 사용하는 방법은 다음 코드와 같이 outer fold와 inner fold를 설정합니다. outer fold는 최종 모델의 일반화 성능을 추정, inner fold는 하이퍼파라미터를 찾아 모델을 튜닝하기 위해 사용됩니다. 다음과 같이 outer = 10, inner = 3으로 한다면, 총 30개의 모델이 생성됩니다.

이때, 데이터 셋의 클래스 분포가 일정하지 않아서 stratified k-fold를 사용하였으며, 셔플을 통해 각 폴드가 전체 데이터 분포를 더 잘 대표하도록 하였습니다. 

그리고 하이퍼파라미터를 찾기 위해 하이퍼파라미터 최적화 프레임워크인 Optuna를 사용했습니다. 탐색할 하이퍼파라미터의 space를 정의하면, 그 안에서 샘플링하여 하이퍼파라미터 최적화를 진행합니다. 다양한 하이퍼파라미터를 튜닝해서 사용할 것이기 때문에 모든 하이퍼파라미터의 space를 확인하는 그리드 서치를 이용하는 방법은 사용하지 않았습니다.(예. 랜덤 서치 후, 그리디 서치)

```
skf_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
skf_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

shap_values = np.zeros_like(X_42)
shap_data = np.zeros_like(X_42)

skf_scores1 = np.zeros(10)
skf_scores2 = np.zeros(10)
base_score = np.zeros(10)

for index, (train_index, test_index) in enumerate(skf_outer.split(X_42, y_42)):    
    direction = "maximize"
    early_stopping = EarlyStoppingCallback(10, direction=direction)
    sampler = optuna.samplers.TPESampler(seed=10)
    study =  optuna.create_study(direction=direction, sampler=sampler, 
                             pruner=optuna.pruners.HyperbandPruner())
    
    print('\n------ Fold Number:',index)
    X_train_42, X_test_42 = X_42.iloc[train_index], X_42.iloc[test_index]
    y_train_42, y_test_42 = y_42.iloc[train_index], y_42.iloc[test_index]
    
    base_42 = XGBClassifier(random_state = 0,booster = 'gbtree',objective = 'binary:logistic')
    base_42.fit(X_train_42, y_train_42)
    base_proba_42 = base_42.predict_proba(X_test_42)[:, 1]
    base_score[index] = roc_auc_score(y_test_42, base_proba_42)
    print('TEST ROC_AUC (Base Model):',base_score[index])

    clf = XGBClassifier(random_state = 0,booster = 'gbtree',objective = 'binary:logistic')
    param_distributions = {
        'n_estimators' :optuna.distributions.IntDistribution(50, 200),
        'learning_rate' :optuna.distributions.FloatDistribution(0.01, 0.1,step=0.01),
        'max_depth' :optuna.distributions.IntDistribution(1, 10),
        'max_leaves': optuna.distributions.IntDistribution(2, 1024, step=2),
        'subsample':optuna.distributions.FloatDistribution(0.1, 1.0,step=0.1),
        'colsample_bytree' :optuna.distributions.FloatDistribution(0.1, 1.0,step=0.1),
        'gamma' :optuna.distributions.IntDistribution(1, 10),
        'reg_alpha' :optuna.distributions.IntDistribution(1, 10),
        'reg_lambda' :optuna.distributions.IntDistribution(1, 10)
        }
    
    optuna_search_42 = optuna.integration.OptunaSearchCV(estimator = clf, param_distributions=param_distributions, 
                                                      n_trials=200, study = study, cv=skf_inner, scoring = 'roc_auc', refit = True,
                                                      random_state=0,callbacks=[early_stopping])
    result_42 = optuna_search_42.fit(X_train_42, y_train_42) 
    result_42.best_estimator_.fit(X_train_42, y_train_42)      

    explainer_42 = shap.TreeExplainer(result_42.best_estimator_)
    shap_values_te = explainer_42(X_test_42)
    shap_values[test_index,:] = shap_values_te.values
    shap_data[test_index,:] = shap_values_te.data
    
    y_train_pred_proba_42 = result_42.best_estimator_.predict_proba(X_train_42)[:, 1]
    skf_scores1[index] = roc_auc_score(y_train_42, y_train_pred_proba_42)
    print('Train ROC_AUC:',skf_scores1[index])
    
    y_test_pred_proba_42 = result_42.best_estimator_.predict_proba(X_test_42)[:, 1]
    skf_scores2[index] = roc_auc_score(y_test_42, y_test_pred_proba_42)
    print('TEST ROC_AUC:',skf_scores2[index])
    
print('')
print('train set mean roc_auc',round(np.mean(skf_scores1),3))
print('test set mean roc_auc(Base)',round(np.mean(base_score),3))
print('test set mean roc_auc',round(np.mean(skf_scores2),3))
```

이 과정 속에서 계산되는 shap value로 shap importance를 확인해서 가장 shap importance가 낮은 변수를 하나씩 제거했습니다. 변수가 하나 남을 때까지 진행했습니다.
```
auc_bootstrap = []
def bootstrap_auc(clf, X_train, y_train, X_test, y_test, nsamples=2000):
    for b in range(nsamples):
        idx = rs.randint(X_train.shape[0], size=X_train.shape[0])
        clf.fit(X_train[idx], y_train[idx])
        pred = clf.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test.ravel(), pred.ravel())
        auc_bootstrap.append(roc_auc)
    return np.percentile(auc_bootstrap, (2.5, 97.5))
```
그리고 변수 개수만큼 만들어진 모델들에 대해 신뢰 구간 95%에서 2,000번 부트스트랩한 roc_auc 값을 기준으로 신뢰 구간 평균 차이를 통해 가장 성능이 좋은 모델을 선정했습니다. 이 내용은 아래 Jupyter Notebook에 기술되어 있습니다.
부트스트랩: https://github.com/HyunJae0/SHAP-CV-Feature-Selection/blob/main/%EC%8B%A0%ED%98%BC%EA%B0%80%EA%B5%AC%20xgboost%20AUC%20%EB%B6%80%ED%8A%B8%EC%8A%A4%ED%8A%B8%EB%9E%A9%2095%25%20%EC%8B%A0%EB%A2%B0%20%EA%B5%AC%EA%B0%84-Copy1.ipynb

신뢰 구간 평균 차이 https://github.com/HyunJae0/SHAP-CV-Feature-Selection/blob/main/%EC%84%B1%EB%8A%A5%20%EC%A0%95%EB%A6%AC_%EC%8B%A0%ED%98%BC%EA%B0%80%EA%B5%AC-Copy1.ipynb


# 결과
예를 들어 신혼 가구의 공공임대주택 입주의향 모델 중 XGBoost 모델은 변수 개수별 모델별로 부트스트랩을 2000번 수행해서 비교한 결과, 17개의 변수를 사용했을 때 가장 성능이 높았습니다.

![image](https://github.com/user-attachments/assets/badbcfb5-f4da-4244-bb7e-d07d96d65966)

이 17개의 변수들이 신혼 가구의 공공임대주택 입주의향 모델에 어떠한 영향을 미쳤는지 shap value를 plot해서 확인할 수 있습니다. 단, 이 결과는 모델이 왜 양성과 음성을 예측했는지에 대한 행동을 설명하는 것이지 인과관계를 설명하는 것이 아닙니다.

![image](https://github.com/user-attachments/assets/8eed24a6-d20b-44f0-9a34-b961caeaa386)

위의 결과는 shap importance 순으로 나타낸 변수들입니다. 신혼 가구 공공임대주택 입주의향 모델에 가장 큰 영향력을 미친 변수는 '현재 가장 필요한 주거지원 1순위'. '현재 주택의 면적', '현재 주택의 점유형태', '주택 마련 예상 소요연수' .... 순입니다.

앞에 Cat_이 붙은 것은 원핫 인코딩된 범주형 변수들의 shap value를 하나로 합친 것을 의미합니다. (즉, 범주형 변수임을 의미합니다.)

Cat_이 붙지 않은 수치형 변수들에 대해 먼저 해석하자면, 신혼 가구에게는 '현재 주택의 면적'이 작을수록 '입주의향 있음(1)'에 좋은 영향을 미쳤습니다. 그리고 '장기부채부담지표'와 '중기부채부담지표'가 높을수록 '입주의향 있음(1)'에 영향을 미쳤음을 확인할 수 있고, '소득 대비 주택 임대료'의 비율은 낮을수록 '입주의향 있음(1)'에 영향을 미쳤습니다. '소득 대비 주거관리비의 비율'이 높을수록 '입주의향 있음(1)'에 영향을 미쳤습니다.

즉, 신혼 가구 데이터 중에서 현재 주택 면적이 좁고, 장기부채부담과 중기부채부담이 높으며, 현재 거주 중인 주택 임대료의 비율이 소득에서 차지하는 비율이 낮을수록, 소득 대비 주거관리비의 비율이 높을수록 '입주의향 없음(0)'과 '입주의향 있음(1)' 중 '입주의향 있음(1)'의 예측을 증가시키는 영향을 미친 것입니다.

다음 그림은 이 17개 변수에 대한 shap feature importance 입니다.

![image](https://github.com/user-attachments/assets/919dd092-b4d9-4695-b903-c83aa9f9c05f)

범주형 변수에 대한 영향 확인은 다음과 같습니다.

원-핫 인코딩된 범주형 변수를 다시 합친 이유는 해당 범주형 질문이 가지는 전체 영향력을 파악하기 위함입니다. 이 합친 범주형 변수를 다시 세분화해서 확인하려면,

예를 들어 '현재 가장 필요한 주거지원 1순위'라는 범주형 변수의 고유 카테고리 개수를 확인한 다음, '주거지원 1순위'에 해당하는 여러 원-핫 컬럼들의 shap값을 카테고리 개수 기준으로 나눕니다. 그리고 shap 값에 대응되는 범주형 데이터를 매핑 및 그룹화하면 됩니다.

![image](https://github.com/user-attachments/assets/d80c2f4c-1ed2-4338-aaf5-5ad929882208)

다음 그림은 '현재 가장 필요한 주거지원 1순위'라는 범주형 변수의 카테고리별 boxplot을 나타낸 것입니다.

![image](https://github.com/user-attachments/assets/fcc45350-accf-44ee-9910-ed23ba1001d3)

x축은 카테고리이고, y축 값이 shap value입니다. 신혼 가구의 경우, 현재 가장 필요한 주거지원 1순위가 '장기공공임대주택 공급'인 경우 '입주의향 있음(1)'에 좋은 영향을 미쳤고, 현재 가장 필요한 주거지원 1순위가 '없음'인 경우 shap value가 음수가 됩니다. 이는 '현재 가장 필요한 주거지원 1순위 - 없음'이 예측을 감소시키는 영향을 미친 것입니다.

다음은 '주택 마련 예상 소요연수'에 대한 결과입니다.

![image](https://github.com/user-attachments/assets/5af4866c-7ea6-4c26-90d8-9c9aa3c1f137)

신혼 가구의 주택 마련 예상 소요연수가 '1년 미만' 또는 '1~3년'으로 비교적 짧을수록 '입주의향 있음(1)'에 부정적인 영향을. 반면, 주택 마련 계획이 아예 없는 경우 '입주의향 있음(1)'에 긍정적인 영향을 준 것을 확인할 수 있습니다

다음은 '현재 주택의 점유형태'에 대한 결과입니다.

![image](https://github.com/user-attachments/assets/e6580fa9-97d4-497d-bf94-eb050b27d88c)

현재 거주 중인 주택의 점유 형태가 '무상'인 경우만 '입주의향 있음(1)'에 부정적인 영향을 미쳤고, 월세와 전세인 경우 모두 긍정적인 영향을 미쳤으며, 특히 보증금이 있는 월세인 경우 다른 카테고리에 비해 '입주의향 있음(1)'에 가장 높은 영향력을 미친 것을 확인할 수 있습니다.

