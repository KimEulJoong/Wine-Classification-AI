# Wine Classification with Random Forest

본 프로젝트는 와인 데이터셋을 활용하여 와인의 종류(`type`)를 분류하는 머신러닝 모델을 구축한 것입니다. `scikit-learn`의 `RandomForestClassifier`를 활용하여 모델을 학습하였으며, 다양한 데이터 전처리 및 시각화 기법을 통해 데이터에 대한 이해를 높이고, 성능을 개선하였습니다.

---

## Files

- `train.csv`, `test.csv` : 학습/테스트 데이터
- `2017015026_김을중.model` : 학습된 Random Forest 모델 (pickle로 저장됨)
- `wine_classification.ipynb` : 전체 분석 코드 (Jupyter Notebook)

---

## Dataset Description

- **Target variable**: `type` (와인의 종류)
- **Features**: 와인의 화학적 성분 등 수치형 데이터
- **Shape**: 
  - Train: `(N_train, 13)` — index와 quality 열은 제거
  - Test: `(N_test, 12)`

데이터는 다음과 같이 구성됩니다.

```text
- 수치형 변수: ['fixed acidity', 'volatile acidity', ..., 'alcohol']
- 범주형 변수: ['type']
```

---

## Exploratory Data Analysis (EDA)

- 범주형 변수 `type`에 대한 분포 시각화
- 수치형 변수의 boxplot을 통해 이상치 탐색
- `LabelEncoder`를 활용한 범주형 변수 전처리
- `Seaborn Heatmap`을 통한 변수 간 상관관계 분석
- `Barplot`을 통한 각 feature와 `type` 간의 관계 시각화

---

##  Preprocessing

- `LabelEncoder`: `type`을 수치형으로 인코딩
- `SimpleImputer`: 평균값으로 결측치 보간
- `train_test_split`: 훈련/검증 데이터 분할 (`test_size=0.2`)

---

## Modeling

- **모델**: `RandomForestClassifier` (sklearn)
- **성능 평가 지표**:
  - Accuracy
  - Recall (macro)
  - Precision (macro)
  - F1 Score (macro)

모델 훈련 및 예측 후 성능은 아래와 같습니다:

| Metric    | Score      |
|-----------|------------|
| Accuracy  | `0.XXX`    |
| Recall    | `0.XXX`    |
| Precision | `0.XXX`    |
| F1 Score  | `0.XXX`    |

> 정확한 수치는 실행 결과에 따라 다름

---

## Model Saving

학습된 모델은 `pickle`을 통해 `2017015026_김을중.model` 이름으로 저장되어 추후 배포나 평가에 재사용 가능합니다.

```python
with open('2017015026_김을중.model', 'wb') as f:
    pickle.dump(rf_classifier, f)
```

---

## Requirements

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## Author

김을중 (Euljoong Kim)  
2017015026  
AI 기반 와인 분류 모델 프로젝트

---

## License

MIT License (or specify as needed)
```

---
