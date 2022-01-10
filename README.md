# Location_validation

Location validation is an API aims to predict the probability of occurrence of bird species in a given location in Switzerland.

## To launch:
```python api_RF.py```

## Requirements:

```pip install -r r.txt```

## endpoints:

There are three endpoints in this project:

#### Get species names:

```GET /names/```

This endpoint generates a list of the common names (in English) of all bird species in this study.

#### Get suggestion:

```GET /suggestion?lat={LATITUDE}&lon={LOGITUDE}/```

This endpoint aims at providing suggestions to users based on their location, such as possible species that can be observed within a 1km radius of the user’s location.

#### Predict probability of occurrence:

```POST /predict?lat={LATITUDE}&lon={LOGITUDE}&sp_name={SPECIES COMMON NAME}```

This endpoint aims at predicting the probability of occurrence of a species in a given location using the trained species distribution models
