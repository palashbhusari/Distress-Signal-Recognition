import joblib

HandRaiseModel = joblib.load('ML/HandRaiseModels/model_random_forest.sav')

def infer(input): # input data
    input = [input] # 2d array
    return HandRaiseModel.predict(input)

# print(infer([[120,120]]))