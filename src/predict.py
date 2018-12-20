from CNN import CNN
import sys


if __name__ == "__main__":
    model = CNN(saved_model_path="/users/btech/sanketyd/cs771/ML_Project/model/model_2018-11-07 10:08:39.123544.hd5")
    print(sys.argv[1])
    model.predict(sys.argv[1])
