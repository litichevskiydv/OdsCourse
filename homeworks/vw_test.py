import os
import numpy as np
from sklearn.metrics import accuracy_score


def main():
    predict_command = 'vw -t -i vw_model_passes_1_ngram_2.vw -d stackoverflow_test.vw -p vw_predict_passes_1_ngram_2_test1.csv --random_seed 17 --quiet'
    print(predict_command)
    os.system(predict_command)
	
    vw_pred = np.loadtxt('vw_predict_passes_1_ngram_2_test1.csv')
    test_labels = np.loadtxt('stackoverflow_test_labels.txt')
    score = accuracy_score(test_labels, vw_pred)
    print('Passes: 1, ngram: 2, score on test part: {0}'.format(score))


if __name__ == '__main__':
    main()
