import scipy.io


def LoadORL():
    orl_data = scipy.io.loadmat('orl_data.mat')
    orl_label = scipy.io.loadmat('orl_lbls.mat')
    from sklearn.cross_validation import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.33,
                                                        random_state=42)
    data = {'train': {'X': x_train,
                      'y': y_train},
            'test': {'X': x_test,
                     'y': y_test}}
    return data