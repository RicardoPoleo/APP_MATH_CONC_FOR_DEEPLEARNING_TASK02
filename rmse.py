import numpy as np

def our_rmse(predictions, targets):
    pred = np.array(predictions)
    tar = np.array(targets)
    pred_tar = (pred - tar)
    rmse = np.sqrt((pred_tar ** 2).mean())
    print(f"rmse={rmse}, for target {tar} and pred {pred}")
    return rmse