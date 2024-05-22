import rmse
def test_rmse():
    predictions = [1, 2, 3, 4, 5]
    targets = [1, 2, 3, 4, 5]

    assert rmse.our_rmse(predictions, targets) == 0

def test_rmse_2():
    predictions = [2, 3, 4, 5, 6]
    targets = [1, 2, 3, 4, 5]

    assert rmse.our_rmse(predictions, targets) == 1

if __name__ == "__main__":
    test_rmse()
    test_rmse_2()