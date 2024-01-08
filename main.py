import numpy as np
import pandas as pd


def load(filename):
    df = pd.read_csv(filename)
    df = df.drop(columns=["Demographic"])
    return df


def get_p_y(y_column, df):
    """
    Compute P(Y = 1)
    """

    # Initializes the counter for P(Y = 1)
    y1 = 0
    y1 = sum(df[y_column] == 1)

    # Finds the probability of P(Y = 1)
    return (y1 + 1) / (len(df) + 2)


def get_p_x_given_y(x_column, y_column, df):
    """
    Computes P(X = 1 | Y = 1) and P(X = 1 | Y = 0), where X is a single feature (column).
    x_column: name of the column containing the feature X.
    y_column: name of the class containing the class label.

    return: [P(X = 1 | Y = 1), P(X = 1 | Y = 0)]
    """

    # Initializes the two counters, which track the number of training samples where X_i is 1 given Y is 1 or 0
    x1_and_y0 = 0
    x1_and_y1 = 0

    # Counts the number of times x_i = 1 and y = 0
    x1_and_y0 = ((df[x_column] == 1) & (df[y_column] == 0)).sum()
    x1_and_y1 = ((df[x_column] == 1) & (df[y_column] == 1)).sum()

    # Counts the number of times y is 1
    y_is_1 = (df[y_column] == 1).sum()

    # Finds the probabilities for P(X = 1 | Y = 0) and P(X = 1 | Y = 1) using MAP with Laplace smoothing
    p_x1_given_y0 = (x1_and_y0 + 1) / ((len(df)-y_is_1) + 2)
    p_x1_given_y1 = (x1_and_y1 + 1) / (y_is_1 + 2)

    return [p_x1_given_y0, p_x1_given_y1]



def get_all_p_x_given_y(y_column, df):
    """
    Stores P(X_i = 1 | Y = y) in all_p_x_given_y[i][y] and returns all_p_x_given_y.
    """

    # We want to store P(X_i=1 | Y=y) in all_p_x_given_y[i][y]. Here, we are creating a new array
    # with df.shape[1] - 1 rows (that correspond to the number of X_is), and 2 columns (that
    # correspond to Y = 0 and Y = 1.
    all_p_x_given_y = np.zeros((df.shape[1] - 1, 2))

    # For each i in X_i, we loop through every row in df to see when X_i = 1 and Y = y, and add that to our counter
    for i in range(len(df.columns) - 1):
        all_p_x_given_y[i] = get_p_x_given_y(df.columns[i], y_column, df)

    return all_p_x_given_y


def get_prob_y_given_x(xs, y, all_p_x_given_y, p_y):
    """
    Computes the probability of a single row and y. In other words, we want to find P(Y | X) =
    P(Y) * P(X_1 | Y) * P(X_2 | Y) * ... * P(X_n | Y).
    """

    # Loops through the values of the Xs to find P(X_i = x_i | Y = y)
    prob_xs_given_y = 1
    for i in range(len(xs)):
        if xs.iloc[i] == 1:
            prob_xs_given_y *= all_p_x_given_y[i][y]
        elif xs.iloc[i] == 0:
            prob_xs_given_y *= (1 - all_p_x_given_y[i][y])

    return p_y * prob_xs_given_y


def compute_accuracy(all_p_x_given_y, p_y, df):
    # split the test set into X and y. The predictions should not be able to refer to the test y's.
    X_test = df.drop(columns="Label")
    y_test = df["Label"]

    num_correct = 0
    total = len(y_test)

    for i, xs in X_test.iterrows():
        p_y1_given_xs = get_prob_y_given_x(xs, 1, all_p_x_given_y, p_y)
        p_y0_given_xs = get_prob_y_given_x(xs, 0, all_p_x_given_y, 1-p_y)
        if p_y1_given_xs > p_y0_given_xs:
            if y_test[i] == 1:
                num_correct += 1
        else:
            if y_test[i] == 0:
                num_correct += 1


    accuracy = num_correct / total

    return accuracy


def main():
    # load the training set
    df_train = load("heart-train.csv")

    # compute model parameters (i.e. P(Y), P(X_i|Y))
    all_p_x_given_y = get_all_p_x_given_y("Label", df_train)
    p_y = get_p_y("Label", df_train)

    # load the test set
    df_test = load("heart-test.csv")

    print(f"Training accuracy: {compute_accuracy(all_p_x_given_y, p_y, df_train)}")
    print(f"Test accuracy: {compute_accuracy(all_p_x_given_y, p_y, df_test)}")

    """
    ratios = {}
    for i in range(len(df_train.columns)-1):
        p_y1_given_x1 = (all_p_x_given_y[i][1] * p_y) / ((all_p_x_given_y[i][1] * p_y) + (all_p_x_given_y[i][0] * (1-p_y)))
        p_y1_given_x0 = ((1-all_p_x_given_y[i][1]) * p_y) / (((1-all_p_x_given_y[i][1]) * p_y) + ((1-all_p_x_given_y[i][0]) * (1-p_y)))
        ratios[df_train.columns[i]] = p_y1_given_x1 / p_y1_given_x0

    sorted_items = sorted(ratios.items(), key=lambda item: item[1])
    print(sorted_items)
    """

if __name__ == "__main__":
    main()