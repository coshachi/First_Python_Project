"""
The approach is to create a 'classifier' - a program that takes a new example record and, based on previous examples,
determines which 'class' it belongs to. We begin with a training data set - examples with known solutions. The
classifier looks for patterns that indicate classification. These patterns can be applied against new data to predict
outcomes. If we already know the outcomes of the test data, we can test the reliability of our model.
If it proves reliable we could then use it to classify data with unknown outcomes.

    Marcin Koszalinski
    student ID: D20125156
    12 Dec 2020
"""

DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

# PERCENT is somewhat arbitrary. We can change this training/test data split to see if the outcome changes.
PERCENT = 75
CONDITION_1 = '<=50K'
CONDITION_2 = '>50K'

from read_from_file_and_net import get_file_from_net as read_url


def get_data(url):
    """
    This function reads data file directly from the web and splits it out into a list of tuples, one tuple per record.
    It also tests for value errors and silently drops any malformed rows.
    """
    bad_records = 0
    count = 0
    cleaned_dataset = []

    data = read_url(url)
    data = data.strip().split("\n")

    for record in data:
        count += 1
        try:
            record = record.strip().split(",")
            if not record:
                raise ValueError("Empty Record.")
            for i in range(1, len(record)):
                record[i] = record[i].strip()  # Deletes spaces for each element
            if '?' in record:  # Deletes incomplete records
                raise ValueError("Incomplete data.")
            else:
                cleaned_dataset.append(record)
            del record[2:4], record[-2]  # Deletes columns which are not needed as per instruction to the assignment.
        except ValueError as val_err:
            bad_records += 1
            print(f"Record #{count} rejected: {val_err}")
            continue

    print(f"\n{bad_records} out of {count} lines rejected.")
    print(f"\n{'-' * 125}")

    return tuple(cleaned_dataset)


def get_count_of_rows(data):
    """
    This function computes the total number of rows for each condition.
    """
    rows_condition_1, rows_condition_2 = 0, 0

    for record in data:
        if record[-1] == CONDITION_1:
            rows_condition_1 += 1
        elif record[-1] == CONDITION_2:
            rows_condition_2 += 1

    return rows_condition_1, rows_condition_2


def classifier_words(data):
    """
    This function creates a dictionary with unique words from the dataset with the condition assigned to them.
    Function firstly computes how many times each word appears in each condition. After that, it calculates a numeric
    weight by dividing count of word appearance by the count of all rows for each condition. Function later creates
    a dictionary with words grouped by the condition with higher fraction (numeric weight) which is a classifier for
    non-numeric columns.
    """
    count_1, count_2 = {}, {}
    factor_1, factor_2 = {}, {}
    words = {}
    rows_count_1, rows_count_2 = get_count_of_rows(data)

    # Computes the total count of words appearance in each condition
    for record in data:
        for i in range(0, len(record) - 1):
            try:
                int(record[i])
            except ValueError:
                if record[-1] == CONDITION_1:
                    if record[i] not in count_1:
                        count_1[record[i]] = 1
                    else:
                        count_1[record[i]] += 1
                elif record[-1] == CONDITION_2:
                    if record[i] not in count_2:
                        count_2[record[i]] = 1
                    else:
                        count_2[record[i]] += 1
                else:
                    continue

    # Divides each element's count_1 by the total number of rows with condition 1
    for key, value in count_1.items():
        factor_1[key] = count_1[key] / rows_count_1
    # Divides each element's count_2 by the number of rows with condition 2
    for key, value in count_2.items():
        factor_2[key] = count_2[key] / rows_count_2

    # Merges factor_1 and factor_2 to get a full list of words as a dictionary, used in 'for' loop
    factor = {**factor_1, **factor_2}

    # Groups words to the condition with higher fraction (numeric weight)
    for word, value in factor.items():
        try:
            if factor_1[word] >= factor_2[word]:
                words[word] = CONDITION_1
            else:
                words[word] = CONDITION_2
        except KeyError:  # Considers words which are assigned to only one of the conditions.
            if word in factor_1:
                words[word] = CONDITION_1
            else:
                words[word] = CONDITION_2

    return words


def classifier_numeric(data):
    """
    For each numeric element from the records, average the values for each attribute in a list of known results for
    condition #1 and, separately, a list of known results for condition #2. The condition #1 and condition #2 averages
    are then averaged against each other to compute midpoint values. That is the classifier for numeric columns which
    is returned by this function.
    """
    avg_1, avg_2 = [], []
    avg = []
    rows_count_1, rows_count_2 = get_count_of_rows(data)

    numeric_sum_1 = [0] * (len(data[0]) - 1)
    numeric_sum_2 = [0] * (len(data[0]) - 1)

    # Computes the totals for each numeric column
    for record in data:
        for i in range(0, len(record) - 1):
            try:
                int(record[i])
            except ValueError:
                numeric_sum_1[i] = '-'
                numeric_sum_2[i] = '-'
            else:
                record[i] = int(record[i])  # Changes strings to integers
                if record[-1] == CONDITION_1:
                    numeric_sum_1[i] += record[i]
                elif record[-1] == CONDITION_2:
                    numeric_sum_2[i] += record[i]

    # Computes the average for each numeric column
    for i in numeric_sum_1:
        try:
            i = i / rows_count_1
        except TypeError:
            pass
        avg_1.append(i)
    for i in numeric_sum_2:
        try:
            i /= rows_count_2
        except TypeError:
            pass
        avg_2.append(i)

    # Computes the midpoints - the average of averages for condition 1 and 2
    for i in range(0, len(record) - 1):
        try:
            avg.append((avg_1[i] + avg_2[i]) / 2)
        except TypeError:
            avg.append('-')

    return tuple(avg)


def create_classifier(data):
    """
    This function uses functions classifier_numeric() and classifier_words() to return respectively a tuple with
    average numbers and a dictionary with words assigned to the condition. Function prints results for classifier with
    words and classifier with numbers.
    """
    words = classifier_words(data)
    numbers = classifier_numeric(data)

    print(f"Words grouped by condition:")
    for word, value in words.items():
        print(f"{word}: {value}")
    print(f"\n{'-' * 125}")
    print(f"Mid points for numeric columns:")
    print(numbers)
    print(f"\n{'-' * 125}")

    return words, numbers


def test_classifier(data, words, numbers):
    """
    This function tests the classifier.
    For numeric elements: if value is lower than the mid point from the classifier, insert condition#1 in the adequate
    index in the list with results, if value is higher, insert condition#2.
    For words: if word is assigned to condition#1 in the words classifier, insert condition#1 in the adequate index in
    the list with results, if word is assigned to condition#2, insert condition#2.
    As the last index in the list with results, insert condition which appears more times in this list. After that,
    check if condition in the last index matches the condition from the original record.
    Reset the last index in the list with results before checking a next record.
    If the tested word did not exist in the training dataset, exclude the record from testing and print the details.
    """
    count_total = 0
    true_count = 0
    false_count = 0
    bad_records = 0
    temp_result_list = [''] * len(data[0])

    for record in data:
        temp_result_list[-1] = ''
        try:
            count_total += 1
            for i in range(0, len(record) - 1):
                try:
                    int(record[i])
                    if int(record[i]) < numbers[i]:
                        temp_result_list[i] = CONDITION_1
                    else:
                        temp_result_list[i] = CONDITION_2
                except ValueError:
                    if words[record[i]] == CONDITION_1:
                        temp_result_list[i] = CONDITION_1
                    else:
                        temp_result_list[i] = CONDITION_2
            if temp_result_list.count(CONDITION_1) >= len(record) / 2:
                temp_result_list[-1] = CONDITION_1
            else:
                temp_result_list[-1] = CONDITION_2

            print(f"Line {count_total}: {temp_result_list} ", end='')
            if record[-1] == temp_result_list[-1]:
                true_count += 1
                print("CORRECT")
            else:
                false_count += 1
                print("INCORRECT")
        # Excludes records with words which were not checked in training_dataset
        except KeyError as key_err:
            print(f"Line {count_total}: REJECTED. Value {key_err} was not included in the 'Training Dataset'.")
            bad_records += 1
            continue

    print(f"\n{'-' * 125}")
    print(f"{100 - PERCENT}% of all data was considered in the 'Test Dataset'.")
    print(f"All records tested: {count_total}.")
    print(f"Rejected records: {bad_records}")
    print(f"\nCorrectly classified records: {true_count}.")
    print(f"Incorrectly classified records: {false_count}.")
    print(f"{(true_count / count_total) * 100:.2f}% of records correctly classified.")
    print(f"{(false_count / count_total) * 100:.2f}% of records incorrectly classified.")


def main():
    # Make a tuple of tuples from the raw data (a spreadsheet-like 2D array)
    cleaned_dataset = get_data(DATA_URL)

    # Break out our dataset into a training and test sets where the training set has a number of records determined
    # by the PERCENT value. The test set has the remaining records.
    training_dataset = cleaned_dataset[:int(len(cleaned_dataset) * PERCENT / 100)]
    test_dataset = cleaned_dataset[int(len(cleaned_dataset) * PERCENT / 100):]

    # Create the classifier values
    words, numbers = create_classifier(training_dataset)

    # Test the records
    test_classifier(test_dataset, words, numbers)


if __name__ == "__main__":
    main()
