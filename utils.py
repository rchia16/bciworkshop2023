import numpy as np

def make_labels(data, is_target:bool=False):
    data_len = data.shape[-1]
    if is_target:
        labels = np.ones((data_len, 1), dtype=bool)
    else:
        labels = np.zeros((data_len, 1), dtype=bool)
    return labels

def append_true_false_set(true_set, false_set):
    # assume shape is (n_trials, time_steps, n_channels)
    true_data, true_labels = true_set
    false_data, false_labels = false_set

    data_set = np.concatenate((true_data, false_data), axis=-1)
    label_set = np.concatenate((true_labels, false_labels), axis=0)
    return data_set, label_set

def shuffle_dset(data_set, label_set):
    N_d = data_set.shape[-1]
    N_l = len(label_set)
    print(data_set.shape)
    print(label_set.shape)
    assert N_d==N_l, "Incorrect labelling for data length: {data_set}"
    p = np.random.permutation(N_d)
    return data_set[...,p], label_set[p]

# Do not need to worry about this, just a function to extract the relevant data
# from the preprocessing files
def get_data_and_labels(selected_json, is_target:bool=False):
    keys = selected_json.keys()
    data_key = [k for k in keys if 'data' in k.lower()].pop()
    data = selected_json[data_key]
    data = np.array(data)
    labels = make_labels(data, is_target=is_target)

    return data, labels
