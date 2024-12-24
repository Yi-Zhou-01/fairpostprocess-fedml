import numpy as np

def adult_iid(dataset, num_users):
    """
    Sample 
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def adult_noniid_new(dataset, num_users, partition_file):
    dict_clients = np.load(partition_file, allow_pickle=True)
    print("*********************")
    print("Using partitio file: ", partition_file.split("/")[-1])
    print(type(dict_clients))
    print(dict_clients.item().keys())
    return dict_clients.item()



def adult_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from Adult dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    # Datapoints in adult only train set: 80train: 35994 | 179*200=35800 178*200=35600 | 178/2=89 clients max
    # 70train: 31494
    # num_shards: must be even number
    num_shards, num_imgs = 178, 200
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.train_labels)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        # Random choose two shards for each client
        # print("idx_shard: ", len(idx_shard))
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        # Add the data within the selected shard to the corresponding client
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


