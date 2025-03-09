def get_data_subset(n_samples, dataset, seed):
        if n_samples == -1:
            subset = dataset
        else:
            subset = dataset.shuffle(seed=seed)
            subset = subset.select(range(n_samples))

        return subset

