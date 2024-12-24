# A Post-Processing-Based Fair Federated Learning Framework

<!--- README Template from: https://github.com/paperswithcode/releasing-research-code/blob/master/templates/README.md --->

This repository is the official implementation of [A Post-Processing-Based Fair Federated Learning Framework](https://arxiv.org/). 

<!-- >📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials -->

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

<!-- >📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc... -->

---
## Data Pre-processing

Please save the pre-processed dataset under `data/<dataset_name>`, e.g. `data/adult/adult.csv`

Please change the corresponding dataset path and definition in `dataset.py`.

We follow the following data pre-processing precedure for each data used in the experiments:


* ##### [ Adult dataset](https://archive.ics.uci.edu/dataset/2)
    * “income” is used as target label with two classes “<=50K” and “>50K”.
    * “sex” is used as the sensitive attribute with “male”==1 and “female”==0.
    * One-hot encoding for all categorical features
    * Scikit-learn StandardScaler to standardise continuous features.



* ##### [Compas](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm)
    we followed the pre-processing procedures used in [How We Analyzed the COMPAS Recidivism Algorithm](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm) and [Fairness Beyond Disparate Treatment & Disparate Impact: Learning Classification without Disparate Mistreatment](http://arxiv.org/abs/1610.08452).
    * "race" is used as the sensitive attribute. Samples with the race “Caucasian” are categorised as the privileged group and samples with the race “African-American” as the unprivileged group. Samples within other race groups are dropped for simplicity. 
    * We only use a subset of the data features including “age_cat”, “sex”, “priors_count”, “c_charge_degree”, as well as sensitive attribute “race” and target label “two_year_recid”.
    * One-hot encoding for all categorical features
    * Scikit-learn StandardScaler to standardise continuous features.


* ##### [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)
    * All 12 leads are used with a sampling frequency of 100Hz.
    * Only samples with 100% confidence diagnosis are included.
    * Target label is as “normal” (ECG), and ECGs labelled with different heart diseases are considered as “abnormal” (i.e. normal==0).
    * We use patient age as the sensitive attribute with "age>60"==0 or "age>60"==1.


* ##### [NIH Chest X-Ray](http://arxiv.org/abs/1705.02315)
    * Only samples with label “No Findings” or “Effusion” are included, and “Effusion” is used as target label.
    * “Patient gender” is used as the sensitive attribute
    * Resize each image into size (256 * 256 * 3) with 3 channels.
    * Images are also normalised using the required mean and standard deviation based on the pre-trained model used.


## Data Partition

Please save partition file under path `data/<dataset_name>/partition/<partition_index>`

> **This step is optional**: Sample partition files can be found in `data/adult/partition`, which can be directly used for future training.

To simulate the federated setting and partition data across clients, run this command:

```
python src/partition.py --partition=diri --dataset adult --n_clients=4 --partition_idx 20 --alpha 0.5
```


##### Parameters:

* `partition`: Parition method.

    *Options*:

    * `diri`: Dirichlet distribution (Non-IID option).

    * `iid`: IID data distribution.

* `alpha`: Parameter controlling the level of data heterogeneity in Dirichlet distribution, where smaller alpha provides a more heterogeneous distribution. alpha values used in our experiments `0.5`, `5`, `500`.

* `dataset`: Dataset name.

* `n_clients`: Number of clients.

* `partition_idx`: Index of the output partition strategy.



---

## Run experiments


To run the experiments, run this file with differnt experimental settings:

```
python src/fairpp_main.py --model=<model> --dataset=<dataset> --partition_idx=<partition_idx>  --idx=<idx>
```

This will initiate a train-test run on each method including:

* Our method:
    * Post-processing: Linear EoD
    * Post-Processing: Final-layer fine-tuning

* Baselines:

    * [FedAvg](https://proceedings.mlr.press/v54/mcmahan17a.html) (Standard Federated Learning) 

    * [FairFed](https://ojs.aaai.org/index.php/AAAI/article/view/25911)

    * FairFed with [Fair Representation](https://dl.acm.org/doi/abs/10.1145/3375627.3375864)


We use the following model strcuture for different dataset:

* Compas & Adult (`adult` & `compas-binary`):
    * Logistic regression: `plain`
    * Two-layer mlp: `plain2`
* PTB-XL ECG dataset (`ptb-xl`):
    * A ResNet-based model structure proposed and implemented by [ResNet-based model for ECG](https://www.nature.com/articles/s41467-021-25351-7): `resNet`
* NIH Chest X-Ray (`nih-chest-eff`):
    * Pre-trained model [MobileNetV2](http://arxiv.org/abs/1801.04381): `mobile`


A sample experiment run for Adult dataset:

```
python src/fairpp_main.py --model plain --dataset adult  \
      --epochs 4  --fairfed_ep 4 --ft_ep 0  \
      --idx 0 --num_users 4  --partition_idx 14 --beta 0.1  --local_bs 32  --ft_bs 256 \
      --ft_alpha 0.1 --ft_alpha2 1.0 --platform local --rep 0 \
      --lr 0.01 --optimizer sgd --ft_lr 0.005 \
      --fair_rep True
```

The experiment result will be saved under `save/statistics/<idx>`.




##### Parameters:

* `model`: 

    *Options*:

    * `plain`: Logistic regression for Adult and Compas dataset.

    * `plain2`: 2-layer model for Adult and Compas dataset.

    * `resNet`: [ResNet-based model for ECG](https://www.nature.com/articles/s41467-021-25351-7)

    * `mobile`: Pre-trained ModileNet from [MobileNetV2](https://arxiv.org/abs/2030.12345)

    * Details see `src/models.py`.


* `dataset`: Dataset name.
    *Options*:

    * `adult`
    * `compas`
    * `ptb-xl`
    * `resNet`


* `fair_rep`: `True` if use fair representation for FairFed

* `platform`: `local` if run experiments locally

* `partition_idx`: index of partition strategy used

* `num_users`: number of clients(users)

* `idx`: index of the experiment


Fedavg Parameters:
* `epochs`: global rounds for FedAvg training

* `local_bs`: local batch size

* `lr`: learning rate

* `optimizer`: optimizer used


FairFed paramters:
* `fairfed_ep`: FairFed global rounds

* `beta`: Fairfed "fairness budget" parameter

Final-layer fine-tuning parameters:

* `ft_ep`: Final-layer fine-tuning epoch

* `ft_bs`: Final-layer fine-tuning batch size

* `ft_alpha`: parameter for calculating fine-tuning loss:  loss = ft_alpha2 * loss + ft_alpha * loss_fairness

* `ft_alpha2`: parameter for calculating fine-tuning loss: loss = ft_alpha2 * loss + ft_alpha * loss_fairness

* `ft_lr`: Final-layer fine-tuning learning rate



<!-- ## Contributing

>📋  Pick a licence and describe how to contribute to your code repository.  -->

