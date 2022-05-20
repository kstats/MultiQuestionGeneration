# Multi-Question Generation

This is the codebase for the paper, Educational Multi-Question Generation for Reading Comprehension.

## Data

We use the SQuAD v1.1 dataset for our experiments augmented with paraphrases of each of the given questions. We release this data [here](https://drive.google.com/drive/folders/1j5qSvzOlO_OftB_m5L6ZKPuQvEUFXxnn?usp=sharing).

## Setup

Run the following commands to create all of the necessary folders and install the depenedencies:

```bash
mkdir eval
pip install -r requirements.txt
```

## 2QG Training

To train the 2QG model, we use the following command:

```bash
python prophetnet_twoq_finetune.py
```

Training takes about 2 days. Hence, we also release our trained model. It is located in the above google drive folder.

## Evaluation

We perform evaluation on the SQuAD validation set. For evaluating the different models as in the paper, we provide the bash script `eval_experiments.sh`. The script needs to be modified with the correct path to trained 2QG model.

To conduct the analysis shown in the section Toward Multi-Question Generation, we use the following command:

```bash
python eval_two_qg_n_samples.py --model <path_to_trained_eq_model> --num_samples <number_of_samples>
```

This will generated `.npy` files with containing the PINC scores. You can then generated the boxplot distribution using the command:

```bash
python make_boxplot.py
```

## Citation

If you extend or use this work, please cite our paper:

#TODO: Add citation

If you have any questions about this work, feel free to reach out!