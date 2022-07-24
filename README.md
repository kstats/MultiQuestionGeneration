# Multi-Question Generation

This is the codebase for the paper, [Educational Multi-Question Generation for Reading Comprehension](https://aclanthology.org/2022.bea-1.26/).

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

```
@inproceedings{rathod-etal-2022-educational,
    title = "Educational Multi-Question Generation for Reading Comprehension",
    author = "Rathod, Manav  and
      Tu, Tony  and
      Stasaski, Katherine",
    booktitle = "Proceedings of the 17th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2022)",
    month = jul,
    year = "2022",
    address = "Seattle, Washington",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.bea-1.26",
    pages = "216--223",
    abstract = "Automated question generation has made great advances with the help of large NLP generation models. However, typically only one question is generated for each intended answer. We propose a new task, Multi-Question Generation, aimed at generating multiple semantically similar but lexically diverse questions assessing the same concept. We develop an evaluation framework based on desirable qualities of the resulting questions. Results comparing multiple question generation approaches in the two-question generation condition show a trade-off between question answerability and lexical diversity between the two questions. We also report preliminary results from sampling multiple questions from our model, to explore generating more than two questions. Our task can be used to further explore the educational impact of showing multiple distinct question wordings to students.",
}
```

If you have any questions about this work, feel free to reach out!
