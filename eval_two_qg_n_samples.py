import argparse
import itertools
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ProphetNetForConditionalGeneration, ProphetNetTokenizer

from metrics import PINCscore

QG_MODEL = "microsoft/prophetnet-large-uncased-squad-qg"


def main(args):
    torch_device = args.device
    qg_tokenizer = ProphetNetTokenizer.from_pretrained(QG_MODEL)
    qg_model = ProphetNetForConditionalGeneration.from_pretrained(QG_MODEL)
    qg_model.load_state_dict(torch.load(args.model, map_location=torch.device("cpu")))
    qg_model = qg_model.to(torch_device)
    qg_model.eval()
    pinc = PINCscore(3)

    df = pd.read_csv(args.input, converters={"context": str, "question": str, "answer": str})
    dataset = [f"{row['answer']} [SEP] {row['context']}" for _, row in df.iterrows()]
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, sampler=None, drop_last=False)

    scores = [[] for _ in range(args.num_samples)]
    for data in tqdm(dataloader, total=len(dataloader)):
        qg_inputs = qg_tokenizer(
            data,
            return_tensors="pt",
            truncation=True,
            max_length=300,
            padding=True,
        ).to(torch_device)

        question_ids = qg_model.generate(
            input_ids=qg_inputs["input_ids"],
            attention_mask=qg_inputs["attention_mask"],
            max_length=120,
            num_beams=10,
            num_return_sequences=args.num_samples,
            encoder_no_repeat_ngram_size=args.encoder_no_repeat_ngram_size,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            do_sample=args.do_sample,
            top_p=args.top_p,
        )
        batch_decode = qg_tokenizer.batch_decode(question_ids, skip_special_tokens=True)
        for batch in range(0, len(batch_decode), args.num_samples):
            samples = batch_decode[batch : batch + args.num_samples]
            questions = []
            for i, sample in enumerate(samples):
                matches = re.findall(r"(.*) \[X_SEP\] (.*)", sample)
                if not matches:
                    continue
                questions.append(matches[0][0])
                questions.append(matches[0][1])
                combos = itertools.combinations(questions, 2)
                list1, list2 = list(zip(*combos))
                pinc_scores = pinc.score_two_questions(list1, list2)
                scores[i].append(np.mean(pinc_scores))

    for i in range(1, args.num_samples + 1):
        np.save(f"{args.output}_{2*i}_samples.npy", np.array(scores[i - 1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1", type=str, help="Device to run on")
    parser.add_argument("--input", default="data/squad_v1.1_valid.csv", type=str, help="Dataset to evaluate on")
    parser.add_argument("--output", default="eval/squad_valid.csv", type=str, help="Output file")
    parser.add_argument(
        "--model", default="models/prophetnet_two_qg_10_24_21/pytorch_model.bin", type=str, help="Model to evaluate on"
    )
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--encoder_no_repeat_ngram_size", default=0, type=int, help="encoder_no_repeat_ngram_size")
    parser.add_argument("--no_repeat_ngram_size", default=0, type=int, help="no_repeat_ngram_size")
    parser.add_argument("--do_sample", default=False, action="store_true", help="do_sample")
    parser.add_argument("--top_p", default=1, type=float, help="top_p")
    parser.add_argument("--num_samples", default=4, type=int, help="num_samples")
    args = parser.parse_args()
    main(args)
