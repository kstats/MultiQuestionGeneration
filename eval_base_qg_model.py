import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ProphetNetForConditionalGeneration, ProphetNetTokenizer

QG_MODEL = "microsoft/prophetnet-large-uncased-squad-qg"


def main(args):
    # set random seed
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    torch_device = args.device
    qg_tokenizer = ProphetNetTokenizer.from_pretrained(QG_MODEL)
    qg_model = ProphetNetForConditionalGeneration.from_pretrained(QG_MODEL)
    qg_model = qg_model.to(torch_device)
    qg_model.eval()

    df = pd.read_csv(args.input, converters={"context": str, "question": str, "answer": str})
    dataset = [f"{row['answer']} [SEP] {row['context']}" for _, row in df.iterrows()]
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, sampler=None, drop_last=False)

    all_questions = []
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
            num_return_sequences=2,
            top_p=args.top_p,
        )
        questions = qg_tokenizer.batch_decode(question_ids, skip_special_tokens=True)
        all_questions.extend(questions)

    df["gen_question1"] = all_questions[::2]
    df["gen_question2"] = all_questions[1::2]
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="input file")
    parser.add_argument("--output", type=str, help="Output file")
    parser.add_argument("--device", default="cuda:1", type=str, help="Device to run on")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--top_p", default=0.95, type=float, help="top_p")
    args = parser.parse_args()
    main(args)
