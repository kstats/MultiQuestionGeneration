import argparse

import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import ProphetNetForConditionalGeneration, ProphetNetTokenizer
from transformers.optimization import AdamW


def main(args):
    # set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = args.device
    tokenizer = ProphetNetTokenizer.from_pretrained(args.model)
    model = ProphetNetForConditionalGeneration.from_pretrained(args.model).to(device)
    model = model.train()

    df = pd.read_csv(args.data)
    dataset = [
        [f"{row['answer']} [SEP] {row['context']}", f"{row['question']} [X_SEP] {row['paraphrase']}"]
        for _, row in df.iterrows()
    ]
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        sampler=RandomSampler(dataset),
        drop_last=True,
    )

    def build_optimizer(model, learning_rate=1e-5):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        return optimizer

    optimizer = build_optimizer(model)

    wandb.init(project="double_qg_prophetnet")
    wandb.watch(model)

    for _ in tqdm(range(10)):
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            contexts, questions = data[0], data[1]
            batch_contexts = tokenizer(contexts, return_tensors="pt", max_length=350, truncation=True, padding=True)
            batch_questions = tokenizer(questions, return_tensors="pt")
            batch_input_ids = batch_contexts["input_ids"].to(device)
            batch_decoder_labels = batch_questions["input_ids"].to(device)
            batch_attention_mask = batch_contexts["attention_mask"].to(device)
            batch_decoder_attention_mask = batch_questions["attention_mask"].to(device)
            output = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_decoder_labels,
                decoder_attention_mask=batch_decoder_attention_mask,
            )
            loss = output["loss"]
            wandb.log({"loss": loss.item()})
            loss.backward()
            if i % 5 == 0:
                optimizer.step()
                optimizer.zero_grad()
            if i % 100 == 0:
                model.eval()
                with torch.no_grad():
                    print(tokenizer.batch_decode(model.generate(batch_input_ids, max_length=130)))
                model.train()

            if i % 1000 == 0:
                model.save_pretrained("checkpoints/")
                tokenizer.save_pretrained("checkpoints/")
                wandb.save("checkpoints/*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/squad_train_paraphrase.csv")
    parser.add_argument("--model", type=str, default="microsoft/prophetnet-large-uncased-squad-qg")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
