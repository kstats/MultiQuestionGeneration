import torch
from transformers import pipeline
from transformers.data.metrics.squad_metrics import compute_f1


class QAMetric:
    def __init__(
        self,
        model="bert-large-uncased-whole-word-masking-finetuned-squad",
        device="cpu",
    ):
        self.question_answerer = pipeline("question-answering", model=model, device=-1)
        self.device = device

    def score(self, contexts, questions, answers, **kwargs):
        self.question_answerer.device = torch.device(self.device)
        self.question_answerer.model = self.question_answerer.model.to(self.device)
        preds = self.question_answerer(context=contexts, question=questions)
        scores = []
        for i, pred in enumerate(preds):
            score = compute_f1(answers[i], pred["answer"])
            scores.append(score)
        self.question_answerer.model = self.question_answerer.model.cpu()
        self.question_answerer.device = torch.device("cpu")
        return scores
