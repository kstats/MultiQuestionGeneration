import argparse

import numpy as np
import pandas as pd

from metrics import PINCscore, QAMetric, SBERTScore


def main(args):
    print(args.input)
    df = pd.read_csv(args.input)
    df = df.fillna("?")

    contexts = df["context"].tolist()
    answers = df["answer"].tolist()
    questions1 = df["gen_question1"].tolist()
    questions2 = df["gen_question2"].tolist()

    pinc = PINCscore(3)
    inter_question_pinc_scores = pinc.score_two_questions(questions1, questions2)
    context_question1_pinc_scores = pinc.score(contexts, questions1, answers)
    context_question2_pinc_scores = pinc.score(contexts, questions2, answers)

    df["inter_question_pinc_score"] = inter_question_pinc_scores
    df["context_question1_pinc_score"] = context_question1_pinc_scores
    df["context_question2_pinc_score"] = context_question2_pinc_scores

    print(
        "AVG & STD INTER QUESTION PINC SCORE:",
        np.mean(df["inter_question_pinc_score"]),
        np.std(df["inter_question_pinc_score"]),
    )
    print(
        "AVG & STD CONTEXT QUESTION1 PINC SCORE:",
        np.mean(df["context_question1_pinc_score"]),
        np.std(df["context_question1_pinc_score"]),
    )
    print(
        "AVG & STD CONTEXT QUESTION2 PINC SCORE:",
        np.mean(df["context_question2_pinc_score"]),
        np.std(df["context_question2_pinc_score"]),
    )

    sbert_score = SBERTScore(device=args.device)
    sbert_similarity = sbert_score.score_two_questions(questions1, questions2)
    df["sbert_similarity"] = sbert_similarity
    print("AVG & STD SBERT SIMILARITY:", np.mean(df["sbert_similarity"]), np.std(df["sbert_similarity"]))

    qa = QAMetric(device=args.device)
    questions1_qa_scores = qa.score(contexts, questions1, answers)
    questions2_qa_scores = qa.score(contexts, questions2, answers)

    print("AVG & STD QUESTIONS1 QA SCORE:", np.mean(df["questions1_qa_score"]), np.std(df["questions1_qa_score"]))
    print("AVG & STD QUESTIONS2 QA SCORE:", np.mean(df["questions2_qa_score"]), np.std(df["questions2_qa_score"]))
    print("-" * 50)

    df["questions1_qa_score"] = questions1_qa_scores
    df["questions2_qa_score"] = questions2_qa_scores
    df.to_csv(args.input, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate questions")
    parser.add_argument("--input", type=str, help="csv of generated questions")
    parser.add_argument("--device", type=str, default="cuda:1", help="device to use")
    args = parser.parse_args()
    main(args)
