python eval_two_qg_model.py --input data/squad_v1.1_valid.csv --output eval/base_prophetnet_two_qg.csv --model models/prophetnet_two_qg_10_24_21/pytorch_model.bin --batch_size 16
python eval_two_qg_model.py --input data/squad_v1.1_valid.csv --output eval/base_prophetnet_two_qg_sample.csv --model models/prophetnet_two_qg_10_24_21/pytorch_model.bin --batch_size 16 --do_sample --top_p 0.95
python eval_two_qg_model.py --input data/squad_v1.1_valid.csv --output eval/no_encoder_trigram_repeat_prophetnet_two_qg.csv --model models/prophetnet_two_qg_10_24_21/pytorch_model.bin --batch_size 16 --encoder_no_repeat_ngram_size 3
python eval_two_qg_model.py --input data/squad_v1.1_valid.csv --output eval/no_question_trigram_repeat_prophetnet_two_qg.csv --model models/prophetnet_two_qg_10_24_21/pytorch_model.bin --batch_size 16 --no_repeat_ngram_size 3
python eval_two_qg_model.py --input data/squad_v1.1_valid.csv --output eval/no_encoder_question_trigram_repeat_prophetnet_two_qg.csv --model models/prophetnet_two_qg_10_24_21/pytorch_model.bin --batch_size 16 --no_repeat_ngram_size 3 --encoder_no_repeat_ngram_size 3
python eval_base_qg_para_model.py --input data/squad_v1.1_valid.csv --output eval/base_prophetnet_para.csv --batch_size 16
python eval_base_qg_model.py --input data/squad_v1.1_valid.csv --output eval/base_prophetnet_sample.csv --batch_size 16 --top_p 0.95
python eval_questions.py --input eval/base_prophetnet_sample.csv
python eval_questions.py --input eval/base_prophetnet_para.csv
python eval_questions.py --input eval/base_prophetnet_two_qg.csv
python eval_questions.py --input eval/base_prophetnet_two_qg_sample.csv
python eval_questions.py --input eval/no_encoder_trigram_repeat_prophetnet_two_qg.csv
python eval_questions.py --input eval/no_question_trigram_repeat_prophetnet_two_qg.csv
python eval_questions.py --input eval/no_encoder_question_trigram_repeat_prophetnet_two_qg.csv