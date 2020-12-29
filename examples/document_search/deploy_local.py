import sys
import torch
from ratsnlp.nlpbook.search import *
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertConfig, BertTokenizer, BertModel
from ratsnlp.nlpbook import load_arguments, data_collator, set_logger, download_downstream_dataset


if __name__ == "__main__":
    # case1 : python deploy_local.py
    if len(sys.argv) == 1:
        args = SearchDeployArguments(
            pretrained_model_name="beomi/kcbert-base",
            # downstream_model_checkpoint_path="checkpoint/document-search/last.ckpt",
            downstream_model_checkpoint_path="/Users/david/Downloads/last.ckpt",
            downstream_corpus_root_dir="data",
            downstream_corpus_name="korquad-v1",
            # passage_embedding_dir="checkpoint/document-search",
            passage_embedding_dir="/Users/david/Downloads",
            question_max_seq_length=32,
            passage_max_seq_length=256,
            batch_size=128 if torch.cuda.is_available() else 1,
            threshold=0.95,
            top_k=5,
        )
    # case2 : python deploy_local.py deploy_config.json
    elif len(sys.argv) == 2 and sys.argv[-1].endswith(".json"):
        args = load_arguments(SearchDeployArguments, json_file_path=sys.argv[-1])
    # case3 : python deploy_local.py --pretrained_model_name beomi/kcbert-base --downstream_model_checkpoint_path checkpoint/document-classification/epoch=10.ckpt --downstream_task_name document-classification --max_seq_length 128
    else:
        args = load_arguments(SearchDeployArguments)
    set_logger(args)
    download_downstream_dataset(args)
    fine_tuned_model_ckpt = torch.load(
        args.downstream_model_checkpoint_path,
        map_location=torch.device("cpu")
    )
    pretrained_model_config = BertConfig.from_pretrained(args.pretrained_model_name)
    question_tower = BertModel.from_pretrained(
        args.pretrained_model_name,
        config=pretrained_model_config,
    )
    passage_tower = BertModel.from_pretrained(
        args.pretrained_model_name,
        config=pretrained_model_config,
    )
    model = SearchModelForInference(
        question_tower=question_tower,
        passage_tower=passage_tower,
    )
    model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model_name,
        do_lower_case=False,
    )
    corpus = KorQuADV1Corpus()
    inference_dataset = SearchDataset(
        args=args,
        corpus=corpus,
        tokenizer=tokenizer,
        mode="inference",
    )
    inference_dataloader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(inference_dataset),
        collate_fn=data_collator,
        drop_last=False,
        num_workers=args.cpu_workers,
    )
    all_passages, all_passage_embeddings = encoding_passage(
        inference_dataloader=inference_dataloader,
        model=model,
        tokenizer=tokenizer,
        args=args,
    )
    if torch.cuda.is_available():
        model.cuda()
        all_passage_embeddings.cuda()

    def inference_fn(question):
        if question:
            inputs = tokenizer(
                [question],
                max_length=args.question_max_seq_length,
                padding="max_length",
                truncation=True,
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                scores = model(**{**{f"question_{k}": torch.tensor(v) for k, v in inputs.items()},
                                  "passage_embeddings": all_passage_embeddings,
                                  "mode": "inference"})
                top_scores = torch.topk(scores, dim=1, k=args.top_k)
                results = [{"passage": all_passages[idx.item()], "score": str(round(score.item(), 4))}
                           for idx, score in zip(top_scores.indices[0], top_scores.values[0])
                           if score > args.threshold]
        if not question or len(results) == 0:
            results = [{"passage": "적절한 문서가 없습니다", "score": str(0.0000)}]
        return {
            'question': question,
            'results': results,
        }

    app = get_web_service_app(inference_fn, is_colab=False)
    app.run()
