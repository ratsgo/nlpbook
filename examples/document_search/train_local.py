import sys
from ratsnlp import nlpbook
from ratsnlp.nlpbook.search import *
from ratsnlp.nlpbook import load_arguments
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertConfig, BertTokenizer, BertModel


if __name__ == "__main__":
    # case1 : python train_local.py
    if len(sys.argv) == 1:
        args = SearchTrainArguments(
            pretrained_model_name="beomi/kcbert-base",
            downstream_corpus_root_dir="data",
            downstream_corpus_name="korquad-v1",
            force_download=False,
            downstream_model_dir="checkpoint/document-search",
            question_max_seq_length=32,
            passage_max_seq_length=256,
            batch_size=32,
            epochs=5,
        )
    # case2 : python train_local.py train_config.json
    elif len(sys.argv) == 2 and sys.argv[-1].endswith(".json"):
        args = load_arguments(SearchTrainArguments, json_file_path=sys.argv[-1])
    # case3 : python train_local.py --pretrained_model_name beomi/kcbert-base --downstream_corpus_root_dir data --downstream_corpus_name korean_chatbot_data --downstream_model_dir checkpoint/document-search
    else:
        args = load_arguments(SearchTrainArguments)
    nlpbook.set_logger(args)
    nlpbook.download_downstream_dataset(args)
    nlpbook.seed_setting(args)
    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model_name,
        do_lower_case=False,
    )
    corpus = KorQuADV1Corpus()
    dataset = SearchDataset(
        args=args,
        tokenizer=tokenizer,
        corpus=corpus,
        mode="train",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(dataset, replacement=False),
        collate_fn=search_train_collator,
        drop_last=False,
        num_workers=args.cpu_workers,
    )
    pretrained_model_config = BertConfig.from_pretrained(
        args.pretrained_model_name,
    )
    question_tower = BertModel.from_pretrained(
        args.pretrained_model_name,
        config=pretrained_model_config,
    )
    passage_tower = BertModel.from_pretrained(
        args.pretrained_model_name,
        config=pretrained_model_config,
    )
    model = SearchModelForTrain(
        question_tower=question_tower,
        passage_tower=passage_tower,
    )
    task = SearchTask(model, args)
    trainer = nlpbook.get_trainer(args, eval=False)
    trainer.fit(
        task,
        train_dataloader=dataloader,
    )
