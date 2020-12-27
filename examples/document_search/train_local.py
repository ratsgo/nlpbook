import sys
from ratsnlp import nlpbook
from Korpora import Korpora
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
            downstream_corpus_name="korean_chatbot_data",
            force_download=False,
            downstream_model_dir="checkpoint/document-search",
            max_seq_length=48,
            batch_size=32,
            epochs=30,
        )
    # case2 : python train_local.py train_config.json
    elif len(sys.argv) == 2 and sys.argv[-1].endswith(".json"):
        args = load_arguments(SearchTrainArguments, json_file_path=sys.argv[-1])
    # case3 : python train_local.py --pretrained_model_name beomi/kcbert-base --downstream_corpus_root_dir data --downstream_corpus_name korean_chatbot_data --downstream_model_dir checkpoint/document-search
    else:
        args = load_arguments(SearchTrainArguments)
    nlpbook.set_logger(args)
    Korpora.fetch(
        corpus_name=args.downstream_corpus_name,
        root_dir=args.downstream_corpus_root_dir,
        force_download=args.force_download,
    )
    nlpbook.seed_setting(args)
    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model_name,
        do_lower_case=False,
    )
    corpus = KoreanChatbotDataCorpus()
    train_dataset = SearchDataset(
        args=args,
        corpus=corpus,
        tokenizer=tokenizer,
        mode="train",
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset, replacement=False),
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
    model = SearchModel(
        question_tower=question_tower,
        passage_tower=passage_tower,
    )
    task = SearchTask(model, args)
    trainer = nlpbook.get_trainer(args, eval=False)
    trainer.fit(
        task,
        train_dataloader=train_dataloader,
    )
