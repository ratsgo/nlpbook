import sys
from ratsnlp import nlpbook
from ratsnlp.nlpbook import load_arguments
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from ratsnlp.nlpbook.classification import ClassificationTrainArguments, ClassificationDataset, ClassificationTask
from ratsnlp.nlpbook.paircls import KlueNLICorpus


if __name__ == "__main__":
    # case1 : python train_local.py
    if len(sys.argv) == 1:
        args = ClassificationTrainArguments(
            pretrained_model_name="beomi/kcbert-base",
            downstream_task_name="pair-classification",
            downstream_corpus_root_dir="data",
            downstream_corpus_name="klue-nli",
            force_download=True,
            downstream_model_dir="checkpoint/pair-classification",
            batch_size=32,
            learning_rate=5e-5,
            max_seq_length=64,
            epochs=3,
            seed=7,
        )
    # case2 : python train_local.py train_config.json
    elif len(sys.argv) == 2 and sys.argv[-1].endswith(".json"):
        args = load_arguments(ClassificationTrainArguments, json_file_path=sys.argv[-1])
    # case3 : python train_local.py --pretrained_model_name beomi/kcbert-base --downstream_corpus_root_dir data --downstream_corpus_name nsmc --downstream_task_name document-classification --downstream_model_dir checkpoint/document-classification --do_eval --batch_size 32
    else:
        args = load_arguments(ClassificationTrainArguments)
    nlpbook.set_logger(args)
    nlpbook.download_downstream_dataset(args)
    nlpbook.set_seed(args)
    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model_name,
        do_lower_case=False,
    )
    corpus = KlueNLICorpus()
    train_dataset = ClassificationDataset(
        args=args,
        corpus=corpus,
        tokenizer=tokenizer,
        mode="train",
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset, replacement=False),
        collate_fn=nlpbook.data_collator,
        drop_last=False,
        num_workers=args.cpu_workers,
    )
    val_dataset = ClassificationDataset(
        args=args,
        corpus=corpus,
        tokenizer=tokenizer,
        mode="test",
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(val_dataset),
        collate_fn=nlpbook.data_collator,
        drop_last=False,
        num_workers=args.cpu_workers,
    )
    pretrained_model_config = BertConfig.from_pretrained(
        args.pretrained_model_name,
        num_labels=corpus.num_labels,
    )
    model = BertForSequenceClassification.from_pretrained(
            args.pretrained_model_name,
            config=pretrained_model_config,
    )
    task = ClassificationTask(model, args)
    trainer = nlpbook.get_trainer(args)
    trainer.fit(
        task,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )
