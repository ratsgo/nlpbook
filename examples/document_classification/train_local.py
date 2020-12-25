from google.colab import drive
from Korpora import Korpora
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from ratsnlp import nlpbook
from ratsnlp.nlpbook.classification import NsmcCorpus, ClassificationDataset, ClassificationTask


if __name__ == "__main__":
    drive.mount('/gdrive', force_remount=True)
    args = nlpbook.TrainArguments(
        pretrained_model_name="beomi/kcbert-base",
        downstream_corpus_root_dir="/root/Korpora",
        downstream_corpus_name="nsmc",
        downstream_task_name="document-classification",
        downstream_model_dir="/gdrive/My Drive/nlpbook/checkpoint-cls",
        do_eval=True,
        batch_size=32,
    )
    nlpbook.set_logger(args)
    Korpora.fetch(
        corpus_name=args.downstream_corpus_name,
        root_dir=args.downstream_corpus_root_dir,
        force_download=True,
    )
    nlpbook.seed_setting(args)
    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model_name,
        do_lower_case=False,
    )
    corpus = NsmcCorpus()
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
    if args.do_eval:
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
    else:
        val_dataloader = None
    pretrained_model_config = BertConfig.from_pretrained(
        args.pretrained_model_name,
        num_labels=corpus.num_labels,
    )
    model = BertForSequenceClassification.from_pretrained(
            args.pretrained_model_name,
            config=pretrained_model_config,
    )
    task = ClassificationTask(model, args)
    _, trainer = nlpbook.get_trainer(args)
    trainer.fit(
        task,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )