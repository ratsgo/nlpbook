import sys
from ratsnlp import nlpbook
from ratsnlp.nlpbook import load_arguments
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import BertConfig, BertTokenizer, BertForTokenClassification
from ratsnlp.nlpbook.ner import NERTrainArguments, NERCorpus, NERDataset, NERTask


if __name__ == "__main__":
    # case1 : python train_local.py
    if len(sys.argv) == 1:
        args = NERTrainArguments(
            pretrained_model_name="beomi/kcbert-base",
            downstream_corpus_root_dir="data",
            downstream_corpus_name="ner",
            downstream_model_dir="checkpoint/ner",
            do_eval=True,
            batch_size=32,
            epochs=30,
        )
    # case2 : python train_local.py train_config.json
    elif len(sys.argv) == 2 and sys.argv[-1].endswith(".json"):
        args = load_arguments(NERTrainArguments, json_file_path=sys.argv[-1])
    # case3 : python train_local.py --pretrained_model_name beomi/kcbert-base --downstream_corpus_root_dir data --downstream_corpus_name ner
    else:
        args = load_arguments(NERTrainArguments)
    nlpbook.set_logger(args)
    nlpbook.download_downstream_dataset(args)
    nlpbook.seed_setting(args)
    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model_name,
        do_lower_case=False,
    )
    corpus = NERCorpus(args)
    train_dataset = NERDataset(
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
        val_dataset = NERDataset(
            args=args,
            corpus=corpus,
            tokenizer=tokenizer,
            mode="val",
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
        args.pretrained_model_cache_dir,
        num_labels=corpus.num_labels,
    )
    model = BertForTokenClassification.from_pretrained(
            args.pretrained_model_name,
            config=pretrained_model_config,
    )
    task = NERTask(model, args)
    trainer = nlpbook.get_trainer(args)
    trainer.fit(
        task,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )
