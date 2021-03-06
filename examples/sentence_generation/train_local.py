import sys
from ratsnlp import nlpbook
from Korpora import Korpora
from transformers import GPT2LMHeadModel
from ratsnlp.nlpbook import load_arguments
from ratsnlp.nlpbook.generation import *
from ratsnlp.nlpbook.tokenizers import KoGPT2Tokenizer
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler


if __name__ == "__main__":
    # case1 : python train_local.py
    if len(sys.argv) == 1:
        args = GenerationTrainArguments(
            pretrained_model_name="taeminlee/kogpt2",
            downstream_corpus_root_dir="data",
            downstream_corpus_name="nsmc",
            force_download=True,
            downstream_model_dir="checkpoint/sentence-generation",
            do_eval=True,
            batch_size=96,
            max_seq_length=32,
            epochs=10,
        )
    # case2 : python train_local.py train_config.json
    elif len(sys.argv) == 2 and sys.argv[-1].endswith(".json"):
        args = load_arguments(GenerationTrainArguments, json_file_path=sys.argv[-1])
    # case3 : python train_local.py --pretrained_model_name beomi/kcbert-base --downstream_corpus_root_dir data --downstream_corpus_name nsmc --downstream_task_name document-classification --downstream_model_dir checkpoint/document-classification --do_eval --batch_size 32
    else:
        args = load_arguments(GenerationTrainArguments)
    nlpbook.set_logger(args)
    Korpora.fetch(
        corpus_name=args.downstream_corpus_name,
        root_dir=args.downstream_corpus_root_dir,
        force_download=args.force_download,
    )
    nlpbook.seed_setting(args)
    tokenizer = KoGPT2Tokenizer.from_pretrained(
        args.pretrained_model_name,
    )
    corpus = NsmcCorpus()
    train_dataset = GenerationDataset(
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
        val_dataset = GenerationDataset(
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
    model = GPT2LMHeadModel.from_pretrained(
            args.pretrained_model_name,
    )
    task = GenerationTask(model, args)
    trainer = nlpbook.get_trainer(args)
    trainer.fit(
        task,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )
