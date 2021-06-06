import sys
from ratsnlp import nlpbook
from Korpora import Korpora
from ratsnlp.nlpbook.generation import *
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast


if __name__ == "__main__":
    # case1 : python train_local.py
    if len(sys.argv) == 1:
        args = GenerationTrainArguments(
            pretrained_model_name="skt/kogpt2-base-v2",
            downstream_corpus_root_dir="data",
            downstream_corpus_name="nsmc",
            force_download=True,
            downstream_model_dir="checkpoint/sentence-generation",
            batch_size=32,
            max_seq_length=32,
            learning_rate=5e-5,
            epochs=3,
            seed=7,
        )
    # case2 : python train_local.py train_config.json
    elif len(sys.argv) == 2 and sys.argv[-1].endswith(".json"):
        args = nlpbook.load_arguments(GenerationTrainArguments, json_file_path=sys.argv[-1])
    # case3 : python train_local.py --pretrained_model_name beomi/kcbert-base --downstream_corpus_root_dir data --downstream_corpus_name nsmc --downstream_task_name document-classification --downstream_model_dir checkpoint/document-classification --do_eval --batch_size 32
    else:
        args = nlpbook.load_arguments(GenerationTrainArguments)
    nlpbook.set_logger(args)
    Korpora.fetch(
        corpus_name=args.downstream_corpus_name,
        root_dir=args.downstream_corpus_root_dir,
        force_download=args.force_download,
    )
    nlpbook.set_seed(args)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        args.pretrained_model_name,
        eos_token='</s>',
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
