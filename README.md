# SDRN
Source code of the paper "Synchronous Double-channel Recurrent Network for Aspect-Opinion Pair Extraction, ACL 2020."

#### Requirement:

```
  python==3.6.8
  torch==0.4.0
  numpy==1.15.4
```

#### Dataset:
14-Res, 14-Lap, 15-Res: Download from https://drive.google.com/drive/folders/1wWK6fIvfYP-54afGDRN44VWlXuUAHs-l?usp=sharing

MPQA：Download from http://www.cs.pitt.edu/mpqa/

JDPA: Download from http://verbs.colorado.edu/jdpacorpus/

#### Download BERT_Base:
https://github.com/google-research/bert

#### How to run:
```
  python main.py --mode train # For training
  python main.py --mode test --test_model ./modelFinal.model # For testing
```
