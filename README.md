<h1 align="center">NLP Projects</h1>
<p align="center">
    <img src="https://img.shields.io/badge/Made%20With-PyTorch-blue"></img>
</p>
<p align="center">Neural Models for some NLP related tasks implemented using PyTorch</p>

***

Following are some of the NLP related tasks that I have tried to implement using PyTorch:

- [Neural Machine Translation](#neural-machine-translation)
- [Semantic Slot Filling](#semantic-slot-filling)

## Neural Machine Translation 

Machine translation is the task of automatically converting source text in one language to text in another language. Neural machine translation, or NMT for short, is the use of neural network models to learn a statistical model for machine translation.

NMT generally uses an Encoder-Decoder RNN structure. To increase the performance, Decoders also use something known as _Attention mechanism_ which enables them to learn to focus on certain portions of the input sentence in order to predict the correct output word in the sequence.

The NMT Model implemented by me uses an Encoder + Attention-based Decoder to try and translate English sentences to Hindi.

The __dataset__ was obtained from [__Tab-delimited Bilingual Sentence Pairs__](http://www.manythings.org/anki/) & can be found [here](https://github.com/tezansahu/NLP_Projects/blob/master/Eng-Hin%20Machine%20Translation%20using%20Seq2Seq/hin-eng.txt).

Here is the [__IPython Notebook__ for the project](https://github.com/tezansahu/NLP_Projects/blob/master/Eng-Hin%20Machine%20Translation%20using%20Seq2Seq/Eng_to_Hin_Translation_using_Seq2Seq.ipynb).

The __trained models__ can be found here: [Encoder](https://github.com/tezansahu/NLP_Projects/blob/master/Eng-Hin%20Machine%20Translation%20using%20Seq2Seq/eng_to_hin_encoder) & [Decoder](https://github.com/tezansahu/NLP_Projects/blob/master/Eng-Hin%20Machine%20Translation%20using%20Seq2Seq/eng_to_hin_attn_decoder).

__References:__
- [Neural Machine Translation by Jointly Learning to Align & Translate](https://arxiv.org/pdf/1409.0473.pdf) by Bahdanau, Cho & Bengio
- [NLP from Scratch: Translation with a Sequence to Sequence Network & Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)



## Semantic Slot Filling

Semantic slot filling is one of the most challenging problems in spoken language understanding (SLU). __Intent Detection__ and __Slot Filling__ is the task of interpreting user commands/queries by extracting the intent and the relevant slots. The semantic parsing of input utterances in SLU typically consists of three tasks: domain detection, intent determination, and slot filling.

The task at hand is to annotate (tag) each word of a query whether it belongs to a specific item of information (slot), and which one.

The model that I have tried to implement is a recurrent model consisting of an embedding layer, a bidirectional GRU cell, and a dense layer to compute the posterior probabilities.

__SNIPS__ is a dataset by Snips.ai for Intent Detection and Slot Filling benchmarking available from the [github page](https://github.com/snipsco/nlu-benchmark). This dataset contains several day to day user command categories (e.g. play a song, book a restaurant). I have used a slightly pre-processed version of this dataset, which can be found [here](https://github.com/tezansahu/NLP_Projects/tree/master/Semantic%20Slot%20Filling/snips).

Here is the [__IPython Notebook__ for the project](https://github.com/tezansahu/NLP_Projects/blob/master/Semantic%20Slot%20Filling/Semantic_Slot_Filling.ipynb)

The __trained model__ can be found [here](https://github.com/tezansahu/NLP_Projects/blob/master/Semantic%20Slot%20Filling/snips_slot_filling_model).

__References:__
- [Using Recurrent Neural Networks for Slot Filling in Spoken Language Understanding](http://www.iro.umontreal.ca/~lisa/pointeurs/taslp_RNNSLU_final_doubleColumn.pdf)
- [Snips Voice Platform: an embedded Spoken Language Understanding system for private-by-design voice interfaces](https://arxiv.org/pdf/1805.10190.pdf)
- [CNTK 202: Language Understanding with Recurrent Networks](https://www.cntk.ai/pythondocs/CNTK_202_Language_Understanding.html)
- [Pad pack sequences for Pytorch batch processing with DataLoader](https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html)

***
<p align='center'>Created with :heart: by <a href="https://www.linkedin.com/in/tezan-sahu/">Tezan Sahu</a></p>
