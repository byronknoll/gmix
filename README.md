# gmix
https://github.com/byronknoll/gmix

gmix is an attempt to build a large language model using techniques from the field of lossless data compression. LLMs typically use transformers (end-to-end neural networks). gmix uses a substantially different architecture: an ensemble of different types of predictive compression models combined with a [gated linear network](https://arxiv.org/abs/1712.01897v1). gmix does not use GPUs - training/inference uses a single CPU thread. gmix is free software distributed under the GNU General Public License. Feel free to contact me at byron@byronknoll.com if you have any questions.

gmix can be used for:

1. **Generating text**: given a pretrained model and a prompt, gmix can generate new data.
1. **Training**: models can be trained from scratch on any type of data.
1. **Fine-tuning**: given a pretrained model, the model can be fine-tuned on a new training set.
1. **Lossless data compression**: files can be compressed, either using a pretrained model or from scratch.

gmix is a successor to [cmix](https://www.byronknoll.com/cmix.html). It has a similar architecture to cmix, with the following design differences:

1. Ability to serialize memory to a file on disk. This allows compression checkpointing (saving progress and resuming later) and sharing trained models.
1. Ability to generate new data based on a trained model (i.e. LLM-style text generation). cmix is not good at text generation because there is no way to disable training/learning, which causes the output to quickly spiral into repetitive cycles. With gmix, learning can be disabled during text generation.
1. Models are designed to scale to large amounts of training data.
1. All models are designed to generalize to different types of data. In cmix, some models are specialized for specific file types or benchmarks. For example, cmix hardcodes knowledge like "words are usually separated by the space character", while gmix doesn't assume that specific bytes have special meanings.
1. Improved readability. Cleaner design and source code comments compared to cmix.

gmix is still early in development:
1. Its compression rate is not competitive with cmix yet.
1. Pretrained models (trained using large amounts of training data) are not ready to be shared yet.

Use "make" to compile gmix. In Windows, gmix can be compiled with MinGW (http://nuwen.net/mingw.html) or Cygwin (https://www.cygwin.com).