# gmix
https://github.com/byronknoll/gmix

gmix is a lossless data compression program aimed at optimizing compression ratio at the cost of high CPU/memory usage. gmix is free software distributed under the GNU General Public License. Feel free to contact me at byron@byronknoll.com if you have any questions.

gmix is a successor to [cmix](https://github.com/byronknoll/cmix). It has a similar compression architecture to cmix, with the following design differences:

1. Ability to serialize memory to a file on disk. This allows compression checkpointing (saving progress and resuming later) and sharing trained models.
1. Ability to generate new data based on a trained model (i.e. LLM-style text generation). cmix is not good at text generation because there is no way to disable training/learning, which causes the output to quickly spiral into repetitive cycles. With gmix, learning can be disabled during text generation.
1. Models are designed to scale to large amounts of training data.
1. All models are designed to generalize to different types of data. In cmix, some models are specialized for specific filetypes or benchmarks.
1. Improved readability. Cleaner design and source code comments compared to cmix.

Due to the changed memory architecture, models from other programs (e.g. cmix/PAQ8) need to be rewritten in order to be used in gmix. gmix is still early in development, so it doesn't have a competitive compression rate yet.
