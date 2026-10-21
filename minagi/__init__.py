"""
mini-AGI: a character-level model that assembles its own architecture.

The pieces, in the order they depend on each other:

    config      reading config.yaml, which both building and training read
    precision   what the model computes in, and how moments are stored
    tokenizer   bytes in, bytes out - 256 values plus structural markers
    model       the transformer: RMSNorm, rotary positions, SwiGLU, flash
                attention, tied embeddings
    decode      how a character is chosen, without a random number generator
    pool        the expert pool, and the rules by which it grows and shrinks
    paged       the same pool spread over disk, RAM and VRAM
    recur       latent recurrence with adaptive depth, built on model + pool
    stream      reading a corpus behind a KV cache, one chunk at a time
    ingest      turning a pile of files into something to read
    ingestd     scheduled, opt-in file ingestion
    agency      sandboxed tool registry, permissions, and receipts
    corpus      the fixed-window view, for the batch trainer and benchmarks
    store       the weights directory, which IS the model
    optim       optimisers for the trunk, and the gradient noise scale
    plasticity  the learning rate, governed by held-out loss
    schedule    the fixed cosine schedule, for the batch trainer
    live        serving a model that is being trained underneath
    retention   durable held-out readings for the long-run dashboard
    report      the model reading statistics off its own weights
    notify      telling someone when a long run needs attention

Nothing here has a command line except the ingestion daemon's `main`.
`train.py`, `serve.py`, and `ingestd.py` are the entry points, alongside the
`benchmarks`, `corpora` and `experiments` packages.
"""

__all__ = ["tokenizer", "model", "decode", "pool", "recur", "stream",
           "corpus", "store", "report", "agency", "retention"]
