"""
SASRec: Self-Attentive Sequential Recommendation for AAC Icon Prediction

Reference: Kang & McAuley, "Self-Attentive Sequential Recommendation", ICLR 2018

Adapted for AAC icon sequences with Colourful Semantics (CS) role embeddings.
"""

from .sasrec import SASRec, CausalSelfAttentionBlock, SASRecDataset
from .fusion import FusedIconPredictor
