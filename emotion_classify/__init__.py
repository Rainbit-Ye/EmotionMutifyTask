# Emotion Classification Module

Emotion classification and prediction models.

## Files

| File | Description |
|------|-------------|
| `cls_trainer.py` | LoRA fine-tuning |
| `cls_multitask_trainer.py` | Multi-task training with next emotion prediction |
| `simple_trainer.py` | Full parameter training |
| `cls_evaluate.py` | Model evaluation |
| `cls_inference.py` | Model inference |
| `dynamic_emotion_analyzer.py` | Dynamic emotion analysis |
| `evaluate_full_comparison.py` | Full comparison evaluation |
| `evaluate_next_emotion.py` | Next emotion prediction evaluation |

## Usage

```python
import sys
sys.path.append('../emotion_classify')

from dynamic_emotion_analyzer import DynamicEmotionAnalyzer
```
