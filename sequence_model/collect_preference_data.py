#!/usr/bin/env python3
"""
Collect preference data for S-DPO training.

For SASRec: When the model presents K candidate icons and the user selects one,
this forms a (chosen, K-1 rejected) preference pair.

For Emotion Classifier: When the user corrects a predicted emotion,
this forms a (correct, wrong) preference pair.

Initial data: Simulated from test sequences.
"""

import json
import os
import random
import argparse
from typing import List, Dict, Tuple
from collections import Counter

EMOTION_LIST = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]


def collect_sasrec_preference_data(
    sequences: List[Dict],
    ontology: Dict,
    num_negatives: int = 4,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate S-DPO preference data for SASRec from icon sequences.

    For each sequence, at each position t:
    - The icon at position t is the "chosen" item
    - K-1 other icons are "rejected" items
    """
    rng = random.Random(seed)

    # Build CS role index for hard negative sampling
    cs_role_index = {}
    for icon_id, info in ontology.items():
        cs_role = info.get('cs_role', 'WHAT')
        if cs_role not in cs_role_index:
            cs_role_index[cs_role] = []
        cs_role_index[cs_role].append(icon_id)

    preference_data = []

    for seq in sequences:
        icons = seq['sequence']
        cs_roles = seq.get('cs_roles', [])
        emotion = seq.get('emotion', 'neutral')

        if len(icons) < 2:
            continue

        # For each position, create a preference pair
        for t in range(1, len(icons)):
            chosen = icons[t]
            chosen_cs = cs_roles[t] if t < len(cs_roles) else 'WHAT'

            # Context: icons before position t
            context = icons[:t]
            context_cs = cs_roles[:t]

            # Generate rejected items (hard negatives from same CS role)
            same_role_icons = cs_role_index.get(chosen_cs, [])
            candidates = [i for i in same_role_icons if i != chosen]

            if len(candidates) >= num_negatives:
                rejected = rng.sample(candidates, num_negatives)
            else:
                # Fill with random icons
                rejected = candidates[:]
                all_other = [i for i in ontology.keys() if i != chosen and i not in rejected]
                remaining = num_negatives - len(rejected)
                if remaining > 0 and all_other:
                    rejected.extend(rng.sample(all_other, min(remaining, len(all_other))))

            preference_data.append({
                'prompt': {
                    'sequence': context,
                    'cs_roles': context_cs,
                },
                'chosen': chosen,
                'rejected': rejected[:num_negatives],
                'emotion_context': emotion,
            })

    return preference_data


def collect_emotion_preference_data(
    sequences: List[Dict],
    num_samples: int = 5000,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate DPO preference data for emotion classifier.

    Strategy: Use emotion distributions to create preference pairs
    where the dominant emotion is chosen and a confused emotion is rejected.
    """
    rng = random.Random(seed)
    preference_data = []

    # Confusion pairs (commonly confused emotions)
    confusion_pairs = [
        ('anger', 'disgust'),
        ('sadness', 'fear'),
        ('happiness', 'surprise'),
        ('neutral', 'sadness'),
        ('neutral', 'happiness'),
    ]

    for seq in sequences:
        sentence = seq.get('sentence', '')
        if not sentence:
            continue

        emotion = seq.get('emotion', 'neutral')

        # Find a confused emotion
        rejected_emotion = None
        for e1, e2 in confusion_pairs:
            if emotion == e1:
                rejected_emotion = e2
                break
            elif emotion == e2:
                rejected_emotion = e1
                break

        if rejected_emotion is None:
            # Random different emotion
            other = [e for e in EMOTION_LIST if e != emotion]
            rejected_emotion = rng.choice(other)

        preference_data.append({
            'prompt': sentence,
            'chosen_emotion': emotion,
            'rejected_emotion': rejected_emotion,
        })

    # Subsample if too many
    if len(preference_data) > num_samples:
        preference_data = rng.sample(preference_data, num_samples)

    return preference_data


def main():
    parser = argparse.ArgumentParser(description='Collect DPO preference data')
    parser.add_argument('--ontology', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/AAC2Text/data/processed/aac_full_ontology.json')
    parser.add_argument('--train-sequences', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/data/icon_sequences_train.json')
    parser.add_argument('--val-sequences', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/data/icon_sequences_val.json')
    parser.add_argument('--emotion-data', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/data/sft_train.json')
    parser.add_argument('--output-dir', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/data')
    parser.add_argument('--num-negatives', type=int, default=4)
    parser.add_argument('--num-emotion-samples', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    num_samples = args.num_emotion_samples

    # Emotion confusion pairs (commonly confused emotions)
    confusion_pairs = [
        ('anger', 'disgust'),
        ('sadness', 'fear'),
        ('happiness', 'surprise'),
        ('neutral', 'sadness'),
        ('neutral', 'happiness'),
    ]

    # Load ontology
    with open(args.ontology, 'r') as f:
        ont_data = json.load(f)
    ontology = {}
    for item in ont_data['ontology']:
        if item.get('icon_id'):
            ontology[item['icon_id']] = item

    # Load sequences
    with open(args.train_sequences, 'r') as f:
        train_sequences = json.load(f)
    with open(args.val_sequences, 'r') as f:
        val_sequences = json.load(f)

    print(f"Loaded {len(train_sequences)} train, {len(val_sequences)} val sequences")

    # Generate SASRec preference data
    print("\n=== Generating SASRec preference data ===")
    sasrec_train_prefs = collect_sasrec_preference_data(
        train_sequences, ontology, args.num_negatives, args.seed
    )
    sasrec_val_prefs = collect_sasrec_preference_data(
        val_sequences, ontology, args.num_negatives, args.seed + 1
    )
    print(f"Train preference pairs: {len(sasrec_train_prefs)}")
    print(f"Val preference pairs: {len(sasrec_val_prefs)}")

    # Stats
    cs_role_counts = Counter(p['prompt']['cs_roles'][-1] if p['prompt']['cs_roles'] else 'N/A'
                             for p in sasrec_train_prefs)
    print(f"CS role distribution in last position:")
    for role, count in cs_role_counts.most_common():
        print(f"  {role}: {count}")

    # Save SASRec preference data
    os.makedirs(args.output_dir, exist_ok=True)
    for name, data in [('sasrec_dpo_train', sasrec_train_prefs), ('sasrec_dpo_val', sasrec_val_prefs)]:
        path = os.path.join(args.output_dir, f'{name}.json')
        with open(path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} pairs to {path}")

    # Generate emotion classifier preference data
    print("\n=== Generating Emotion Classifier preference data ===")
    if os.path.exists(args.emotion_data):
        with open(args.emotion_data, 'r') as f:
            emotion_data = json.load(f)

        # Convert sft format to preference format
        emotion_prefs = []
        for entry in emotion_data:
            main_emotion = entry.get('main_emotion', 'neutral')
            # Get text from conversation
            conversation = entry.get('conversation', '')
            if isinstance(conversation, str):
                # Try to extract user utterances
                import re
                user_matches = re.findall(r"'content':\s*'([^']*)'", conversation)
                if user_matches:
                    text = ' '.join(user_matches[:3])
                else:
                    text = conversation[:200]
            elif isinstance(conversation, list):
                text = ' '.join(t.get('content', '') for t in conversation if t.get('role') == 'user')
            else:
                continue

            if not text.strip():
                continue

            # Find a confused emotion
            rejected_emotion = None
            for e1, e2 in confusion_pairs:
                if main_emotion == e1:
                    rejected_emotion = e2
                    break
                elif main_emotion == e2:
                    rejected_emotion = e1
                    break

            if rejected_emotion is None:
                other = [e for e in EMOTION_LIST if e != main_emotion]
                rejected_emotion = rng.choice(other)

            emotion_prefs.append({
                'prompt': text,
                'chosen_emotion': main_emotion,
                'rejected_emotion': rejected_emotion,
            })

        # Subsample if too many
        if len(emotion_prefs) > num_samples:
            emotion_prefs = rng.sample(emotion_prefs, num_samples)

        cls_prefs = emotion_prefs
        print(f"Emotion preference pairs: {len(cls_prefs)}")

        # Split 90/10
        split = int(len(cls_prefs) * 0.9)
        cls_train = cls_prefs[:split]
        cls_val = cls_prefs[split:]

        for name, data in [('cls_dpo_train', cls_train), ('cls_dpo_val', cls_val)]:
            path = os.path.join(args.output_dir, f'{name}.json')
            with open(path, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(data)} pairs to {path}")
    else:
        print(f"Emotion data not found at {args.emotion_data}, skipping")


if __name__ == '__main__':
    main()
