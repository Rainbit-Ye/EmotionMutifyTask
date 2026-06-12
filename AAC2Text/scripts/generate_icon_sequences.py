#!/usr/bin/env python3
"""
Generate synthetic icon sequences for SASRec training.

Data sources:
1. Parse existing training_data.json (30K entries) into CS-annotated sequences
2. Template-based augmentation from ontology (CS role patterns)
3. Emotion-conditioned sequences from sv_emo/svo_emo types
4. Negative sampling for hard negatives

Output:
  data/icon_sequences_train.json (~50K)
  data/icon_sequences_val.json (~5K)
  data/icon_sequences_test.json (~5K)
"""

import json
import os
import random
import argparse
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

# CS role constants
CS_WHO = 'WHO'
CS_WHAT_DOING = 'WHAT_DOING'
CS_WHAT = 'WHAT'
CS_WHERE = 'WHERE'
CS_WHEN = 'WHEN'
CS_HOW = 'HOW'
CS_ROLES = [CS_WHO, CS_WHAT_DOING, CS_WHAT, CS_WHERE, CS_WHEN, CS_HOW]

# CS templates: pattern name -> list of CS role sequences
CS_TEMPLATES = {
    'sv':        [[CS_WHO, CS_WHAT_DOING]],
    'svo':       [[CS_WHO, CS_WHAT_DOING, CS_WHAT]],
    'sv_time':   [[CS_WHO, CS_WHAT_DOING, CS_WHEN]],
    'svo_time':  [[CS_WHO, CS_WHAT_DOING, CS_WHAT, CS_WHEN]],
    'sv_emo':    [[CS_WHO, CS_WHAT_DOING, CS_HOW]],
    'svo_emo':   [[CS_WHO, CS_WHAT_DOING, CS_WHAT, CS_HOW]],
    'svo_place': [[CS_WHO, CS_WHAT_DOING, CS_WHAT, CS_WHERE]],
    # Additional templates not in original data
    'sv_where':  [[CS_WHO, CS_WHAT_DOING, CS_WHERE]],
    'sv_how':    [[CS_WHO, CS_WHAT_DOING, CS_HOW]],
    'svo_how_where': [[CS_WHO, CS_WHAT_DOING, CS_WHAT, CS_HOW, CS_WHERE]],
    'who_what':  [[CS_WHO, CS_WHAT]],
    'what_doing_what': [[CS_WHAT_DOING, CS_WHAT]],
}

# Emotion list
EMOTIONS = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]


def load_ontology(ontology_path: str) -> Tuple[Dict, Dict[str, List]]:
    """Load ontology and build CS role -> icon index."""
    with open(ontology_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ontology = {}
    cs_role_index = defaultdict(list)  # cs_role -> [icon_info, ...]

    for item in data['ontology']:
        icon_id = item.get('icon_id', '')
        if not icon_id:
            continue
        ontology[icon_id] = item
        cs_role = item.get('cs_role', 'WHAT')
        cs_role_index[cs_role].append(item)

    print(f"Loaded {len(ontology)} icons from ontology")
    for role in CS_ROLES:
        print(f"  {role}: {len(cs_role_index[role])} icons")

    return ontology, cs_role_index


def load_training_data(training_data_path: str) -> List[Dict]:
    """Load AAC2Text training data."""
    with open(training_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} training entries")
    return data


def parse_type_to_cs_roles(pattern_type: str) -> List[str]:
    """Convert training data type (e.g., 'svo', 'sv_time') to CS role sequence."""
    # Map from pattern type to CS roles
    type_to_cs = {
        'sv':        [CS_WHO, CS_WHAT_DOING],
        'svo':       [CS_WHO, CS_WHAT_DOING, CS_WHAT],
        'sv_time':   [CS_WHO, CS_WHAT_DOING, CS_WHEN],
        'svo_time':  [CS_WHO, CS_WHAT_DOING, CS_WHAT, CS_WHEN],
        'sv_emo':    [CS_WHO, CS_WHAT_DOING, CS_HOW],
        'svo_emo':   [CS_WHO, CS_WHAT_DOING, CS_WHAT, CS_HOW],
        'svo_place': [CS_WHO, CS_WHAT_DOING, CS_WHAT, CS_WHERE],
    }
    return type_to_cs.get(pattern_type, [])


def extract_emotion_from_labels(labels: List[str], pattern_type: str) -> str:
    """Extract emotion from labels based on pattern type."""
    if '_emo' not in pattern_type:
        return 'neutral'

    # For _emo patterns, the last label or emotion-type label is the emotion
    # This is a heuristic; actual emotion labels in data often contain emotion words
    emotion_keywords = {
        'happy': 'happiness', 'sad': 'sadness', 'angry': 'anger',
        'afraid': 'fear', 'scared': 'fear', 'disgust': 'disgust',
        'surprise': 'surprise', 'excited': 'happiness',
        'man': None, 'woman': None, 'boy': None, 'girl': None,
    }

    for label in reversed(labels):
        label_lower = label.lower()
        for kw, emotion in emotion_keywords.items():
            if kw in label_lower and emotion is not None:
                return emotion

    return 'neutral'


def generate_sequences_from_training_data(
    training_data: List[Dict],
    ontology: Dict
) -> List[Dict]:
    """Convert training_data.json entries into CS-annotated icon sequences."""
    sequences = []
    skipped = 0

    for entry in training_data:
        labels = entry.get('labels', [])
        pattern_type = entry.get('type', '')
        sentence = entry.get('sentence', '')

        if not labels or not pattern_type:
            skipped += 1
            continue

        # Get CS role sequence for this pattern
        cs_roles = parse_type_to_cs_roles(pattern_type)
        if not cs_roles:
            skipped += 1
            continue

        # If labels length matches CS roles, direct mapping
        if len(labels) == len(cs_roles):
            seq_cs_roles = cs_roles[:]
        else:
            # Try to infer CS roles from ontology for each label
            seq_cs_roles = []
            for label in labels:
                icon_info = ontology.get(label, {})
                cs_role = icon_info.get('cs_role', 'WHAT')
                seq_cs_roles.append(cs_role)

        # Get emotion
        emotion = extract_emotion_from_labels(labels, pattern_type)

        sequences.append({
            'sequence': labels[:],
            'cs_roles': seq_cs_roles,
            'emotion': emotion,
            'session_id': f'td_{len(sequences):05d}',
            'type': pattern_type,
            'source': 'training_data',
            'sentence': sentence,
        })

    print(f"Generated {len(sequences)} sequences from training data (skipped {skipped})")
    return sequences


def generate_template_sequences(
    cs_role_index: Dict[str, List],
    num_sequences: int = 50000,
    seed: int = 42
) -> List[Dict]:
    """Generate sequences by sampling from CS templates and ontology."""
    rng = random.Random(seed)
    sequences = []

    template_names = list(CS_TEMPLATES.keys())
    template_weights = [5, 10, 4, 6, 3, 8, 5, 2, 2, 1, 1, 2]  # svo/svo_emo more common

    for i in range(num_sequences):
        # Choose template
        template_name = rng.choices(template_names, weights=template_weights, k=1)[0]
        cs_roles = CS_TEMPLATES[template_name][0]

        # Sample icons for each CS role
        sequence = []
        valid = True
        for role in cs_roles:
            candidates = cs_role_index.get(role, [])
            if not candidates:
                valid = False
                break
            icon_info = rng.choice(candidates)
            sequence.append(icon_info['icon_id'])

        if not valid:
            continue

        # Assign random emotion with neutral being most common
        emotion = rng.choices(EMOTIONS, weights=[30, 5, 3, 3, 15, 8, 3], k=1)[0]

        sequences.append({
            'sequence': sequence,
            'cs_roles': cs_roles,
            'emotion': emotion,
            'session_id': f'syn_{i:05d}',
            'type': template_name,
            'source': 'template_augmentation',
            'sentence': '',
        })

    print(f"Generated {len(sequences)} template-based sequences")
    return sequences


def generate_multi_turn_sessions(
    cs_role_index: Dict[str, List],
    num_sessions: int = 5000,
    turns_per_session: Tuple[int, int] = (2, 5),
    seed: int = 42
) -> List[Dict]:
    """Generate multi-turn dialogue sessions for SASRec context training."""
    rng = random.Random(seed + 1)
    sessions = []

    template_names = list(CS_TEMPLATES.keys())

    for i in range(num_sessions):
        num_turns = rng.randint(*turns_per_session)
        session_sequence = []
        session_cs_roles = []
        session_emotions = []

        for t in range(num_turns):
            template_name = rng.choice(template_names)
            cs_roles = CS_TEMPLATES[template_name][0]

            turn_sequence = []
            valid = True
            for role in cs_roles:
                candidates = cs_role_index.get(role, [])
                if not candidates:
                    valid = False
                    break
                icon_info = rng.choice(candidates)
                turn_sequence.append(icon_info['icon_id'])

            if not valid:
                continue

            emotion = rng.choices(EMOTIONS, weights=[30, 5, 3, 3, 15, 8, 3], k=1)[0]
            session_sequence.extend(turn_sequence)
            session_cs_roles.extend(cs_roles)
            session_emotions.append(emotion)

        if len(session_sequence) < 2:
            continue

        # Most recent emotion for the session
        dominant_emotion = session_emotions[-1] if session_emotions else 'neutral'

        sessions.append({
            'sequence': session_sequence,
            'cs_roles': session_cs_roles,
            'emotion': dominant_emotion,
            'session_id': f'mt_{i:05d}',
            'type': 'multi_turn',
            'source': 'multi_turn_augmentation',
            'sentence': '',
            'num_turns': num_turns,
        })

    print(f"Generated {len(sessions)} multi-turn sessions")
    return sessions


def add_negative_samples(
    sequences: List[Dict],
    cs_role_index: Dict[str, List],
    num_neg_per_seq: int = 1,
    seed: int = 42
) -> List[Dict]:
    """Add negative samples by replacing the last icon with same-CS-role alternatives."""
    rng = random.Random(seed + 2)
    neg_sequences = []

    for seq in sequences:
        if len(seq['sequence']) < 2:
            continue

        for _ in range(num_neg_per_seq):
            neg_seq = seq.copy()
            neg_seq['sequence'] = seq['sequence'][:]
            neg_seq['cs_roles'] = seq['cs_roles'][:]

            # Replace the last icon with another of the same CS role
            last_role = seq['cs_roles'][-1]
            candidates = cs_role_index.get(last_role, [])

            if len(candidates) < 2:
                continue

            # Pick a different icon
            current_icon = seq['sequence'][-1]
            alternatives = [c for c in candidates if c['icon_id'] != current_icon]
            if not alternatives:
                continue

            replacement = rng.choice(alternatives)
            neg_seq['sequence'][-1] = replacement['icon_id']
            neg_seq['session_id'] = seq['session_id'] + '_neg'
            neg_seq['is_negative'] = True

            neg_sequences.append(neg_seq)

    print(f"Generated {len(neg_sequences)} negative samples")
    return neg_sequences


def split_data(sequences: List[Dict], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split sequences into train/val/test."""
    rng = random.Random(seed)
    rng.shuffle(sequences)

    n = len(sequences)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = sequences[:train_end]
    val = sequences[train_end:val_end]
    test = sequences[val_end:]

    return train, val, test


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic icon sequences')
    parser.add_argument('--ontology', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/AAC2Text/data/processed/aac_full_ontology.json')
    parser.add_argument('--training-data', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/AAC2Text/data/processed/training_data.json')
    parser.add_argument('--output-dir', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/data')
    parser.add_argument('--num-template', type=int, default=50000)
    parser.add_argument('--num-multi-turn', type=int, default=5000)
    parser.add_argument('--num-neg-per-seq', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Load data
    ontology, cs_role_index = load_ontology(args.ontology)
    training_data = load_training_data(args.training_data)

    # Generate all sequences
    print("\n=== Generating sequences ===")
    all_sequences = []

    # 1. From training data
    td_sequences = generate_sequences_from_training_data(training_data, ontology)
    all_sequences.extend(td_sequences)

    # 2. Template-based augmentation
    template_sequences = generate_template_sequences(cs_role_index, args.num_template, args.seed)
    all_sequences.extend(template_sequences)

    # 3. Multi-turn sessions
    mt_sequences = generate_multi_turn_sessions(cs_role_index, args.num_multi_turn, seed=args.seed)
    all_sequences.extend(mt_sequences)

    # 4. Negative samples
    neg_sequences = add_negative_samples(
        all_sequences, cs_role_index, args.num_neg_per_seq, args.seed
    )

    # Split positive sequences
    train, val, test = split_data(all_sequences, seed=args.seed)

    # Add negative samples to train only
    train.extend(neg_sequences)
    random.Random(args.seed).shuffle(train)

    # Print stats
    print(f"\n=== Final Dataset ===")
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")

    # Stats
    type_counter = Counter(s['type'] for s in train)
    print(f"\nTrain type distribution:")
    for t, c in type_counter.most_common():
        print(f"  {t}: {c}")

    emotion_counter = Counter(s['emotion'] for s in train)
    print(f"\nTrain emotion distribution:")
    for e, c in emotion_counter.most_common():
        print(f"  {e}: {c}")

    seq_len_counter = Counter(len(s['sequence']) for s in train)
    print(f"\nTrain sequence length distribution:")
    for l, c in sorted(seq_len_counter.items()):
        print(f"  len={l}: {c}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    for name, data in [('train', train), ('val', val), ('test', test)]:
        path = os.path.join(args.output_dir, f'icon_sequences_{name}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} sequences to {path}")


if __name__ == '__main__':
    main()
