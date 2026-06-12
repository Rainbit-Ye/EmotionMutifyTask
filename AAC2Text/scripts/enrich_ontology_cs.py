#!/usr/bin/env python3
"""
Enrich AAC ontology with Colourful Semantics (CS) roles and normalize grammar_role.

CS Roles (Colourful Semantics):
  WHO        - Agent/Subject (I, you, mum, doctor)
  WHAT_DOING - Action/Predicate (eat, drink, go, help)
  WHAT       - Object/Patient (water, food, apple)
  WHERE      - Location (home, school, hospital)
  WHEN       - Time (morning, today, week)
  HOW        - Manner/Modifier (quickly, happy, sad)

Strategy:
  1. Normalize grammar_role (merge inconsistent values)
  2. Map (grammar_role, semantic_type) -> cs_role using deterministic rules
  3. For ambiguous/unmapped icons, use Qwen2.5-1.5B to infer CS role
"""

import json
import os
import sys
from collections import Counter
from typing import Dict, Optional

# ==================== grammar_role normalization ====================

GRAMMAR_ROLE_NORMALIZE_MAP = {
    # OBJ variants
    'obj': 'OBJ', 'object': 'OBJ', 'OBJ': 'OBJ',
    # SUBJ variants
    'subj': 'SUBJ', 'subject': 'SUBJ', 'SUBJ': 'SUBJ',
    # TRANS variants
    'trans': 'TRANS', 'verb': 'TRANS', 'VERB': 'TRANS',
    'transitive verb': 'TRANS', 'predicate': 'TRANS', 'ACTION': 'TRANS',
    # INTR variants
    'intransitive verb': 'INTR',
    # MOD variants
    'mod': 'MOD', 'modifier': 'MOD', 'MODIFIER': 'MOD', 'MOD': 'MOD',
    # LOC variants
    'loc': 'LOC', 'location': 'LOC', 'LOC': 'LOC',
    # COMPL
    'complement': 'COMPL', 'COMPL': 'COMPL',
    # DUR
    'DUR': 'DUR',
    # INST
    'instrument': 'INST',
    # DIR
    'DIR(direction)': 'DIR',
    # Special cases
    'subj_obj': 'SUBJ', 'SUBJ OBJ': 'SUBJ', 'SUBJ/OBJ': 'SUBJ',
    'container': 'OBJ', 'accessory': 'OBJ', 'symbol': 'HOW',
    'subordinate': 'COMPL',
    # Empty/none
    '': '', 'none': '',
}

# Canonical grammar_role values
CANONICAL_ROLES = {'', 'SUBJ', 'OBJ', 'TRANS', 'INTR', 'MOD', 'LOC', 'COMPL', 'DUR', 'INST', 'DIR'}


def normalize_grammar_role(role: str) -> str:
    """Normalize grammar_role to canonical form."""
    return GRAMMAR_ROLE_NORMALIZE_MAP.get(role, role)


# ==================== CS Role mapping ====================

# CS role constants
CS_WHO = 'WHO'
CS_WHAT_DOING = 'WHAT_DOING'
CS_WHAT = 'WHAT'
CS_WHERE = 'WHERE'
CS_WHEN = 'WHEN'
CS_HOW = 'HOW'

CS_ROLES = [CS_WHO, CS_WHAT_DOING, CS_WHAT, CS_WHERE, CS_WHEN, CS_HOW]

# Deterministic mapping: (normalized_grammar_role, semantic_type) -> cs_role
# Order matters: more specific rules first
CS_ROLE_RULES = [
    # === WHO: Subjects/Agents ===
    ('SUBJ', 'person', CS_WHO),
    ('SUBJ', 'entity', CS_WHO),
    ('SUBJ', 'object', CS_WHO),
    ('SUBJ', 'body', CS_WHO),
    ('SUBJ', 'body part', CS_WHO),
    ('SUBJ', 'body_part', CS_WHO),
    ('SUBJ', 'animal', CS_WHO),
    ('SUBJ', 'food', CS_WHO),       # "I eat food" - food as subject is rare but possible
    ('SUBJ', 'tool', CS_WHO),
    ('SUBJ', 'device', CS_WHO),
    ('SUBJ', 'clothing', CS_WHO),

    # === WHAT_DOING: Actions/Verbs ===
    ('TRANS', 'action', CS_WHAT_DOING),
    ('TRANS', '', CS_WHAT_DOING),
    ('INTR', 'action', CS_WHAT_DOING),
    ('SUBJ', 'action', CS_WHAT_DOING),     # some verbs tagged as SUBJ (mislabel)
    ('OBJ', 'action', CS_WHAT_DOING),      # some verbs tagged as OBJ (mislabel)

    # === WHAT: Objects/Patients ===
    ('OBJ', 'object', CS_WHAT),
    ('OBJ', 'entity', CS_WHAT),
    ('OBJ', 'food', CS_WHAT),
    ('OBJ', 'drink', CS_WHAT),
    ('OBJ', 'body', CS_WHAT),
    ('OBJ', 'body part', CS_WHAT),
    ('OBJ', 'body_part', CS_WHAT),
    ('OBJ', 'tool', CS_WHAT),
    ('OBJ', 'device', CS_WHAT),
    ('OBJ', 'quantity', CS_WHAT),
    ('OBJ', 'animal', CS_WHAT),
    ('OBJ', 'clothing', CS_WHAT),
    ('OBJ', 'quality', CS_WHAT),
    ('OBJ', 'substance', CS_WHAT),
    ('OBJ', 'material', CS_WHAT),
    ('OBJ', 'electronics', CS_WHAT),
    ('OBJ', 'medicine', CS_WHAT),
    ('OBJ', 'art', CS_WHAT),
    ('OBJ', 'place', CS_WHAT),
    ('OBJ', '', CS_WHAT),                 # OBJ with no type -> likely WHAT

    # === WHERE: Locations ===
    ('LOC', 'place', CS_WHERE),
    ('LOC', 'location', CS_WHERE),
    ('LOC', '', CS_WHERE),
    ('SUBJ', 'place', CS_WHERE),          # place as subject
    ('SUBJ', 'location', CS_WHERE),

    # === WHEN: Time ===
    ('SUBJ', 'time', CS_WHEN),
    ('DUR', '', CS_WHEN),
    ('DUR', 'time', CS_WHEN),

    # === HOW: Modifiers ===
    ('MOD', 'quality', CS_HOW),
    ('MOD', 'emotion', CS_HOW),
    ('MOD', 'adjective', CS_HOW),
    ('MOD', 'adverb', CS_HOW),
    ('MOD', 'adv', CS_HOW),
    ('MOD', 'modifier', CS_HOW),
    ('MOD', '', CS_HOW),
    ('SUBJ', 'emotion', CS_HOW),          # emotions as subject -> modifier-like
    ('SUBJ', 'quality', CS_HOW),
    ('OBJ', 'emotion', CS_HOW),
    ('OBJ', 'quality', CS_HOW),

    # === COMPL: Complements ===
    ('COMPL', '', CS_WHAT),               # complements usually act as WHAT
    ('COMPL', 'action', CS_WHAT_DOING),
    ('COMPL', 'place', CS_WHERE),
    ('COMPL', 'time', CS_WHEN),

    # === INST: Instruments ===
    ('INST', '', CS_WHAT),                # instruments are usually WHAT
    ('INST', 'tool', CS_WHAT),
    ('INST', 'object', CS_WHAT),

    # === DIR: Directions ===
    ('DIR', '', CS_WHERE),                # directions map to WHERE
]

# Fallback: semantic_type only (for empty grammar_role)
SEMANTIC_TYPE_CS_MAP = {
    'action': CS_WHAT_DOING,
    'verb': CS_WHAT_DOING,
    'person': CS_WHO,
    'food': CS_WHAT,
    'drink': CS_WHAT,
    'place': CS_WHERE,
    'location': CS_WHERE,
    'time': CS_WHEN,
    'emotion': CS_HOW,
    'quality': CS_HOW,
    'adjective': CS_HOW,
    'adverb': CS_HOW,
    'adv': CS_HOW,
    'object': CS_WHAT,
    'entity': CS_WHO,        # entities without grammar_role -> assume WHO (common for SUBJ)
    'body': CS_WHO,
    'body part': CS_WHO,
    'body_part': CS_WHO,
    'animal': CS_WHAT,
    'tool': CS_WHAT,
    'device': CS_WHAT,
    'quantity': CS_WHAT,
    'symbol': CS_HOW,
    'activity': CS_WHAT_DOING,
    'event': CS_WHAT_DOING,
    'noun': CS_WHAT,
    'number': CS_WHAT,
    'numeral': CS_WHAT,
    'clothing': CS_WHAT,
    'medicine': CS_WHAT,
    'substance': CS_WHAT,
    'material': CS_WHAT,
    'electronics': CS_WHAT,
    'art': CS_WHAT,
    'shape': CS_HOW,
    'modifier': CS_HOW,
    'moderator': CS_HOW,
}


def infer_cs_role(grammar_role: str, semantic_type: str, label: str = '', core_semantic: str = '') -> str:
    """Infer CS role from grammar_role and semantic_type using deterministic rules."""
    # Normalize grammar_role first
    norm_role = normalize_grammar_role(grammar_role)

    # Try exact (grammar_role, semantic_type) match
    for rule_role, rule_type, cs_role in CS_ROLE_RULES:
        if rule_role == norm_role and rule_type == semantic_type:
            return cs_role

    # Try grammar_role only match (semantic_type='')
    for rule_role, rule_type, cs_role in CS_ROLE_RULES:
        if rule_role == norm_role and rule_type == '':
            return cs_role

    # Fallback: semantic_type only (for empty/unknown grammar_role)
    if semantic_type in SEMANTIC_TYPE_CS_MAP:
        return SEMANTIC_TYPE_CS_MAP[semantic_type]

    # Last resort: use grammar_role heuristics
    if norm_role == 'SUBJ':
        return CS_WHO
    elif norm_role in ('TRANS', 'INTR'):
        return CS_WHAT_DOING
    elif norm_role == 'OBJ':
        return CS_WHAT
    elif norm_role == 'LOC':
        return CS_WHERE
    elif norm_role == 'MOD':
        return CS_HOW

    return CS_WHAT  # default


def infer_cs_role_with_llm(icon_info: dict, model, tokenizer) -> str:
    """Use LLM to infer CS role for ambiguous icons."""
    label = icon_info.get('label', '')
    core_semantic = icon_info.get('core_semantic', '')
    semantic_type = icon_info.get('semantic_type', '')
    grammar_role = icon_info.get('grammar_role', '')

    prompt = f"""Classify this AAC pictographic symbol into exactly one Colourful Semantics role:

- WHO: Agent/Subject who does an action (e.g., I, you, mum, doctor, teacher)
- WHAT_DOING: The action/predicate (e.g., eat, drink, go, help, want)
- WHAT: Object/patient that receives the action (e.g., water, food, apple, book)
- WHERE: Location/place (e.g., home, school, hospital, kitchen)
- WHEN: Time expression (e.g., morning, today, week, now)
- HOW: Manner/modifier/emotion (e.g., quickly, happy, sad, big, red)

Symbol: {label}
Meaning: {core_semantic}
Category: {semantic_type}
Grammar role: {grammar_role}

Answer with ONLY one word: WHO, WHAT_DOING, WHAT, WHERE, WHEN, or HOW"""

    import torch
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=10, do_sample=False,
            stop_strings=["\n", "."], tokenizer=tokenizer,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip().upper()

    # Parse response
    for role in CS_ROLES:
        if role in response:
            return role

    return CS_WHAT  # default


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Enrich ontology with CS roles')
    parser.add_argument('--input', type=str,
                        default='/home/user1/liuduanye/EmotionClassify/AAC2Text/data/processed/aac_full_ontology.json',
                        help='Input ontology path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ontology path (default: overwrite input)')
    parser.add_argument('--use-llm', action='store_true',
                        help='Use LLM for ambiguous icons (requires GPU)')
    parser.add_argument('--llm-model', type=str,
                        default='/home/user1/liuduanye/qwen/Qwen2_5-1_5B-Instruct',
                        help='LLM model path for ambiguous cases')
    args = parser.parse_args()

    output_path = args.output or args.input

    # Load ontology
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ontology = data['ontology']
    print(f"Loaded {len(ontology)} icons")

    # Step 1: Normalize grammar_role and infer CS roles
    cs_role_counter = Counter()
    grammar_role_changes = Counter()
    ambiguous_icons = []

    for item in ontology:
        # Normalize grammar_role
        old_role = item.get('grammar_role', '')
        new_role = normalize_grammar_role(old_role)
        if old_role != new_role:
            grammar_role_changes[(old_role, new_role)] += 1
        item['grammar_role'] = new_role

        # Infer CS role
        semantic_type = item.get('semantic_type', '')
        label = item.get('label', '')
        core_semantic = item.get('core_semantic', '')

        cs_role = infer_cs_role(new_role, semantic_type, label, core_semantic)
        item['cs_role'] = cs_role
        cs_role_counter[cs_role] += 1

        # Track ambiguous cases (empty grammar_role + uncommon semantic_type)
        if not new_role and semantic_type not in SEMANTIC_TYPE_CS_MAP:
            ambiguous_icons.append(item)

    # Step 2: Use LLM for truly ambiguous icons (if enabled)
    if args.use_llm and ambiguous_icons:
        print(f"\nLoading LLM for {len(ambiguous_icons)} ambiguous icons...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.llm_model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model, torch_dtype=torch.float16,
            device_map='auto', local_files_only=True
        )
        model.eval()

        for i, item in enumerate(ambiguous_icons):
            cs_role = infer_cs_role_with_llm(item, model, tokenizer)
            old_cs = item['cs_role']
            item['cs_role'] = cs_role
            cs_role_counter[old_cs] -= 1
            cs_role_counter[cs_role] += 1
            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(ambiguous_icons)} ambiguous icons")

    # Step 3: Update metadata
    data['metadata']['cs_roles'] = CS_ROLES
    data['metadata']['canonical_grammar_roles'] = sorted(CANONICAL_ROLES - {''})

    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Print stats
    print(f"\n=== CS Role Distribution ===")
    for role in CS_ROLES:
        print(f"  {role}: {cs_role_counter.get(role, 0)}")

    print(f"\n=== Grammar Role Normalization ===")
    total_changes = sum(grammar_role_changes.values())
    print(f"  Total normalized: {total_changes}")
    for (old, new), count in grammar_role_changes.most_common(10):
        print(f"  {repr(old)} -> {repr(new)}: {count}")

    print(f"\nAmbiguous icons (no grammar_role + uncommon semantic_type): {len(ambiguous_icons)}")
    print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    main()
