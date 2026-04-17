from ctypes import alignment
import json
import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle
import numpy as np
import load_data

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch
import sqlglot
from sqlglot import exp
from collections import defaultdict

def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):

    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x

def compute_t5_stats(nl_data, sql_data, tokenizer, processing = False):
    assert len(nl_data) == len(sql_data)

    num_examples = len(nl_data)

    nl_lengths = []
    sql_lengths = []

    nl_token_ids = Counter()
    sql_token_ids = Counter()

    for nl, sql in zip(nl_data, sql_data):
        # encode (no padding, no truncation)
        if processing:
            nl = "Translate English to SQL: " + nl
        nl_ids = tokenizer.encode(nl, add_special_tokens=False)
        sql_ids = tokenizer.encode(sql, add_special_tokens=False)

        # lengths
        nl_lengths.append(len(nl_ids))
        sql_lengths.append(len(sql_ids))

        # vocab (track token IDs)
        nl_token_ids.update(nl_ids)
        sql_token_ids.update(sql_ids)

    stats = {
        "num_examples": num_examples,
        "mean_nl_length": np.mean(nl_lengths),
        "mean_sql_length": np.mean(sql_lengths),
        "nl_vocab_size": len(nl_token_ids),
        "sql_vocab_size": len(sql_token_ids),
    }

    return stats



import sqlglot
from sqlglot import exp
from collections import defaultdict


def analyze_sql_file(sql_lines):
    errors = defaultdict(int)


    queries = [q.strip() for q in sql_lines]

    for query in queries:
        duplicate_condition = False
        try:
            tree = sqlglot.parse_one(query)
        except Exception:
            errors["parse_failure"] += 1
            continue

        # -------------------------
        # WHERE logic checks
        # -------------------------
        where = tree.find(exp.Where)

        if where:
            conditions = list(where.find_all(exp.Condition))
            seen = set()

            for c in conditions:
                sql = c.sql()

                # duplicate condition
                if sql in seen:
                    errors["duplicate_condition"] += 1
                    print(f"Duplicate condition found: {sql} in query: {query}") 
                    duplicate_condition = True
                    
                seen.add(sql)

                # always-true patterns
                if sql.strip() in ("1 = 1", "TRUE"):
                    errors["always_true_condition"] += 1

                # self comparison (x = x)
                if isinstance(c, exp.EQ):
                    if c.left and c.right and c.left.sql() == c.right.sql():
                        errors["self_comparison"] += 1

        # -------------------------
        # NULL misuse
        # -------------------------
        for node in tree.find_all(exp.EQ):
            if node.right and "NULL" in node.right.sql().upper():
                errors["null_equals_used"] += 1

        # -------------------------
        # Contradictory ranges (simple heuristic)
        # -------------------------
        gt = {}
        lt = {}

        for node in tree.find_all(exp.GT):
            if isinstance(node.right, exp.Literal):
                gt[node.left.sql()] = float(node.right.name)

        for node in tree.find_all(exp.LT):
            if isinstance(node.right, exp.Literal):
                lt[node.left.sql()] = float(node.right.name)

        for col in gt:
            if col in lt and gt[col] > lt[col]:
                errors["contradictory_range"] += 1

        # -------------------------
        # Cartesian join heuristic
        # -------------------------
        tables = {t.name for t in tree.find_all(exp.Table)}

        if len(tables) > 2:
            has_join = any(isinstance(j, exp.Join) for j in tree.find_all(exp.Join))
            if not has_join:
                errors["possible_cartesian_join"] += 1

        # -------------------------
        # DISTINCT overuse
        # -------------------------
        if isinstance(tree, exp.Select):
            if tree.args.get("distinct"):
                errors["distinct_used"] += 1

        # -------------------------
        # Unused table heuristic
        # -------------------------
        columns = {c.name for c in tree.find_all(exp.Column)}

        for t in tables:
            if not any(t in col for col in columns):
                errors["possible_unused_table"] += 1
    
        if duplicate_condition:
            errors["queries_with_duplicate_conditions"] += 1

    return dict(errors)



if __name__ == "__main__":

    results_folder = 'results'

    sql_lines = load_lines(os.path.join(results_folder, 't5_ft_experiment_dev.sql'))
    errors = analyze_sql_file(sql_lines)

    print("SQL Error Summary:\n")
    for k, v in errors.items():
        print(f"{k}: {v}")
    '''data_folder = 'data'
    train_x, train_y, dev_x, dev_y, test_x = load_prompting_data(data_folder)

    tokenizer = T5TokenizerFast.from_pretrained("t5-small")

    train_stats = compute_t5_stats(train_x, train_y, tokenizer, processing = True)
    dev_stats = compute_t5_stats(dev_x, dev_y, tokenizer, processing = True)

    print("Train stats:", train_stats)
    print("Dev stats:", dev_stats)'''