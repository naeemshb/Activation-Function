# ============================================================================
# IMPORTS
# ============================================================================

import math
import random
import copy
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from activation_viz import plot_generational_shapes, plot_run_winners
from metrics_report import compute_extra_metrics, print_aggregated_results


# ============================================================================
# TREE STRUCTURE COMPONENTS
# ============================================================================

INPUTS = ["value", "missing", "confidence"]
CONSTANTS = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1]


def create_input_node(input_name: str) -> dict:
    assert input_name in INPUTS, f"Invalid input name: {input_name}"
    return {"type": "input", "input_name": input_name, "children": []}


def create_constant_node(value: float) -> dict:
    return {"type": "constant", "value": float(value), "children": []}


def create_unary_node(operator: str, child: dict) -> dict:
    return {"type": "unary", "operator": operator, "children": [child]}


def create_binary_node(operator: str, left_child: dict, right_child: dict) -> dict:
    return {"type": "binary", "operator": operator, "children": [left_child, right_child]}


def safe_divide(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    denom = torch.where(
        torch.abs(y) < 1e-3,
        torch.sign(y) * 1e-3 + (y == 0).float() * 1e-3,
        y,
    )
    return x / denom


def safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(x, min=0.0))


def safe_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp(torch.abs(x), min=1e-8))


def safe_pow(x: torch.Tensor, power: float) -> torch.Tensor:
    return torch.pow(torch.clamp(torch.abs(x), min=1e-8, max=1000.0), power)


def clip_result(x: torch.Tensor, min_val: float = -1000.0, max_val: float = 1000.0) -> torch.Tensor:
    return torch.clamp(x, min=min_val, max=max_val)


UNARY_OPERATORS = {
    "identity": lambda x: x,
    "negate": lambda x: -x,
    "abs": lambda x: torch.abs(x),
    "square": lambda x: clip_result(x * x),
    "cube": lambda x: clip_result(x * x * x),
    "sqrt": lambda x: safe_sqrt(x),
    "exp": lambda x: clip_result(torch.exp(torch.clamp(x, min=-10, max=10))),
    "log": lambda x: safe_log(x),
    "sin": lambda x: torch.sin(x),
    "cos": lambda x: torch.cos(x),
    "tanh": lambda x: torch.tanh(x),
    "sigmoid": lambda x: torch.sigmoid(x),
    "relu": lambda x: torch.clamp(x, min=0),
    "softplus": lambda x: torch.nn.functional.softplus(x),
}

BINARY_OPERATORS = {
    "add": lambda x, y: x + y,
    "subtract": lambda x, y: x - y,
    "multiply": lambda x, y: clip_result(x * y),
    "divide": lambda x, y: safe_divide(x, y),
    "max": lambda x, y: torch.maximum(x, y),
    "min": lambda x, y: torch.minimum(x, y),
}


def evaluate_tree(node: dict, value: torch.Tensor, missing: torch.Tensor, 
                  confidence: torch.Tensor) -> torch.Tensor:
    ntype = node["type"]

    if ntype == "input":
        mapping = {"value": value, "missing": missing, "confidence": confidence}
        return mapping[node["input_name"]]

    if ntype == "constant":
        return torch.full_like(value, node["value"])

    if ntype == "unary":
        child = evaluate_tree(node["children"][0], value, missing, confidence)
        op = UNARY_OPERATORS[node["operator"]]
        return clip_result(op(child))

    if ntype == "binary":
        left = evaluate_tree(node["children"][0], value, missing, confidence)
        right = evaluate_tree(node["children"][1], value, missing, confidence)
        op = BINARY_OPERATORS[node["operator"]]
        return clip_result(op(left, right))

    raise ValueError(f"Unknown node type: {ntype}")


def tree_to_string(node: dict, indent: int = 0) -> str:
    spacing = "  " * indent
    t = node["type"]

    if t == "input":
        return f"{spacing}{node['input_name']}"
    if t == "constant":
        return f"{spacing}{node['value']}"

    if t == "unary":
        child_str = tree_to_string(node["children"][0], indent + 1)
        return f"{spacing}{node['operator']}(\n{child_str}\n{spacing})"

    if t == "binary":
        left_str = tree_to_string(node["children"][0], indent + 1)
        right_str = tree_to_string(node["children"][1], indent + 1)
        return f"{spacing}{node['operator']}(\n{left_str},\n{right_str}\n{spacing})"

    return f"{spacing}<?>"


def tree_to_infix(node: dict) -> str:
    t = node["type"]

    if t == "input":
        return {"value": "x", "missing": "m", "confidence": "c"}[node["input_name"]]

    if t == "constant":
        v = node["value"]
        return str(int(v)) if float(v).is_integer() else f"{v}"

    if t == "unary":
        op = node["operator"]
        s = tree_to_infix(node["children"][0])
        mapping = {
            "identity": s,
            "negate": f"(-{s})",
            "abs": f"|{s}|",
            "square": f"({s})^2",
            "cube": f"({s})^3",
            "sqrt": f"sqrt({s})",
            "exp": f"exp({s})",
            "log": f"log(|{s}|)",
            "sin": f"sin({s})",
            "cos": f"cos({s})",
            "tanh": f"tanh({s})",
            "sigmoid": f"sigmoid({s})",
            "relu": f"relu({s})",
            "softplus": f"softplus({s})",
        }
        return mapping.get(op, f"{op}({s})")

    if t == "binary":
        op = node["operator"]
        a = tree_to_infix(node["children"][0])
        b = tree_to_infix(node["children"][1])
        mapping = {
            "add": f"({a} + {b})",
            "subtract": f"({a} - {b})",
            "multiply": f"({a} * {b})",
            "divide": f"({a} / {b})",
            "max": f"max({a}, {b})",
            "min": f"min({a}, {b})",
        }
        return mapping.get(op, f"{op}({a}, {b})")

    return "<?>"


def get_tree_depth(node: dict) -> int:
    t = node["type"]
    if t in ["input", "constant"]:
        return 1
    if t == "unary":
        return 1 + get_tree_depth(node["children"][0])
    if t == "binary":
        return 1 + max(get_tree_depth(node["children"][0]),
                       get_tree_depth(node["children"][1]))
    raise ValueError(f"Invalid node type: {t}")


def count_nodes(node: dict) -> int:
    t = node["type"]
    if t in ["input", "constant"]:
        return 1
    if t == "unary":
        return 1 + count_nodes(node["children"][0])
    if t == "binary":
        return 1 + count_nodes(node["children"][0]) + count_nodes(node["children"][1])
    raise ValueError(f"Invalid node type: {t}")


def generate_terminal_node() -> dict:
    if random.random() < 0.7:
        input_name = random.choice(INPUTS)
        return create_input_node(input_name)
    constant_value = random.choice(CONSTANTS)
    return create_constant_node(constant_value)


def generate_non_terminal_node(max_depth: int, current_depth: int) -> dict:
    if random.random() < 0.4:
        op = random.choice(list(UNARY_OPERATORS.keys()))
        child = generate_random_tree(max_depth, current_depth + 1)
        return create_unary_node(op, child)
    op = random.choice(list(BINARY_OPERATORS.keys()))
    left = generate_random_tree(max_depth, current_depth + 1)
    right = generate_random_tree(max_depth, current_depth + 1)
    return create_binary_node(op, left, right)


def generate_random_tree(max_depth: int, current_depth: int = 0) -> dict:
    if current_depth >= max_depth:
        return generate_terminal_node()

    terminal_prob = 0.3 + 0.4 * (current_depth / max_depth)
    if random.random() < terminal_prob:
        return generate_terminal_node()

    return generate_non_terminal_node(max_depth, current_depth)


def validate_tree(node: dict) -> None:
    t = node["type"]

    if t == "input":
        assert node["input_name"] in INPUTS
        assert len(node["children"]) == 0
        return

    if t == "constant":
        assert isinstance(node["value"], (int, float))
        assert len(node["children"]) == 0
        return

    if t == "unary":
        assert node["operator"] in UNARY_OPERATORS
        assert len(node["children"]) == 1
        validate_tree(node["children"][0])
        return

    if t == "binary":
        assert node["operator"] in BINARY_OPERATORS
        assert len(node["children"]) == 2
        validate_tree(node["children"][0])
        validate_tree(node["children"][1])
        return

    raise ValueError(f"Invalid node type: {t}")


def get_all_nodes(tree: dict) -> list:
    nodes = [tree]
    if tree["type"] in ["unary", "binary"]:
        for ch in tree["children"]:
            nodes.extend(get_all_nodes(ch))
    return nodes


def mutate_single_node(node: dict, max_depth: int, current_tree_depth: int) -> None:
    t = node["type"]

    if t == "input":
        current_input = node["input_name"]
        candidates = [inp for inp in INPUTS if inp != current_input]
        if candidates:
            node["input_name"] = random.choice(candidates)

    elif t == "constant":
        node["value"] = random.choice(CONSTANTS)

    elif t == "unary":
        if random.random() < 0.7:
            current_op = node["operator"]
            candidates = [op for op in UNARY_OPERATORS.keys() if op != current_op]
            if candidates:
                node["operator"] = random.choice(candidates)
        else:
            if current_tree_depth < max_depth:
                remaining = max_depth - current_tree_depth + get_tree_depth(node)
                new_subtree = generate_random_tree(min(remaining, 3))
                node.clear()
                node.update(new_subtree)

    elif t == "binary":
        if random.random() < 0.7:
            current_op = node["operator"]
            candidates = [op for op in BINARY_OPERATORS.keys() if op != current_op]
            if candidates:
                node["operator"] = random.choice(candidates)
        else:
            if current_tree_depth < max_depth:
                remaining = max_depth - current_tree_depth + get_tree_depth(node)
                new_subtree = generate_random_tree(min(remaining, 3))
                node.clear()
                node.update(new_subtree)


def mutate_tree(tree: dict, mutation_rate: float = 0.1, max_depth: int = 5) -> dict:
    mutated = copy.deepcopy(tree)
    all_nodes = get_all_nodes(mutated)

    for node in all_nodes:
        if random.random() < mutation_rate:
            mutate_single_node(node, max_depth, get_tree_depth(mutated))

    validate_tree(mutated)
    return mutated


def perform_crossover(node1: dict, node2: dict) -> None:
    temp = copy.deepcopy(node1)
    node1.clear()
    node1.update(copy.deepcopy(node2))
    node2.clear()
    node2.update(temp)


def crossover_trees(parent1: dict, parent2: dict, max_depth: int = 5) -> tuple:
    offspring1 = copy.deepcopy(parent1)
    offspring2 = copy.deepcopy(parent2)

    nodes1 = get_all_nodes(offspring1)
    nodes2 = get_all_nodes(offspring2)

    crossover_node1 = random.choice(nodes1)
    crossover_node2 = random.choice(nodes2)
    perform_crossover(crossover_node1, crossover_node2)

    try:
        validate_tree(offspring1)
        validate_tree(offspring2)
    except Exception:
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    return offspring1, offspring2


def create_initial_population(population_size: int, max_depth: int = 4) -> list:
    population = []
    for _ in range(population_size):
        tree = generate_random_tree(max_depth)
        population.append({"tree": tree, "fitness": None})
    return population


def weighted_random_choice(population: list, probabilities: list) -> dict:
    r = random.random()
    cumulative = 0.0
    for ind, p in zip(population, probabilities):
        cumulative += p
        if r <= cumulative:
            return ind
    return population[-1]


def fitness_proportional_selection(population: list, num_selected: int) -> list:
    fitness_values = [ind["fitness"] for ind in population]
    min_f = min(fitness_values)
    if min_f < 0:
        shifted = [f - min_f + 1.0 for f in fitness_values]
    else:
        shifted = fitness_values

    max_f = max(shifted)
    exp_f = [math.exp(f - max_f) for f in shifted]
    total = sum(exp_f)
    probs = [e / total for e in exp_f]

    selected = []
    for _ in range(num_selected):
        selected.append(weighted_random_choice(population, probs))
    return selected


def evolve_population(population: list, mutation_rate: float = 0.1, 
                     crossover_rate: float = 0.7, elite_size: int = 2) -> list:
    population_size = len(population)
    sorted_pop = sorted(population, key=lambda x: x["fitness"], reverse=True)

    new_population = []

    for i in range(elite_size):
        new_population.append({"tree": copy.deepcopy(sorted_pop[i]["tree"]),
                               "fitness": None})

    while len(new_population) < population_size:
        if random.random() < crossover_rate and len(new_population) < population_size - 1:
            parent1 = fitness_proportional_selection(population, 1)[0]
            parent2 = fitness_proportional_selection(population, 1)[0]
            off1_tree, off2_tree = crossover_trees(parent1["tree"], parent2["tree"])
            new_population.append({"tree": off1_tree, "fitness": None})
            if len(new_population) < population_size:
                new_population.append({"tree": off2_tree, "fitness": None})
        else:
            parent = fitness_proportional_selection(population, 1)[0]
            mutated = mutate_tree(parent["tree"], mutation_rate)
            new_population.append({"tree": mutated, "fitness": None})

    return new_population


def count_input_diversity(node: dict) -> int:
    found = set()

    def walk(n):
        if n.get("type") == "input":
            found.add(n["input_name"])
        for ch in n.get("children", []):
            walk(ch)

    walk(node)
    return len(found)


# ============================================================================
# NEURAL NETWORK: THREE-CHANNEL MLP
# ============================================================================

class ChannelProp(nn.Module):
    def __init__(self, linear: nn.Linear, eps: float = 1e-8):
        super().__init__()
        self.linear = linear
        self.eps = eps

    def forward(self, x, m, c):
        y = self.linear(x)

        W = self.linear.weight
        W_abs = W.abs() + self.eps
        col_sum = W_abs.sum(dim=1, keepdim=True)
        W_norm = W_abs / col_sum
        W_norm_T = W_norm.t()

        c_out = c @ W_norm_T
        obs_in = 1.0 - m
        obs_out = obs_in @ W_norm_T
        m_out = 1.0 - obs_out

        c_out = c_out.clamp(0.0, 1.0)
        m_out = m_out.clamp(0.0, 1.0)
        return y, m_out, c_out


class AdaptiveLayer(nn.Module):
    def __init__(self, activation_tree: dict):
        super().__init__()
        self.activation_tree = activation_tree

    def forward(self, x, m, c):
        return evaluate_tree(self.activation_tree, x, m, c)


class ThreeChannelBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation_tree: dict):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.prop = ChannelProp(self.lin)
        self.act = AdaptiveLayer(activation_tree)

    def forward(self, x, m, c):
        x, m, c = self.prop(x, m, c)
        x = self.act(x, m, c)
        return x, m, c


class ThreeChannelMLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes, output_size: int, activation_tree: dict):
        super().__init__()
        sizes = [input_size] + list(hidden_sizes)
        self.blocks = nn.ModuleList([
            ThreeChannelBlock(sizes[i], sizes[i + 1], activation_tree)
            for i in range(len(hidden_sizes))
        ])
        self.out = nn.Linear(sizes[-1], output_size)

    def forward(self, values, missing_flags, confidence_scores):
        x, m, c = values, missing_flags, confidence_scores
        for block in self.blocks:
            x, m, c = block(x, m, c)
        return self.out(x)


def evaluate_fitness_neural_network(tree: dict, X_train: torch.Tensor, y_train: torch.Tensor,
                                    X_val: torch.Tensor, y_val: torch.Tensor,
                                    missing_train: torch.Tensor, missing_val: torch.Tensor,
                                    conf_train: torch.Tensor, conf_val: torch.Tensor,
                                    num_epochs: int = 50, device=None, hidden_sizes=None) -> float:
    if hidden_sizes is None:
        hidden_sizes = [32, 16]

    if get_tree_depth(tree) == 1:
        return 0.0

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        missing_train = missing_train.to(device)
        missing_val = missing_val.to(device)
        conf_train = conf_train.to(device)
        conf_val = conf_val.to(device)

        model = ThreeChannelMLP(
            input_size=X_train.shape[1],
            hidden_sizes=hidden_sizes,
            output_size=len(torch.unique(y_train)),
            activation_tree=tree,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        patience = 2
        patience_counter = 0

        for _ in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train, missing_train, conf_train)
            loss = criterion(outputs, y_train)

            if not torch.isfinite(loss):
                return 0.0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val, missing_val, conf_val)
                _, preds = torch.max(val_outputs, 1)
                val_acc = (preds == y_val).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        size_penalty = 0.0001 * count_nodes(tree)
        depth_penalty = 0.0002 * (get_tree_depth(tree) - 1)
        diversity = count_input_diversity(tree) / 3.0

        fitness = best_val_acc + 0.01 * diversity - size_penalty - depth_penalty

        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return fitness

    except Exception as e:
        print(f"Error evaluating tree: {e}")
        return 0.0


def run_evolution_with_nn(X_train, y_train, X_val, y_val, missing_train, missing_val,
                         conf_train, conf_val, population_size, generations, max_depth,
                         mutation_rate, crossover_rate, elite_size, hidden_sizes, 
                         num_epochs_fitness):
    population = create_initial_population(population_size, max_depth)

    best_ever = None
    best_ever_fitness = -float("inf")
    gen_best_log = []

    for gen in range(generations):
        for idx, individual in enumerate(population):
            if individual["fitness"] is None:
                fitness = evaluate_fitness_neural_network(
                    individual["tree"],
                    X_train, y_train,
                    X_val, y_val,
                    missing_train, missing_val,
                    conf_train, conf_val,
                    num_epochs=num_epochs_fitness,
                    hidden_sizes=hidden_sizes,
                )
                individual["fitness"] = fitness

        population.sort(key=lambda x: x["fitness"], reverse=True)
        best_ind = population[0]

        gen_best_log.append({
            "generation": gen + 1,
            "fitness": float(best_ind["fitness"]),
            "nodes": int(count_nodes(best_ind["tree"])),
            "depth": int(get_tree_depth(best_ind["tree"])),
            "uses_inputs": int(count_input_diversity(best_ind["tree"])),
            "tree_str": tree_to_string(best_ind["tree"]),
            "infix": tree_to_infix(best_ind["tree"]),
            "tree": copy.deepcopy(best_ind["tree"]),
        })

        if best_ind["fitness"] > best_ever_fitness:
            best_ever_fitness = best_ind["fitness"]
            best_ever = copy.deepcopy(best_ind)
            print(f"\n🎯 NEW BEST! Fitness: {best_ever_fitness:.4f}")
            print(
                f"Tree complexity: {count_nodes(best_ever['tree'])} nodes, "
                f"depth {get_tree_depth(best_ever['tree'])}, "
                f"uses {count_input_diversity(best_ever['tree'])}/3 inputs"
            )

        if gen < generations - 1:
            population = evolve_population(
                population,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                elite_size=elite_size,
            )

    print("\n" + "=" * 60)
    print("Evolution Complete!")
    print(f"Best fitness achieved: {best_ever_fitness:.4f}")
    print("=" * 60)

    return best_ever, gen_best_log


# ============================================================================
# MISSING DATA INJECTION
# ============================================================================

def introduce_missing_data(df: pd.DataFrame, missing_rate: float = 0.10,
                          mechanism: str = "MCAR", seed: int = 42):
    np.random.seed(seed)
    df_missing = df.copy()
    n_rows, n_cols = df.shape

    missing_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    if mechanism == "MCAR":
        print("Strategy: Uniform random selection across all values")
        total_values = n_rows * n_cols
        n_missing = int(total_values * missing_rate)

        positions = np.random.choice(total_values, size=n_missing, replace=False)
        for pos in positions:
            row_idx = pos // n_cols
            col_idx = pos % n_cols
            df_missing.iloc[row_idx, col_idx] = np.nan
            missing_mask.iloc[row_idx, col_idx] = True

    elif mechanism == "MAR":
        print("Strategy: Missingness depends on other features")
        first_col_values = df.iloc[:, 0].values
        threshold = np.percentile(first_col_values, 50)

        for col_idx in range(1, n_cols):
            col = df.columns[col_idx]
            probs = np.where(
                first_col_values > threshold,
                missing_rate * 2,
                missing_rate * 0.5,
            )
            missing_in_col = np.random.random(n_rows) < probs
            df_missing.loc[missing_in_col, col] = np.nan
            missing_mask.loc[missing_in_col, col] = True

    elif mechanism == "MNAR":
        print("Strategy: High/low values are more likely to be missing")
        for col in df.columns:
            col_values = df[col].values
            lower = np.percentile(col_values, 25)
            upper = np.percentile(col_values, 75)
            is_extreme = (col_values < lower) | (col_values > upper)

            probs = np.where(
                is_extreme,
                missing_rate * 1.5,
                missing_rate * 0.5,
            )
            missing_in_col = np.random.random(n_rows) < probs
            df_missing.loc[missing_in_col, col] = np.nan
            missing_mask.loc[missing_in_col, col] = True

    else:
        raise ValueError(f"Unknown mechanism: {mechanism}. Use 'MCAR', 'MAR', or 'MNAR'.")

    actual_missing_rate = df_missing.isnull().sum().sum() / (n_rows * n_cols)

    print("\nResults:")
    print(f"  Total values: {n_rows * n_cols}")
    print(f"  Missing values introduced: {df_missing.isnull().sum().sum()}")
    print(f"  Actual missing rate: {actual_missing_rate * 100:.2f}%")
    print(
        f"  Missing values per feature (range): "
        f"{df_missing.isnull().sum().min()} to {df_missing.isnull().sum().max()}"
    )
    print("=" * 60 + "\n")

    return df_missing, missing_mask


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_sonar_dataset(csv_path: str = "HouseVotes84 (1) (1).csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, na_values='?')
        print(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")

        if "Class" not in df.columns:
            raise ValueError("Expected a 'Class' column in the HouseVotes84 CSV.")

        feature_cols = [c for c in df.columns if c != "Class"]
        df[feature_cols] = df[feature_cols].replace({"y": 1, "n": 0})

        label = df["Class"]
        df = df.drop(columns=["Class"])
        df["Class"] = label

        df = df.apply(pd.to_numeric, errors="coerce")

        existing_missing = df.isnull().sum().sum()
        print(f"Total missing entries in dataset: {int(existing_missing)}")

        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def train_final_model(activation_tree: dict, X_train: torch.Tensor, y_train: torch.Tensor,
                     X_val: torch.Tensor, y_val: torch.Tensor, missing_train: torch.Tensor,
                     missing_val: torch.Tensor, conf_train: torch.Tensor, conf_val: torch.Tensor,
                     epochs: int = 100, patience: int = 15, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    missing_train = missing_train.to(device)
    missing_val = missing_val.to(device)
    conf_train = conf_train.to(device)
    conf_val = conf_val.to(device)

    model = ThreeChannelMLP(
        input_size=X_train.shape[1],
        hidden_sizes=[64, 32],
        output_size=len(torch.unique(y_train)),
        activation_tree=activation_tree,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_val_acc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train, missing_train, conf_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val, missing_val, conf_val)
            _, val_pred = torch.max(val_outputs, 1)
            val_acc = (val_pred == y_val).float().mean().item()

            train_outputs = model(X_train, missing_train, conf_train)
            _, train_pred = torch.max(train_outputs, 1)
            train_acc = (train_pred == y_train).float().mean().item()

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} - "
                f"Loss: {loss.item():.4f}, "
                f"Train Acc: {train_acc:.4f}, "
                f"Val Acc: {val_acc:.4f}, "
                f"Best Val: {best_val_acc:.4f}"
            )

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nFinal Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    return model, best_val_acc


# ============================================================================
# MAIN EXPERIMENT PIPELINE
# ============================================================================

def test_evolved_functions_on_sonar(missing_rate=0.10, missing_mechanism="MAR",
                                   population_size=30, generations=30, max_depth=4,
                                   mutation_rate=0.1, crossover_rate=0.7, elite_size=2,
                                   hidden_sizes=[64, 32], num_epochs_fitness=30,
                                   final_epochs=100, final_patience=15):
    print("=" * 60)
    print(f"EVOLVING ACTIVATION FUNCTIONS")
    print(f"WITH {missing_rate * 100:.0f}% MISSING DATA ({missing_mechanism})")
    print("=" * 60)

    print("\nStep 1: Loading dataset...")
    df = load_sonar_dataset()
    if df is None:
        return

    X_full = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_with_missing, missing_mask = introduce_missing_data(
        X_full, missing_rate=missing_rate, mechanism=missing_mechanism, seed=42
    )
    df_with_missing = pd.concat([X_with_missing, y], axis=1)

    X = df_with_missing.iloc[:, :-1]
    y = df_with_missing.iloc[:, -1]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    train_means = X_train.mean()

    # TRAIN
    values_train = X_train.copy()
    missing_flags_train = X_train.isnull().astype(float)
    confidence_train = (~X_train.isnull()).astype(float)
    values_train = values_train.fillna(train_means)

    train_missing_rates = {}
    for col in X_train.columns:
        mr = missing_flags_train[col].mean()
        train_missing_rates[col] = mr
        imputed_conf = max(0.1, 1.0 - mr)
        confidence_train.loc[missing_flags_train[col] == 1, col] = imputed_conf

    # VAL
    values_val = X_val.copy()
    missing_flags_val = X_val.isnull().astype(float)
    confidence_val = (~X_val.isnull()).astype(float)
    values_val = values_val.fillna(train_means)
    for col in X_val.columns:
        mr = train_missing_rates[col]
        imputed_conf = max(0.1, 1.0 - mr)
        confidence_val.loc[missing_flags_val[col] == 1, col] = imputed_conf

    # TEST
    values_test = X_test.copy()
    missing_flags_test = X_test.isnull().astype(float)
    confidence_test = (~X_test.isnull()).astype(float)
    values_test = values_test.fillna(train_means)
    for col in X_test.columns:
        mr = train_missing_rates[col]
        imputed_conf = max(0.1, 1.0 - mr)
        confidence_test.loc[missing_flags_test[col] == 1, col] = imputed_conf

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(values_train)
    X_val_scaled = scaler.transform(values_val)
    X_test_scaled = scaler.transform(values_test)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

    y_train_t = torch.tensor(y_train_enc, dtype=torch.long)
    y_val_t = torch.tensor(y_val_enc, dtype=torch.long)
    y_test_t = torch.tensor(y_test_enc, dtype=torch.long)

    missing_train_t = torch.tensor(missing_flags_train.values, dtype=torch.float32)
    missing_val_t = torch.tensor(missing_flags_val.values, dtype=torch.float32)
    missing_test_t = torch.tensor(missing_flags_test.values, dtype=torch.float32)

    conf_train_t = torch.tensor(confidence_train.values, dtype=torch.float32)
    conf_val_t = torch.tensor(confidence_val.values, dtype=torch.float32)
    conf_test_t = torch.tensor(confidence_test.values, dtype=torch.float32)

    best_activation, gen_best_log = run_evolution_with_nn(
        X_train_t, y_train_t, X_val_t, y_val_t,
        missing_train_t, missing_val_t, conf_train_t, conf_val_t,
        population_size=population_size, generations=generations, max_depth=max_depth,
        mutation_rate=mutation_rate, crossover_rate=crossover_rate, elite_size=elite_size,
        hidden_sizes=hidden_sizes, num_epochs_fitness=num_epochs_fitness
    )

    plot_generational_shapes(
        gen_best_log, evaluate_tree_fn=evaluate_tree,
        selected_generations=None, save_path=None, show=False
    )

    MAX_LEN = 600
    for row in gen_best_log:
        expr = row["infix"]
        if len(expr) > MAX_LEN:
            expr = expr[:MAX_LEN] + "..."

    infix = tree_to_infix(best_activation["tree"])
    expr = tree_to_string(best_activation["tree"])
    expr = (
        expr.replace("add", "+")
        .replace("subtract", "-")
        .replace("multiply", "*")
        .replace("divide", "/")
        .replace("max", "max")
        .replace("min", "min")
        .replace("sigmoid", "σ")
        .replace("tanh", "tanh")
    )
    print("\nWinner (compact form):\n", expr)

    os.makedirs("evo_logs", exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "hepatitis",
        "missing_rate": float(missing_rate),
        "mechanism": missing_mechanism,
        "best": {
            "fitness": float(best_activation["fitness"]),
            "nodes": int(count_nodes(best_activation["tree"])),
            "depth": int(get_tree_depth(best_activation["tree"])),
            "uses_inputs": int(count_input_diversity(best_activation["tree"])),
            "tree_str": tree_to_string(best_activation["tree"]),
        },
        "per_generation": gen_best_log,
    }
    out_name = f"evo_logs/winner_run_{int(missing_rate * 100)}pct_{missing_mechanism}_{int(time.time())}.json"
    with open(out_name, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Saved run report to: {out_name}")

    print(f"\nValidation fitness: {best_activation['fitness']:.4f}")
    print(f"Complexity: {count_nodes(best_activation['tree'])} nodes")
    print(f"Depth: {get_tree_depth(best_activation['tree'])}")
    print(f"Uses {count_input_diversity(best_activation['tree'])}/3 input channels")

    print("\n" + "=" * 60)
    print("VISUALIZING THE EVOLVED FUNCTION:")
    print("=" * 60)

    try:
        import matplotlib.pyplot as plt

        x_range = torch.linspace(-5, 5, 200)
        x_range = torch.where(
            torch.abs(x_range) < 0.01,
            torch.sign(x_range) * 0.01,
            x_range,
        )
        m_vals = torch.zeros_like(x_range)
        c_vals = torch.ones_like(x_range)

        with torch.no_grad():
            y_vals = evaluate_tree(best_activation["tree"], x_range, m_vals, c_vals)
            y_vals = torch.clamp(y_vals, -10, 10)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(x_range.cpu().numpy(), y_vals.cpu().numpy(), label="Evolved", linewidth=2)
        plt.plot(
            x_range.cpu().numpy(),
            torch.relu(x_range).cpu().numpy(),
            "--",
            label="ReLU",
            alpha=0.7,
            linewidth=2,
        )
        plt.xlabel("x (value)", fontsize=10)
        plt.ylabel("activation(x)", fontsize=10)
        plt.title("Evolved vs ReLU\n(missing=0, confidence=1)", fontsize=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-5, 5)

        c_low = torch.ones_like(x_range) * 0.3
        c_high = torch.ones_like(x_range) * 1.0
        with torch.no_grad():
            y_low = evaluate_tree(best_activation["tree"], x_range, m_vals, c_low)
            y_high = evaluate_tree(best_activation["tree"], x_range, m_vals, c_high)
            y_low = torch.clamp(y_low, -10, 10)
            y_high = torch.clamp(y_high, -10, 10)

        plt.subplot(1, 3, 2)
        plt.plot(x_range.cpu().numpy(), y_high.cpu().numpy(), label="c=1.0", linewidth=2)
        plt.plot(x_range.cpu().numpy(), y_low.cpu().numpy(), label="c=0.3", alpha=0.7, linewidth=2)
        plt.xlabel("x (value)", fontsize=10)
        plt.ylabel("activation(x)", fontsize=10)
        plt.title("Effect of Confidence\n(missing=0)", fontsize=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-5, 5)

        m_missing = torch.ones_like(x_range)
        m_observed = torch.zeros_like(x_range)
        with torch.no_grad():
            y_missing = evaluate_tree(best_activation["tree"], x_range, m_missing, c_vals)
            y_observed = evaluate_tree(best_activation["tree"], x_range, m_observed, c_vals)
            y_missing = torch.clamp(y_missing, -10, 10)
            y_observed = torch.clamp(y_observed, -10, 10)

        plt.subplot(1, 3, 3)
        plt.plot(x_range.cpu().numpy(), y_observed.cpu().numpy(), label="m=0", linewidth=2)
        plt.plot(x_range.cpu().numpy(), y_missing.cpu().numpy(), label="m=1", alpha=0.7, linewidth=2)
        plt.xlabel("x (value)", fontsize=10)
        plt.ylabel("activation(x)", fontsize=10)
        plt.title("Effect of Missing Flag\n(confidence=1)", fontsize=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-5, 5)

        plt.tight_layout()
        filename = (
            f"/mnt/user-data/outputs/evolved_activation_sonar_"
            f"{int(missing_rate * 100)}pct_{missing_mechanism}.png"
        )
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to: {filename}")
        plt.close()

    except Exception as e:
        print(f"⚠️ Could not create visualization: {e}")

    print("\nStep 6: Training final model with best activation...")

    final_model, final_val_acc = train_final_model(
        best_activation["tree"], X_train_t, y_train_t, X_val_t, y_val_t,
        missing_train_t, missing_val_t, conf_train_t, conf_val_t,
        epochs=final_epochs, patience=final_patience
    )

    print("\nStep 7: Evaluating on held-out test set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model.eval()

    with torch.no_grad():
        test_outputs = final_model(
            X_test_t.to(device),
            missing_test_t.to(device),
            conf_test_t.to(device),
        )

        _, test_pred = torch.max(test_outputs, 1)
        test_acc = (test_pred == y_test_t.to(device)).float().mean().item()

        y_true = y_test_t.cpu().numpy()
        y_pred_evolved = test_pred.cpu().numpy()
        probs_evolved = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()

        prec_e, rec_e, f1_e, auc_e = compute_extra_metrics(
            y_true, y_pred_evolved, probs_evolved
        )
        tn_e, fp_e, fn_e, tp_e = confusion_matrix(y_true, y_pred_evolved).ravel()
        spec_e = tn_e / (tn_e + fp_e) if (tn_e + fp_e) > 0 else 0.0

    print(f"Test Accuracy: {test_acc:.4f}")

    relu_tree = create_binary_node("max", create_input_node("value"), create_constant_node(0.0))
    swish_tree = create_binary_node(
        "multiply",
        create_input_node("value"),
        create_unary_node("sigmoid", create_input_node("value")),
    )

    print("\n" + "=" * 60)
    print("BASELINE COMPARISON: Training with ReLU")
    print("=" * 60)
    relu_model, relu_val_acc = train_final_model(
        relu_tree, X_train_t, y_train_t, X_val_t, y_val_t,
        missing_train_t, missing_val_t, conf_train_t, conf_val_t,
        epochs=100, patience=15
    )
    relu_model.eval()
    with torch.no_grad():
        relu_test_outputs = relu_model(
            X_test_t.to(device),
            missing_test_t.to(device),
            conf_test_t.to(device),
        )
        _, relu_pred = torch.max(relu_test_outputs, 1)
        relu_test_acc = (relu_pred == y_test_t.to(device)).float().mean().item()

        y_pred_relu = relu_pred.cpu().numpy()
        probs_relu = torch.softmax(relu_test_outputs, dim=1)[:, 1].cpu().numpy()

        prec_r, rec_r, f1_r, auc_r = compute_extra_metrics(
            y_true, y_pred_relu, probs_relu
        )
        tn_r, fp_r, fn_r, tp_r = confusion_matrix(y_true, relu_pred.cpu().numpy()).ravel()
        spec_r = tn_r / (tn_r + fp_r) if (tn_r + fp_r) > 0 else 0.0
    print(f"ReLU Test Accuracy: {relu_test_acc:.4f}")

    print("\n" + "=" * 60)
    print("BASELINE COMPARISON: Training with Swish")
    print("=" * 60)
    swish_model, swish_val_acc = train_final_model(
        swish_tree, X_train_t, y_train_t, X_val_t, y_val_t,
        missing_train_t, missing_val_t, conf_train_t, conf_val_t,
        epochs=100, patience=15
    )
    swish_model.eval()
    with torch.no_grad():
        swish_test_outputs = swish_model(
            X_test_t.to(device),
            missing_test_t.to(device),
            conf_test_t.to(device),
        )
        _, swish_pred = torch.max(swish_test_outputs, 1)
        swish_test_acc = (swish_pred == y_test_t.to(device)).float().mean().item()

        y_pred_swish = swish_pred.cpu().numpy()
        probs_swish = torch.softmax(swish_test_outputs, dim=1)[:, 1].cpu().numpy()

        prec_s, rec_s, f1_s, auc_s = compute_extra_metrics(
            y_true, y_pred_swish, probs_swish
        )
        tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_true, swish_pred.cpu().numpy()).ravel()
        spec_s = tn_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS SUMMARY ({missing_rate * 100:.0f}% Missing, {missing_mechanism})")
    print("=" * 60)
    print(
        f"{'Method':<18} "
        f"{'ValAcc':<10} {'TestAcc':<10} "
        f"{'Prec':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}"
    )
    print("-" * 60)
    print(
        f"{'Evolved':<18} "
        f"{final_val_acc:<10.4f} {test_acc:<10.4f} "
        f"{prec_e:<10.4f} {rec_e:<10.4f} {f1_e:<10.4f} {auc_e:<10.4f}"
    )
    print(
        f"{'ReLU':<18} "
        f"{relu_val_acc:<10.4f} {relu_test_acc:<10.4f} "
        f"{prec_r:<10.4f} {rec_r:<10.4f} {f1_r:<10.4f} {auc_r:<10.4f}"
    )
    print(
        f"{'Swish':<18} "
        f"{swish_val_acc:<10.4f} {swish_test_acc:<10.4f} "
        f"{prec_s:<10.4f} {rec_s:<10.4f} {f1_s:<10.4f} {auc_s:<10.4f}"
    )

    best_test = max(test_acc, relu_test_acc, swish_test_acc)
    if test_acc == best_test:
        print("\n✅ Evolved activation is THE BEST! Beats both ReLU and Swish!")
    elif test_acc > relu_test_acc:
        print("\n✅ Evolved activation beats ReLU!")
        if swish_test_acc == best_test:
            print("⚠️ But Swish performed slightly better.")
    else:
        print("\n⚠️ Baselines performed better this time.")
        if swish_test_acc > relu_test_acc:
            print(f"   Best: Swish ({swish_test_acc:.4f})")
        else:
            print(f"   Best: ReLU ({relu_test_acc:.4f})")

    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("RUN WINNER (final):")
    print("=" * 60)
    print(
        f"Fitness={best_activation['fitness']:.4f} | "
        f"Nodes={count_nodes(best_activation['tree'])} | "
        f"Depth={get_tree_depth(best_activation['tree'])} | "
        f"InputsUsed={count_input_diversity(best_activation['tree'])}"
    )

    return {
        "missing_rate": missing_rate,
        "mechanism": missing_mechanism,
        "evolved_val": final_val_acc,
        "evolved_test": test_acc,
        "evolved_precision": prec_e,
        "evolved_recall": rec_e,
        "evolved_specificity": spec_e,
        "evolved_f1": f1_e,
        "evolved_auc": auc_e,
        "relu_val": relu_val_acc,
        "relu_test": relu_test_acc,
        "relu_precision": prec_r,
        "relu_recall": rec_r,
        "relu_specificity": spec_r,
        "relu_f1": f1_r,
        "relu_auc": auc_r,
        "swish_val": swish_val_acc,
        "swish_test": swish_test_acc,
        "swish_precision": prec_s,
        "swish_recall": rec_s,
        "swish_specificity": spec_s,
        "swish_f1": f1_s,
        "swish_auc": auc_s,
        "best_tree": best_activation["tree"],
    }


# ============================================================================
# MULTI-RATE EXPERIMENT DRIVER
# ============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_multiple_missing_rates():
    print("\n" + "🔬" * 30)
    print("RUNNING EXPERIMENTS WITH MULTIPLE MISSING RATES")
    print("🔬" * 30)

    missing_rates = [0.05, 0.10, 0.20, 0.30]
    mechanism = "MCAR"

    results = []
    for rate in missing_rates:
        set_seed(42)
        print("\n" + "#" * 60)
        print(f"# EXPERIMENT: {rate * 100:.0f}% Missing Data ({mechanism})")
        print("#" * 60 + "\n")

        res = test_evolved_functions_on_sonar(
            missing_rate=rate,
            missing_mechanism=mechanism,
        )
        results.append(res)

    print("\n" + "=" * 80)
    print("SUMMARY: COMPARISON ACROSS MISSING RATES")
    print("=" * 80)
    print(
        f"{'Missing %':<12} {'Evolved Val':<15} {'Evolved Test':<15} "
        f"{'ReLU Test':<15} {'Swish Test':<15}"
    )
    print("-" * 80)
    for res in results:
        print(
            f"{res['missing_rate'] * 100:<12.0f} "
            f"{res['evolved_val']:<15.4f} "
            f"{res['evolved_test']:<15.4f} "
            f"{res['relu_test']:<15.4f} "
            f"{res['swish_test']:<15.4f}"
        )
    print("=" * 80)


def run_repeated_experiments(num_runs: int, missing_rate: float, missing_mechanism: str,
                            base_seed: int, population_size: int, generations: int,
                            max_depth: int, mutation_rate: float, crossover_rate: float,
                            elite_size: int, hidden_sizes, num_epochs_fitness: int,
                            final_epochs: int, final_patience: int):
    metrics = [
        "evolved_val", "evolved_test",
        "evolved_precision", "evolved_recall", "evolved_specificity", "evolved_f1", "evolved_auc",
        "relu_val", "relu_test",
        "relu_precision", "relu_recall", "relu_specificity", "relu_f1", "relu_auc",
        "swish_val", "swish_test",
        "swish_precision", "swish_recall", "swish_specificity", "swish_f1", "swish_auc",
    ]
    all_results = {m: [] for m in metrics}

    winner_infos = []

    print("\n" + "=" * 80)
    print(f"REPEATED EXPERIMENTS: {num_runs} runs "
          f"({missing_rate * 100:.0f}% missing, {missing_mechanism})")
    print("=" * 80)

    for run_idx in range(num_runs):
        seed = base_seed + run_idx
        print("\n" + "-" * 60)
        print(f"RUN {run_idx + 1}/{num_runs} (seed={seed})")
        print("-" * 60)

        set_seed(seed)

        res = test_evolved_functions_on_sonar(
            missing_rate=missing_rate,
            missing_mechanism=missing_mechanism,
            population_size=population_size,
            generations=generations,
            max_depth=max_depth,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elite_size=elite_size,
            hidden_sizes=hidden_sizes,
            num_epochs_fitness=num_epochs_fitness,
            final_epochs=final_epochs,
            final_patience=final_patience,
        )
        if res is None:
            print("⚠️ Run returned None, skipping.")
            continue

        for m in metrics:
            all_results[m].append(res[m])

        best_tree = res["best_tree"]
        winner_infix = tree_to_infix(best_tree)
        f1_this = res["evolved_f1"]
        test_this = res["evolved_test"]

        winner_infos.append({
            "tree": best_tree,
            "formula": winner_infix,
            "f1": f1_this,
            "test": test_this,
        })

        print("\nWinner of this run (math/infix):")
        print(f"  Run {run_idx + 1}:")
        print(f"     Test Accuracy = {test_this:.4f}")
        print(f"     F1 Score      = {f1_this:.4f}")
        print(f"     f(x, m, c)    = {winner_infix}")
        print("-" * 60)

    for m in metrics:
        all_results[m] = np.array(all_results[m], dtype=float)

    print("\n" + "=" * 80)

    print_aggregated_results(all_results)

    print("-" * 80)

    print("\nWINNER FORMULAS ACROSS RUNS:")
    print("=" * 80)
    for i, win in enumerate(winner_infos, start=1):
        print(f"Run {i}:")
        print(f"   Test Accuracy = {win['test']:.4f}")
        print(f"   F1 Score      = {win['f1']:.4f}")
        print(f"   f(x, m, c)    = {win['formula']}")
        print("-" * 80)
    print("=" * 80)

    winner_formulas = [w["formula"] for w in winner_infos]
    winner_trees = [w["tree"] for w in winner_infos]
    winner_f1s = [w["f1"] for w in winner_infos]
    winner_test_accs = [w["test"] for w in winner_infos]

    return {
        "metrics": all_results,
        "winner_infos": winner_infos,
        "winner_formulas": winner_formulas,
        "winner_trees": winner_trees,
        "winner_f1s": winner_f1s,
        "winner_test_accs": winner_test_accs,
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    CONFIG = {
        "missing_rate": 0.00,
        "missing_mechanism": "MAR",
        "population_size": 80,
        "generations": 20,
        "max_tree_depth": 6,
        "mutation_rate": 0.15,
        "crossover_rate": 0.70,
        "elite_size": 2,
        "hidden_sizes": [64, 32],
        "num_epochs_fitness": 30,
        "final_epochs": 100,
        "final_patience": 15,
        "base_seed": 42,
        "num_runs": 15,
    }

    set_seed(CONFIG["base_seed"])

    print("=== SINGLE EXPERIMENT ===\n")
    test_evolved_functions_on_sonar(
        missing_rate=CONFIG["missing_rate"],
        missing_mechanism=CONFIG["missing_mechanism"],
        population_size=CONFIG["population_size"],
        generations=CONFIG["generations"],
        max_depth=CONFIG["max_tree_depth"],
        mutation_rate=CONFIG["mutation_rate"],
        crossover_rate=CONFIG["crossover_rate"],
        elite_size=CONFIG["elite_size"],
        hidden_sizes=CONFIG["hidden_sizes"],
        num_epochs_fitness=CONFIG["num_epochs_fitness"],
        final_epochs=CONFIG["final_epochs"],
        final_patience=CONFIG["final_patience"]
    )

    print("\n=== REPEATED EXPERIMENTS ===\n")
    results_multi = run_repeated_experiments(
        num_runs=CONFIG["num_runs"],
        missing_rate=CONFIG["missing_rate"],
        missing_mechanism=CONFIG["missing_mechanism"],
        base_seed=CONFIG["base_seed"],
        population_size=CONFIG["population_size"],
        generations=CONFIG["generations"],
        max_depth=CONFIG["max_tree_depth"],
        mutation_rate=CONFIG["mutation_rate"],
        crossover_rate=CONFIG["crossover_rate"],
        elite_size=CONFIG["elite_size"],
        hidden_sizes=CONFIG["hidden_sizes"],
        num_epochs_fitness=CONFIG["num_epochs_fitness"],
        final_epochs=CONFIG["final_epochs"],
        final_patience=CONFIG["final_patience"],
    )

    winner_trees = results_multi["winner_trees"]

    plot_run_winners(
        winner_trees,
        evaluate_tree_fn=evaluate_tree,
        save_path="activation_run_winners.pdf",
        show=True,
    )

    print("\n=== All tests completed ===")
