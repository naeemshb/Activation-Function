import math
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from metrics_report import compute_extra_metrics, print_aggregated_results


INPUTS = ["value", "missing", "confidence"]
CONSTANTS = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1]


def create_input_node(input_name):
    assert input_name in INPUTS, f"Invalid input name: {input_name}"
    return {"type": "input", "input_name": input_name, "children": []}


def create_constant_node(value):
    return {"type": "constant", "value": float(value), "children": []}


def create_unary_node(operator, child):
    return {"type": "unary", "operator": operator, "children": [child]}


def create_binary_node(operator, left_child, right_child):
    return {"type": "binary", "operator": operator, "children": [left_child, right_child]}


def safe_divide(x, y):
    denom = torch.where(
        torch.abs(y) < 1e-3,
        torch.sign(y) * 1e-3 + (y == 0).float() * 1e-3,
        y,
    )
    return x / denom


def safe_sqrt(x):
    return torch.sqrt(torch.clamp(x, min=0.0))


def safe_log(x):
    return torch.log(torch.clamp(torch.abs(x), min=1e-8))


def safe_pow(x, power):
    return torch.pow(torch.clamp(torch.abs(x), min=1e-8, max=1000.0), power)


def clip_result(x, min_val=-1000.0, max_val=1000.0):
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
    "leaky_relu": lambda x: torch.nn.functional.leaky_relu(x, negative_slope=0.01),
    "elu": lambda x: torch.nn.functional.elu(x, alpha=1.0),
}

BINARY_OPERATORS = {
    "add": lambda x, y: x + y,
    "subtract": lambda x, y: x - y,
    "multiply": lambda x, y: clip_result(x * y),
    "divide": lambda x, y: safe_divide(x, y),
    "max": lambda x, y: torch.maximum(x, y),
    "min": lambda x, y: torch.minimum(x, y),
}


def evaluate_tree(node, value, missing, confidence):
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


def tree_to_infix(node):
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
            "leaky_relu": f"leakyrelu({s})",
            "elu": f"elu({s})",
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


def get_tree_depth(node):
    t = node["type"]
    if t in ["input", "constant"]:
        return 1
    if t == "unary":
        return 1 + get_tree_depth(node["children"][0])
    if t == "binary":
        return 1 + max(get_tree_depth(node["children"][0]),
                       get_tree_depth(node["children"][1]))
    raise ValueError(f"Invalid node type: {t}")


def count_nodes(node):
    t = node["type"]
    if t in ["input", "constant"]:
        return 1
    if t == "unary":
        return 1 + count_nodes(node["children"][0])
    if t == "binary":
        return 1 + count_nodes(node["children"][0]) + count_nodes(node["children"][1])
    raise ValueError(f"Invalid node type: {t}")


def generate_terminal_node():
    if random.random() < 0.7:
        input_name = random.choice(INPUTS)
        return create_input_node(input_name)
    constant_value = random.choice(CONSTANTS)
    return create_constant_node(constant_value)


def generate_non_terminal_node(max_depth, current_depth):
    if random.random() < 0.4:
        op = random.choice(list(UNARY_OPERATORS.keys()))
        child = generate_random_tree(max_depth, current_depth + 1)
        return create_unary_node(op, child)
    op = random.choice(list(BINARY_OPERATORS.keys()))
    left = generate_random_tree(max_depth, current_depth + 1)
    right = generate_random_tree(max_depth, current_depth + 1)
    return create_binary_node(op, left, right)


def generate_random_tree(max_depth, current_depth=0):
    if current_depth >= max_depth:
        return generate_terminal_node()

    terminal_prob = 0.3 + 0.4 * (current_depth / max_depth)
    if random.random() < terminal_prob:
        return generate_terminal_node()

    return generate_non_terminal_node(max_depth, current_depth)


def validate_tree(node):
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


def get_all_nodes(tree):
    nodes = [tree]
    if tree["type"] in ["unary", "binary"]:
        for ch in tree["children"]:
            nodes.extend(get_all_nodes(ch))
    return nodes


def mutate_single_node(node, max_depth, current_tree_depth):
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


def mutate_tree(tree, mutation_rate=0.1, max_depth=5):
    mutated = copy.deepcopy(tree)
    all_nodes = get_all_nodes(mutated)

    for node in all_nodes:
        if random.random() < mutation_rate:
            mutate_single_node(node, max_depth, get_tree_depth(mutated))

    validate_tree(mutated)
    return mutated


def perform_crossover(node1, node2):
    temp = copy.deepcopy(node1)
    node1.clear()
    node1.update(copy.deepcopy(node2))
    node2.clear()
    node2.update(temp)


def crossover_trees(parent1, parent2, max_depth=5):
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


def create_initial_population(population_size, max_depth=4):
    population = []
    for _ in range(population_size):
        tree = generate_random_tree(max_depth)
        population.append({"tree": tree, "fitness": None})
    return population


def weighted_random_choice(population, probabilities):
    r = random.random()
    cumulative = 0.0
    for ind, p in zip(population, probabilities):
        cumulative += p
        if r <= cumulative:
            return ind
    return population[-1]


def fitness_proportional_selection(population, num_selected):
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


def evolve_population(population, mutation_rate=0.1, crossover_rate=0.7, elite_size=2):
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


def count_input_diversity(node):
    found = set()

    def walk(n):
        if n.get("type") == "input":
            found.add(n["input_name"])
        for ch in n.get("children", []):
            walk(ch)

    walk(node)
    return len(found)


class ChannelProp(nn.Module):
    def __init__(self, linear, eps=1e-8):
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
    def __init__(self, activation_tree):
        super().__init__()
        self.activation_tree = activation_tree

    def forward(self, x, m, c):
        return evaluate_tree(self.activation_tree, x, m, c)


class ThreeChannelBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation_tree):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.prop = ChannelProp(self.lin)
        self.act = AdaptiveLayer(activation_tree)

    def forward(self, x, m, c):
        x, m, c = self.prop(x, m, c)
        x = self.act(x, m, c)
        return x, m, c


class ThreeChannelMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_tree):
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


def evaluate_fitness_neural_network(tree, X_train, y_train, X_val, y_val,
                                    missing_train, missing_val, conf_train, conf_val,
                                    num_epochs=50, device=None, hidden_sizes=None):
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

    except Exception:
        return 0.0


def run_evolution_with_nn(X_train, y_train, X_val, y_val, missing_train, missing_val,
                          conf_train, conf_val, population_size, generations, max_depth,
                          mutation_rate, crossover_rate, elite_size, hidden_sizes,
                          num_epochs_fitness):
    population = create_initial_population(population_size, max_depth)

    best_ever = None
    best_ever_fitness = -float("inf")

    for gen in range(generations):
        for individual in population:
            if individual["fitness"] is None:
                individual["fitness"] = evaluate_fitness_neural_network(
                    individual["tree"],
                    X_train, y_train,
                    X_val, y_val,
                    missing_train, missing_val,
                    conf_train, conf_val,
                    num_epochs=num_epochs_fitness,
                    hidden_sizes=hidden_sizes,
                )

        population.sort(key=lambda x: x["fitness"], reverse=True)
        best_ind = population[0]

        if best_ind["fitness"] > best_ever_fitness:
            best_ever_fitness = best_ind["fitness"]
            best_ever = copy.deepcopy(best_ind)

        if gen < generations - 1:
            population = evolve_population(
                population,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                elite_size=elite_size,
            )

    return best_ever


def introduce_missing_data(df, missing_rate=0.10, mechanism="MCAR", seed=42):
    np.random.seed(seed)
    df_missing = df.copy()
    n_rows, n_cols = df.shape

    missing_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    if mechanism == "MCAR":
        total_values = n_rows * n_cols
        n_missing = int(total_values * missing_rate)

        positions = np.random.choice(total_values, size=n_missing, replace=False)
        for pos in positions:
            row_idx = pos // n_cols
            col_idx = pos % n_cols
            df_missing.iloc[row_idx, col_idx] = np.nan
            missing_mask.iloc[row_idx, col_idx] = True

    elif mechanism == "MAR":
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

    return df_missing, missing_mask


def load_dataset(csv_path="HouseVotes84.csv"):
    df = pd.read_csv(csv_path, na_values='?')

    if "Class" not in df.columns:
        raise ValueError("Expected a 'Class' column in the HouseVotes84 CSV.")

    feature_cols = [c for c in df.columns if c != "Class"]
    df[feature_cols] = df[feature_cols].replace({"y": 1, "n": 0})

    label = df["Class"]
    df = df.drop(columns=["Class"])
    df["Class"] = label

    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def evaluate_model_on_test(model, X_test_t, missing_test_t, conf_test_t, y_test_t, y_true, device):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t.to(device), missing_test_t.to(device), conf_test_t.to(device))
        _, pred = torch.max(outputs, 1)
        test_acc = (pred == y_test_t.to(device)).float().mean().item()
        y_pred = pred.cpu().numpy()
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

    prec, rec, f1, auc = compute_extra_metrics(y_true, y_pred, probs)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return test_acc, prec, rec, spec, f1, auc


def run_single_experiment(missing_rate=0.10, missing_mechanism="MAR", population_size=30,
                          generations=30, max_depth=4, mutation_rate=0.1, crossover_rate=0.7,
                          elite_size=2, hidden_sizes=None, num_epochs_fitness=30,
                          final_epochs=100, final_patience=15):
    if hidden_sizes is None:
        hidden_sizes = [64, 32]

    df = load_dataset()
    if df is None:
        return None

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

    values_val = X_val.copy()
    missing_flags_val = X_val.isnull().astype(float)
    confidence_val = (~X_val.isnull()).astype(float)
    values_val = values_val.fillna(train_means)
    for col in X_val.columns:
        mr = train_missing_rates[col]
        imputed_conf = max(0.1, 1.0 - mr)
        confidence_val.loc[missing_flags_val[col] == 1, col] = imputed_conf

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

    best_activation = run_evolution_with_nn(
        X_train_t, y_train_t, X_val_t, y_val_t,
        missing_train_t, missing_val_t, conf_train_t, conf_val_t,
        population_size=population_size, generations=generations, max_depth=max_depth,
        mutation_rate=mutation_rate, crossover_rate=crossover_rate, elite_size=elite_size,
        hidden_sizes=hidden_sizes, num_epochs_fitness=num_epochs_fitness,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true = y_test_t.cpu().numpy()

    evolved_model, evolved_val = train_final_model(
        best_activation["tree"], X_train_t, y_train_t, X_val_t, y_val_t,
        missing_train_t, missing_val_t, conf_train_t, conf_val_t,
        epochs=final_epochs, patience=final_patience,
    )
    test_acc, prec_e, rec_e, spec_e, f1_e, auc_e = evaluate_model_on_test(
        evolved_model, X_test_t, missing_test_t, conf_test_t, y_test_t, y_true, device
    )

    relu_tree = create_binary_node("max", create_input_node("value"), create_constant_node(0.0))
    swish_tree = create_binary_node(
        "multiply", create_input_node("value"),
        create_unary_node("sigmoid", create_input_node("value")),
    )
    leakyrelu_tree = create_binary_node(
        "max", create_input_node("value"),
        create_binary_node("multiply", create_constant_node(0.1), create_input_node("value")),
    )
    elu_tree = create_binary_node(
        "max", create_input_node("value"),
        create_binary_node("subtract",
                           create_unary_node("exp", create_input_node("value")),
                           create_constant_node(1.0)),
    )

    baseline_results = {}
    for name, tree in [("relu", relu_tree), ("swish", swish_tree),
                       ("leakyrelu", leakyrelu_tree), ("elu", elu_tree)]:
        model, val_acc = train_final_model(
            tree, X_train_t, y_train_t, X_val_t, y_val_t,
            missing_train_t, missing_val_t, conf_train_t, conf_val_t,
            epochs=100, patience=15,
        )
        t_acc, prec, rec, spec, f1, auc = evaluate_model_on_test(
            model, X_test_t, missing_test_t, conf_test_t, y_test_t, y_true, device
        )
        baseline_results[name] = {
            "val": val_acc, "test": t_acc, "precision": prec, "recall": rec,
            "specificity": spec, "f1": f1, "auc": auc,
        }

    res = {
        "missing_rate": missing_rate,
        "mechanism": missing_mechanism,
        "evolved_val": evolved_val, "evolved_test": test_acc,
        "evolved_precision": prec_e, "evolved_recall": rec_e,
        "evolved_specificity": spec_e, "evolved_f1": f1_e, "evolved_auc": auc_e,
        "best_tree": best_activation["tree"],
    }
    for name in ["relu", "swish", "leakyrelu", "elu"]:
        b = baseline_results[name]
        res[f"{name}_val"] = b["val"]
        res[f"{name}_test"] = b["test"]
        res[f"{name}_precision"] = b["precision"]
        res[f"{name}_recall"] = b["recall"]
        res[f"{name}_specificity"] = b["specificity"]
        res[f"{name}_f1"] = b["f1"]
        res[f"{name}_auc"] = b["auc"]

    return res


def train_final_model(activation_tree, X_train, y_train, X_val, y_val,
                      missing_train, missing_val, conf_train, conf_val,
                      epochs=100, patience=15, device=None):
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

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_acc


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_repeated_experiments(num_runs, missing_rate, missing_mechanism, base_seed,
                             population_size, generations, max_depth, mutation_rate,
                             crossover_rate, elite_size, hidden_sizes, num_epochs_fitness,
                             final_epochs, final_patience):
    metrics = [
        "evolved_val", "evolved_test",
        "evolved_precision", "evolved_recall", "evolved_specificity", "evolved_f1", "evolved_auc",
        "relu_val", "relu_test",
        "relu_precision", "relu_recall", "relu_specificity", "relu_f1", "relu_auc",
        "swish_val", "swish_test",
        "swish_precision", "swish_recall", "swish_specificity", "swish_f1", "swish_auc",
        "leakyrelu_val", "leakyrelu_test",
        "leakyrelu_precision", "leakyrelu_recall", "leakyrelu_specificity", "leakyrelu_f1", "leakyrelu_auc",
        "elu_val", "elu_test",
        "elu_precision", "elu_recall", "elu_specificity", "elu_f1", "elu_auc",
    ]
    all_results = {m: [] for m in metrics}
    winner_infos = []

    for run_idx in range(num_runs):
        seed = base_seed + run_idx
        print(f"Run {run_idx + 1}/{num_runs}")
        set_seed(seed)

        res = run_single_experiment(
            missing_rate=missing_rate, missing_mechanism=missing_mechanism,
            population_size=population_size, generations=generations, max_depth=max_depth,
            mutation_rate=mutation_rate, crossover_rate=crossover_rate, elite_size=elite_size,
            hidden_sizes=hidden_sizes, num_epochs_fitness=num_epochs_fitness,
            final_epochs=final_epochs, final_patience=final_patience,
        )
        if res is None:
            continue

        for m in metrics:
            all_results[m].append(res[m])

        winner_infos.append({
            "tree": res["best_tree"],
            "formula": tree_to_infix(res["best_tree"]),
            "f1": res["evolved_f1"],
            "test": res["evolved_test"],
        })

    for m in metrics:
        all_results[m] = np.array(all_results[m], dtype=float)

    print_aggregated_results(all_results)

    return {"metrics": all_results, "winner_infos": winner_infos}


def write_results_and_wilcoxon(results_multi, dataset_name, missing_rate, missing_mechanism, num_runs):
    metrics_to_test = ["test", "precision", "recall", "specificity", "f1", "auc"]
    baselines = ["relu", "swish", "leakyrelu", "elu"]
    baseline_labels = {"relu": "ReLU", "swish": "Swish", "leakyrelu": "LeakyReLU", "elu": "ELU"}
    sym_map = {"relu": "*", "swish": "†", "leakyrelu": "‡", "elu": "§"}

    out_file = f"results_{dataset_name}.txt"
    M = results_multi["metrics"]

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"{'=' * 100}\n")
        f.write("PER-RUN RESULTS AND WILCOXON TESTS\n")
        f.write(f"Missing: {missing_mechanism} {missing_rate * 100:.0f}% | Runs: {num_runs}\n")
        f.write(f"{'=' * 100}\n\n")

        f.write("SECTION 1: PER-RUN VALUES\n")
        f.write(f"{'-' * 100}\n")

        header = f"{'Run':>4}"
        cols = []
        for met in metrics_to_test:
            for method in ["evolved"] + baselines:
                label = "3C-EA" if method == "evolved" else baseline_labels[method]
                col_name = f"{label}_{met}"
                cols.append((method, met, col_name))
                header += f"  {col_name:>14}"
        f.write(header + "\n")

        for i in range(num_runs):
            row = f"{i + 1:>4}"
            for method, met, _ in cols:
                vals = np.asarray(M[f"{method}_{met}"], dtype=float)
                row += f"  {vals[i]:>14.4f}"
            f.write(row + "\n")

        f.write(f"\n{'=' * 100}\n")
        f.write("SECTION 2: AGGREGATED (MEAN +/- STD)\n")
        f.write(f"{'-' * 100}\n")
        f.write(f"{'Method':<12}")
        for met in metrics_to_test:
            f.write(f"  {met:>20}")
        f.write("\n")
        for method in ["evolved"] + baselines:
            label = "3C-EA" if method == "evolved" else baseline_labels[method]
            f.write(f"{label:<12}")
            for met in metrics_to_test:
                vals = np.asarray(M[f"{method}_{met}"], dtype=float)
                f.write(f"  {vals.mean():.4f}+/-{vals.std():.4f}")
            f.write("\n")

        f.write(f"\n{'=' * 100}\n")
        f.write("SECTION 3: WILCOXON SIGNED-RANK TESTS (3C-EA vs each baseline)\n")
        f.write("Alternative hypothesis: 3C-EA > baseline\n")
        f.write("Significance levels: * p<0.05, ** p<0.01, *** p<0.001\n")
        f.write(f"{'=' * 100}\n")

        print("\n" + "=" * 100)
        print("WILCOXON SIGNED-RANK TESTS (3C-EA vs each baseline)")
        print("=" * 100)

        for met in metrics_to_test:
            header_line = (f"\nMetric: {met}\n  {'Baseline':<12} {'n_eff':>6} {'p-value':>10} "
                           f"{'sig':>6} {'mean_diff':>12} {'median_diff':>12} {'W-stat':>8}")
            f.write(header_line + "\n")
            print(header_line)

            evo = np.asarray(M[f"evolved_{met}"], dtype=float)
            for bl in baselines:
                base = np.asarray(M[f"{bl}_{met}"], dtype=float)
                diff = evo - base
                diff_nz = diff[diff != 0]

                if len(diff_nz) == 0:
                    line = (f"  {baseline_labels[bl]:<12} {0:>6} {'--':>10} {'':>6} "
                            f"{0.0:>+12.4f} {0.0:>+12.4f} {'--':>8}")
                    f.write(line + "\n")
                    print(line)
                    continue

                stat, p = wilcoxon(diff_nz, alternative="greater")
                if p < 0.001:
                    sig = "***"
                elif p < 0.01:
                    sig = "**"
                elif p < 0.05:
                    sig = "*"
                else:
                    sig = ""

                line = (f"  {baseline_labels[bl]:<12} {len(diff_nz):>6} {p:>10.6f} {sig:>6} "
                        f"{np.mean(diff_nz):>+12.4f} {np.median(diff_nz):>+12.4f} {stat:>8.0f}")
                f.write(line + "\n")
                print(line)

        f.write(f"\n{'=' * 100}\n")
        f.write("SECTION 4: LATEX SIGNIFICANCE MARKERS\n")
        f.write("Symbols: * / ** vs ReLU, † / †† vs Swish, "
                "‡ / ‡‡ vs LeakyReLU, § / §§ vs ELU\n")
        f.write("One symbol = p < 0.05, two symbols = p < 0.01\n")
        f.write(f"{'-' * 100}\n")

        for met in metrics_to_test:
            evo = np.asarray(M[f"evolved_{met}"], dtype=float)
            markers = []
            for bl in baselines:
                base = np.asarray(M[f"{bl}_{met}"], dtype=float)
                diff = evo - base
                diff_nz = diff[diff != 0]
                if len(diff_nz) == 0:
                    continue
                _, p = wilcoxon(diff_nz, alternative="greater")
                if p < 0.01:
                    markers.append(sym_map[bl] * 2)
                elif p < 0.05:
                    markers.append(sym_map[bl])
            marker_str = "".join(markers) if markers else "(none)"
            f.write(f"  {met:<14}: {marker_str}\n")

    print(f"\nFull results saved to: {out_file}")


CONFIG = {
    "dataset_name": "HouseVotes84",
    "missing_rate": 0.00,
    "missing_mechanism": "MAR",
    "population_size": 100,
    "generations": 30,
    "max_tree_depth": 3,
    "mutation_rate": 0.15,
    "crossover_rate": 0.70,
    "elite_size": 2,
    "hidden_sizes": [64, 32],
    "num_epochs_fitness": 30,
    "final_epochs": 100,
    "final_patience": 15,
    "base_seed": 42,
    "num_runs": 30,
}


if __name__ == "__main__":
    set_seed(CONFIG["base_seed"])

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

    write_results_and_wilcoxon(
        results_multi,
        dataset_name=CONFIG["dataset_name"],
        missing_rate=CONFIG["missing_rate"],
        missing_mechanism=CONFIG["missing_mechanism"],
        num_runs=CONFIG["num_runs"],
    )
