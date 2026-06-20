from __future__ import annotations

from collections import Counter
from typing import Literal

import numpy as np

from weightlab.metrics import p50_p95_ms, set_seed, timed

Route = Literal["java_lang", "java_coffee", "python_lang", "python_snake"]
ROUTES: tuple[Route, ...] = ("java_lang", "java_coffee", "python_lang", "python_snake")
ROUTE_KEYWORDS: dict[Route, set[str]] = {
    "java_lang": {"compile", "class", "method", "package", "jvm", "gradle"},
    "java_coffee": {"espresso", "roast", "beans", "cafe", "barista", "arabica"},
    "python_lang": {"import", "function", "pytest", "venv", "package", "typing"},
    "python_snake": {"venom", "habitat", "reptile", "scales", "pythonidae", "constrictor"},
}


def _sample_polysemous_contexts(seed: int, n_samples: int) -> list[dict[str, str]]:
    rng = set_seed(seed)
    rows: list[dict[str, str]] = []
    templates = [
        ("java", "compile class method package", "java_lang"),
        ("java", "espresso roast beans cafe", "java_coffee"),
        ("python", "import function package pytest", "python_lang"),
        ("python", "venom habitat reptile scales", "python_snake"),
    ]
    for _ in range(n_samples):
        token, context, route = templates[int(rng.integers(0, len(templates)))]
        noise = " docs mixed issue " if rng.random() < 0.25 else " "
        rows.append({"token": token, "context": context + noise, "route": route})
    return rows


def _static_token_route(row: dict[str, str]) -> Route:
    return "java_lang" if row["token"] == "java" else "python_lang"


def _flat_contextual_route(row: dict[str, str]) -> Route:
    context = row["context"]
    if "espresso" in context or "beans" in context:
        return "java_coffee"
    if "venom" in context or "reptile" in context:
        return "python_snake"
    if row["token"] == "java":
        return "java_lang"
    return "python_lang"


def _hierarchical_contextual_route(row: dict[str, str]) -> Route:
    context = row["context"]
    broad = (
        "technical"
        if {"compile", "import", "pytest", "method"} & set(context.split())
        else "natural"
    )
    if broad == "technical":
        return "java_lang" if row["token"] == "java" else "python_lang"
    return "java_coffee" if row["token"] == "java" else "python_snake"


def _random_route(row: dict[str, str], rng: np.random.Generator) -> Route:
    del row
    return ["java_lang", "java_coffee", "python_lang", "python_snake"][int(rng.integers(0, 4))]


def _entropy(routes: list[str]) -> float:
    counts = Counter(routes)
    total = sum(counts.values())
    return float(-sum((c / total) * np.log2(c / total) for c in counts.values()))


def evaluate_contextual_routing(seed: int = 0, n_samples: int = 400) -> dict[str, dict[str, float]]:
    rows = _sample_polysemous_contexts(seed, n_samples)
    rng = set_seed(seed + 1000)

    methods = {
        "static_token": lambda row: _static_token_route(row),
        "flat_contextual": lambda row: _flat_contextual_route(row),
        "hierarchical_contextual": lambda row: _hierarchical_contextual_route(row),
        "random_control": lambda row: _random_route(row, rng),
    }
    active_components = {
        "static_token": 2,
        "flat_contextual": 4,
        "hierarchical_contextual": 3,
        "random_control": 4,
    }

    results: dict[str, dict[str, float]] = {}
    for name, router in methods.items():

        def run(router=router) -> list[str]:
            return [router(row) for row in rows]

        predictions, samples = timed(run, repeats=5)
        correct = sum(pred == row["route"] for pred, row in zip(predictions, rows, strict=True))
        usage = Counter(predictions)
        max_share = max(usage.values()) / len(predictions)
        latencies = p50_p95_ms(samples)
        results[name] = {
            "route_accuracy": correct / len(rows),
            "routing_entropy": _entropy(predictions),
            "expert_max_share": max_share,
            "active_components": float(active_components[name]),
            "estimated_memory_traffic_units": float(active_components[name] * len(rows)),
            **latencies,
        }
    return results


def _sample_robust_contexts(seed: int, n_samples: int) -> list[dict[str, object]]:
    rng = set_seed(seed)
    rows: list[dict[str, object]] = []
    route_pairs: list[tuple[Route, Route]] = [
        ("java_lang", "java_coffee"),
        ("python_lang", "python_snake"),
        ("java_lang", "python_lang"),
        ("java_coffee", "python_snake"),
    ]
    for idx in range(n_samples):
        mode = idx % 3
        if mode == 0:
            route = ROUTES[int(rng.integers(0, len(ROUTES)))]
            token = "java" if route.startswith("java") else "python"
            words = sorted(ROUTE_KEYWORDS[route])[:4]
            rows.append(
                {
                    "token": token,
                    "context": " ".join(words),
                    "routes": frozenset({route}),
                    "case": "single",
                }
            )
        elif mode == 1:
            first, second = route_pairs[int(rng.integers(0, len(route_pairs)))]
            token = "java" if "java" in first else "python"
            words = sorted(ROUTE_KEYWORDS[first])[:3] + sorted(ROUTE_KEYWORDS[second])[:3]
            rows.append(
                {
                    "token": token,
                    "context": " ".join(words),
                    "routes": frozenset({first, second}),
                    "case": "mixed",
                }
            )
        else:
            route = ROUTES[int(rng.integers(0, len(ROUTES)))]
            misleading_pool = [other for other in ROUTES if other != route]
            misleading = misleading_pool[int(rng.integers(0, len(misleading_pool)))]
            token = "java" if route.startswith("java") else "python"
            words = sorted(ROUTE_KEYWORDS[route])[:4] + [f"not_{word}" for word in sorted(
                ROUTE_KEYWORDS[misleading]
            )[:2]]
            rows.append(
                {
                    "token": token,
                    "context": " ".join(words),
                    "routes": frozenset({route}),
                    "case": "adversarial",
                }
            )
    return rows


def _single_label_contextual_set(row: dict[str, object]) -> frozenset[Route]:
    pseudo_row = {"token": str(row["token"]), "context": str(row["context"])}
    return frozenset({_flat_contextual_route(pseudo_row)})


def _multi_label_contextual_set(row: dict[str, object]) -> frozenset[Route]:
    tokens = set(str(row["context"]).split())
    selected = {
        route
        for route, keywords in ROUTE_KEYWORDS.items()
        if tokens & keywords
    }
    if not selected:
        return _single_label_contextual_set(row)
    return frozenset(selected)


def _latent_centroid_router(rows: list[dict[str, object]]):
    vocabulary = sorted({word for row in rows for word in str(row["context"]).split()})
    index = {word: idx for idx, word in enumerate(vocabulary)}

    def vectorize(context: str) -> np.ndarray:
        vector = np.zeros(len(vocabulary), dtype=np.float32)
        for word in context.split():
            if word in index and not word.startswith("not_"):
                vector[index[word]] += 1.0
        norm = np.linalg.norm(vector) + 1e-9
        return vector / norm

    prototypes: dict[Route, np.ndarray] = {}
    for route, keywords in ROUTE_KEYWORDS.items():
        prototypes[route] = vectorize(" ".join(sorted(keywords)))

    def route(row: dict[str, object]) -> frozenset[Route]:
        vector = vectorize(str(row["context"]))
        scores = {route: float(vector @ prototype) for route, prototype in prototypes.items()}
        best = max(scores.values())
        threshold = max(0.20, best * 0.72)
        selected = {route for route, score in scores.items() if score >= threshold and score > 0}
        if not selected:
            return _single_label_contextual_set(row)
        return frozenset(selected)

    return route


def _set_metrics(
    predictions: list[frozenset[Route]],
    rows: list[dict[str, object]],
    times: list[float],
    active_components: float,
) -> dict[str, float]:
    exact = [prediction == row["routes"] for prediction, row in zip(predictions, rows, strict=True)]
    recalls = []
    false_positives = []
    mixed_recalls = []
    adversarial_exact = []
    for prediction, row in zip(predictions, rows, strict=True):
        truth = set(row["routes"])
        recalls.append(len(set(prediction) & truth) / len(truth))
        false_positives.append(max(0, len(set(prediction) - truth)))
        if row["case"] == "mixed":
            mixed_recalls.append(recalls[-1])
        if row["case"] == "adversarial":
            adversarial_exact.append(float(prediction == row["routes"]))
    return {
        "exact_set_accuracy": float(np.mean(exact)),
        "route_recall": float(np.mean(recalls)),
        "mixed_route_recall": float(np.mean(mixed_recalls)),
        "adversarial_exact_set_accuracy": float(np.mean(adversarial_exact)),
        "false_positive_routes_per_sample": float(np.mean(false_positives)),
        "active_components": active_components,
        "estimated_memory_traffic_units": float(sum(len(prediction) for prediction in predictions)),
        **p50_p95_ms(times),
    }


def evaluate_routing_robustness(seed: int = 0, n_samples: int = 360) -> dict[str, dict[str, float]]:
    rows = _sample_robust_contexts(seed, n_samples)
    latent_router = _latent_centroid_router(rows)
    methods = {
        "single_label_contextual": (_single_label_contextual_set, 1.0),
        "multi_label_contextual": (_multi_label_contextual_set, 2.0),
        "latent_centroid_multilabel": (latent_router, 2.0),
    }
    results: dict[str, dict[str, float]] = {}
    for name, (router, active_components) in methods.items():

        def run(router=router) -> list[frozenset[Route]]:
            return [router(row) for row in rows]

        predictions, samples = timed(run, repeats=5)
        results[name] = _set_metrics(predictions, rows, samples, active_components)
    return results
