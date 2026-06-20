from weightlab.routing import evaluate_contextual_routing, evaluate_routing_robustness


def test_contextual_router_handles_polysemous_token_better_than_static_router():
    result = evaluate_contextual_routing(seed=7, n_samples=240)

    assert result["static_token"]["route_accuracy"] < 0.75
    assert result["flat_contextual"]["route_accuracy"] > result["static_token"]["route_accuracy"]
    assert (
        result["hierarchical_contextual"]["route_accuracy"]
        >= result["flat_contextual"]["route_accuracy"] - 0.05
    )
    assert result["random_control"]["route_accuracy"] < result["flat_contextual"]["route_accuracy"]
    assert (
        result["hierarchical_contextual"]["active_components"]
        <= result["flat_contextual"]["active_components"]
    )


def test_routing_robustness_handles_mixed_and_adversarial_contexts():
    result = evaluate_routing_robustness(seed=17, n_samples=240)

    assert result["single_label_contextual"]["mixed_route_recall"] < 1.0
    assert result["multi_label_contextual"]["mixed_route_recall"] > result[
        "single_label_contextual"
    ]["mixed_route_recall"]
    assert result["latent_centroid_multilabel"]["exact_set_accuracy"] >= result[
        "single_label_contextual"
    ]["exact_set_accuracy"]
    assert result["latent_centroid_multilabel"]["adversarial_exact_set_accuracy"] >= 0.95
    assert result["latent_centroid_multilabel"]["adversarial_exact_set_accuracy"] > result[
        "multi_label_contextual"
    ]["adversarial_exact_set_accuracy"]
    assert result["multi_label_contextual"]["active_components"] > result[
        "single_label_contextual"
    ]["active_components"]
