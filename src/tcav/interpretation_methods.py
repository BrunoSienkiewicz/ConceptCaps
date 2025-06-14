from functools import reduce

import torch
from captum.attr import LayerFeatureAblation, LayerGradientXActivation


def layer_feature_ablation(
        n_groups: int,
        tcav,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        experimental_set,
        layers: list[str],
        device: torch.device = torch.device("cpu")
    ):
    if n_groups == 0:
        layer_masks = None
    else:
        # For now I am creating the layer masks by grouping adjacent neurons
        # into groups of size n_groups. Later we can add a more sophisticated
        # approach to create the layer masks.
        layer_masks = []
        for layer in layers:
            layer_shape = reduce(getattr, layer.split("."), model).weight.shape[0]
            layer_mask = torch.zeros(layer_shape).to(device)
            group_size = layer_mask.shape[0] // n_groups
            for i in range(n_groups + 1):
                layer_mask[i * group_size : (i + 1) * group_size] = i
            layer_masks.append(layer_mask)

        layer_masks = (*layer_masks,)

    tcav.layer_attr_method = LayerFeatureAblation(
        model.forward,
        None
    )
    tcav_scores = tcav.interpret(
        inputs=(input_ids, attention_mask, None),
        experimental_sets=experimental_set,
        target=0,
        layer_mask=layer_masks,
    )
    return tcav_scores

def layer_gradient_x_activation(
        tcav,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        experimental_set,
        layers: list[str],
        device: torch.device = torch.device("cpu")
    ):
    layer = [reduce(getattr, layer.split("."), model) for layer in layers]
    tcav.layer_attr_method = LayerGradientXActivation(
        model.forward,
        layer=layer,
    )
    tcav_scores = tcav.interpret(
        inputs=(input_ids, attention_mask, None),
        experimental_sets=experimental_set,
        target=0,
        attribute_to_layer_input=True
    )
    return tcav_scores
