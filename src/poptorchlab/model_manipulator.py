import logging
import sys
from typing import List, Dict

import poptorch
import re
import torch
from torch.fx import symbolic_trace

from .layers import SerializedEmbedding, SerializedLinear


def replace_layer(parent, field_name, new_layer):
    if isinstance(parent, torch.nn.Sequential):
        parent[int(field_name)] = new_layer
    else:
        setattr(parent, field_name, new_layer)


def get_module_and_parent_by_name(node, split_tokens):
    child_to_find = split_tokens[0]
    for name, child in node.named_children():
        if name == child_to_find:
            if len(split_tokens) == 1:
                return node, child, name
            else:
                return get_module_and_parent_by_name(child, split_tokens[1:])

    return None, None, None


def pipeline_model(model: torch.nn.Module, pipeline_splits: List[str]):
    for name, _ in model.named_modules():
        name = name.replace('.', '/')
        if name in pipeline_splits:
            logging.debug('--------')
        logging.debug(name)
    for split_idx, split in enumerate(pipeline_splits):
        split_tokens = split.split('/')
        logging.info(f'Processing pipeline split {split_tokens}')
        parent, node, field_or_idx_str = get_module_and_parent_by_name(model, split_tokens)
        if parent is None:
            logging.error(f'Split {split} not found')
            sys.exit()
        else:
            replace_layer(parent, field_or_idx_str, poptorch.BeginBlock(ipu_id=split_idx + 1, layer_to_call=node))


def recompute_model(model: torch.nn.Module, recompute_checkpoints: List[str]):
    # Put recomutation checkpoint if regular expression matches
    traced_model = symbolic_trace(model)
    for node in traced_model.graph.nodes:
        name = str(node).replace('_', '/')
        recompute_checkpoint = False
        for checkpoint_re in recompute_checkpoints:
            if re.fullmatch(checkpoint_re, name):
                logging.info(f"RECOMPUTE CHECKPOINT:{name}")
                recompute_checkpoint = True
                with traced_model.graph.inserting_after(node):
                    new_node = traced_model.graph.call_function(
                        poptorch.recomputationCheckpoint, args=(node,))
                    node.replace_all_uses_with(new_node)
                    new_node.args = (node,)
                break
        if not recompute_checkpoint:
            logging.info(f"RECOMPUTE:{name}")

    traced_model.recompile()
    return traced_model


def _goto_layer_address(model: torch.nn.Module, layer_address: str):
    layer = model
    parent = None
    field_name = None
    for field_name in layer_address.split('/'):
        parent = layer
        layer = parent[int(field_name)] if isinstance(parent, torch.nn.Sequential) else getattr(parent, field_name)
    return parent, field_name, layer


def _serialize_layer(model: torch.nn.Module, layer_address: str, factor: int):
    parent, field_name, layer = _goto_layer_address(model, layer_address)
    if isinstance(layer, torch.nn.Linear):
        replace_layer(parent, field_name, SerializedLinear(layer, factor))
    elif isinstance(layer, torch.nn.Embedding):
        replace_layer(parent, field_name, SerializedEmbedding(layer, factor))
    else:
        raise NotImplementedError(f'Cannot serialize the layer {layer_address} of type {type(layer)}')
    logging.debug(f'Serialized the layer {layer_address} of type {type(layer)} by the factor of {factor}.')


def serialize_model(model: torch.nn.Module, serialization_config: Dict[str, int]):
    for layer_address, factor in serialization_config.items():
        _serialize_layer(model, layer_address, factor)


def _deserialize_layer(model: torch.nn.Module, layer_address: str):
    parent, field_name, layer = _goto_layer_address(model, layer_address)
    if hasattr(layer, 'deserialize'):
        replace_layer(parent, field_name, layer.deserialize())
    else:
        raise Exception(f'The layer {layer_address} is not serialized in the first place. (type: {type(layer)})')
    logging.debug(f'Deserialized the layer {layer_address} back to {type(layer)}.')


def deserialize_model(model: torch.nn.Module, serialization_config: Dict[str, int]):
    for layer_address, factor in serialization_config.items():
        _deserialize_layer(model, layer_address)
