import copy
import torch

__all__ = [
    'clip_gamma',
    'get_gamma_percentange',
]


def clip_gamma(state_dict, gamma_threshold):
    tmp_state_dict = copy.deepcopy(state_dict)
    for key in tmp_state_dict.keys():
        if ('norm' in key) and ('weight' in key):
            zeros = torch.zeros_like(tmp_state_dict[key])
            tmp_state_dict[key] = torch.where(tmp_state_dict[key].abs() < gamma_threshold,
                                              zeros, tmp_state_dict[key])
    return tmp_state_dict


def get_gamma_percentange(state_dict, gamma_threshold):
    gammas = []
    for key in state_dict.keys():
        if ('norm' in key) and ('weight' in key):
            gammas.append(state_dict[key])
    gammas = torch.cat(gammas)
    percentage = (gammas.abs() < gamma_threshold).sum().item() / len(gammas) * 100
    return percentage
