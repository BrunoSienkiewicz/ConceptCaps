import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from functools import reduce

from captum.concept._utils.common import concepts_to_str
from captum.concept import TCAV
from captum.attr import LayerFeatureAblation

from transformers import MusicgenForConditionalGeneration, AutoProcessor

from music_gen_interpretability.tcav.transform import select_samples, create_experimental_set
from music_gen_interpretability.tcav.model import CustomMusicGen, ConceptClassifier

def format_float(f):
    return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))

def plot_tcav_scores(experimental_sets, tcav_scores):
    fig, ax = plt.subplots(1, len(experimental_sets), figsize = (25, 7))

    barWidth = 1 / (len(experimental_sets[0]) + 1)

    for idx_es, concepts in enumerate(experimental_sets):

        concepts = experimental_sets[idx_es]
        concepts_key = concepts_to_str(concepts)

        pos = [np.arange(len(layers))]
        for i in range(1, len(concepts)):
            pos.append([(x + barWidth) for x in pos[i-1]])
        _ax = (ax[idx_es] if len(experimental_sets) > 1 else ax)
        for i in range(len(concepts)):
            val = [format_float(scores['sign_count'][i]) for layer, scores in tcav_scores[concepts_key].items()]
            _ax.bar(pos[i], val, width=barWidth, edgecolor='white', label=concepts[i].name)

        # Add xticks on the middle of the group bars
        _ax.set_xlabel('Set {}'.format(str(idx_es)), fontweight='bold', fontsize=16)
        _ax.set_xticks([r + 0.3 * barWidth for r in range(len(layers))])
        _ax.set_xticklabels(layers, fontsize=16, rotation=45)

        # Create legend & Show graphic
        _ax.legend(fontsize=16)

    plt.tight_layout()
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

random_state = 42
np.random.seed(random_state)
torch.manual_seed(random_state)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(random_state)

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
model.to(device)
model = model.half()  # Convert model weights to float16
model.eval()

data_path = "data/music_caps_transformed_instrument.csv"

df = pd.read_csv(data_path)

experimental_set = create_experimental_set(
    concept_name="instrument_guitar",
    genre="rock",
    data_path=data_path,
    batch_size=10,
    num_samples=20,
    experimental_set_size=5,
)
rock_samples = select_samples(
    df=df,
    concept="instrument_guitar",
    genre="rock",
    num_samples=20,
)
rock_text = [row["caption_without_genre"] for _, row in rock_samples.iterrows()]
inputs = processor(
    text=rock_text,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(device)

# layers=['model.decoder.model.decoder.layers.0.fc1', 'model.decoder.model.decoder.layers.0.fc2', 'model.decoder.lm_heads.0', 'model.decoder.lm_heads.1']
# layers=['model.decoder.model.decoder.layers.0.fc1']
layers=['model.decoder.lm_heads.0', 'model.decoder.lm_heads.1']

custom_model = CustomMusicGen(model, processor, max_new_tokens=256)
instrument_tcav = TCAV(
    model=custom_model,
    model_id="musicgen_small_instrument",
    classifier=ConceptClassifier(),
    layer_attr_method=LayerFeatureAblation(custom_model.forward, None),
    layers=layers,
    show_progress=True,
)

n_groups = 10

layer_masks = []
for layer in layers:
    layer_shape = reduce(getattr, layer.split('.'), custom_model).weight.shape[0]
    print(layer_shape)
    layer_mask = torch.zeros(layer_shape).to(device)
    group_size = layer_mask.shape[0] // n_groups
    for i in range(n_groups+1):
        layer_mask[i * group_size:(i + 1) * group_size] = i
    layer_masks.append(layer_mask)

layer_masks = (*layer_masks,)

tcav_scores = instrument_tcav.interpret(
    inputs=(inputs.input_ids, inputs.attention_mask, None),
    experimental_sets=experimental_set,
    target=0,
    layer_mask=layer_masks,
)
