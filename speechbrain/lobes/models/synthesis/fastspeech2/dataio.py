"""
 Data preprocessing for FastSpeech2

 Authors
 * Ayush Agarwal 2021

"""

# from speechbrain.lobes.models.synthesis.dataio import load_datasets
import speechbrain as sb
import torch
from torchaudio import transforms
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.lobes.models.synthesis.fastspeech2.text_to_sequence import (
    text_to_sequence,
)

DEFAULT_TEXT_CLEANERS = ["english_cleaners"]


def encode_text(
    text_cleaners=None,
    takes="txt",
    provides=["text_sequences", "input_lengths"],
):
    """
    A pipeline function that encodes raw text into a tensor

    Arguments
    ---------
    text_cleaners: list
        an list of text cleaners to use
    takes: str
        the name of the pipeline input
    provides: str
        the name of the pipeline output

    Returns
    -------
    result: DymamicItem
        a pipeline element
    """
    text_cleaners = DEFAULT_TEXT_CLEANERS

    @sb.utils.data_pipeline.takes(takes)
    @sb.utils.data_pipeline.provides(*provides)
    def f(txt):
        sequence = text_to_sequence(txt, text_cleaners)
        yield torch.tensor(sequence, dtype=torch.int32)
        yield torch.tensor(len(sequence), dtype=torch.int32)

    return f


def audio_pipeline(hparams):
    """
    A pipeline function that provides text sequences, a mel spectrogram
    and the text length

    Arguments
    ---------
    hparams: dict
        model hyperparameters

    Returns
    -------
    result: DymamicItem
        a pipeline element
    """
    audio_to_mel = transforms.MelSpectrogram(
        sample_rate=hparams["sample_rate"],
        hop_length=hparams["hop_length"],
        win_length=hparams["win_length"],
        n_fft=hparams["n_fft"],
        n_mels=hparams["n_mel_channels"],
        f_min=hparams["mel_fmin"],
        f_max=hparams["mel_fmax"],
        normalized=hparams["mel_normalized"],
    )

    @sb.utils.data_pipeline.takes("wav", "label")
    @sb.utils.data_pipeline.provides("mel_text_pair")
    def f(file_path, words):
        text_seq = torch.IntTensor(
            text_to_sequence(words, hparams["text_cleaners"])
        )
        audio = sb.dataio.dataio.read_audio(file_path)
        mel = audio_to_mel(audio)
        len_text = len(text_seq)
        yield text_seq, mel, len_text

    return f


def dataset_prep(dataset, hparams):
    """
    Adds pipeline elements for Tacotron to a dataset and
    wraps it in a saveable data loader

    Arguments
    ---------
    dataset: DynamicItemDataSet
        a raw dataset

    Returns
    -------
    result: SaveableDataLoader
        a data loader
    """
    dataset.add_dynamic_item(audio_pipeline(hparams))
    dataset.set_output_keys(["mel_text_pair"])
    return SaveableDataLoader(
        dataset,
        batch_size=hparams["batch_size"],
        collate_fn=TextMelCollate(),
        drop_last=hparams.get("drop_last", False),
    )






class TextMelCollate:
    pass
