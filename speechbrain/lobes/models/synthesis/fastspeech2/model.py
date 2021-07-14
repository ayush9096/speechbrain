"""
Neural Network Module for FastSpeech2 : Fast And High Quality End-to-End Text to Speech

Authors
* Ayush Agarwal 2021
"""


import torch
from torch import nn
from torch.nn import functional as F
from speechbrain.lobes.models.synthesis.fastspeech2.helper.transformer_encoder import LayerNorm

from speechbrain.lobes.models.synthesis.fastspeech2.helper.duration_predictor import DurationPredictor, DurationPredictorLoss
from speechbrain.lobes.models.synthesis.fastspeech2.helper.length_regulator import LengthRegulator
from speechbrain.lobes.models.synthesis.fastspeech2.helper.variance_predictor import VariancePredictor
from speechbrain.lobes.models.synthesis.fastspeech2.helper.embedding import PositionalEncoding, ScaledPositionalEncoding
from speechbrain.lobes.models.synthesis.fastspeech2.helper.transformer_encoder import Encoder as TransformerEncoder
from speechbrain.lobes.models.synthesis.fastspeech2.helper.postnet import Postnet

class FastSpeech2(nn.Module):

    def __init__(
    self,
    # Network Structure Related
    idim: int,
    odim: int,
    adim: int = 384,
    aheads: int = 4,
    elayers: int = 6,
    eunits: int = 1536,
    dlayers: int = 6,
    dunits: int = 1536,
    postnet_layers: int = 5,
    postnet_chans: int = 512,
    postnet_filts: int = 5,
    positionwise_layer_type: str = "conv1d",
    positionwise_conv_kernel_size: int = 1,
    use_scaled_pos_enc: bool = True,
    use_batch_norm: bool = True,
    encoder_normalize_before: bool = True,
    decoder_normalize_before: bool = True,
    encoder_concat_after: bool = False,
    decoder_concat_after: bool = False,
    reduction_factor: int = 1,
    # Duration Predictor
    duration_predictor_layers: int = 2,
    duration_predictor_chans: int = 384,
    duration_predictor_kernel_size: int = 3,
    # Energy Predictor
    energy_predictor_layers: int = 2,
    energy_predictor_chans: int = 384,
    energy_predictor_kernel_size: int = 3,
    energy_predictor_dropout: float = 0.5,
    energy_embed_kernel_size: int = 9,
    energy_embed_dropout: float = 0.5,
    stop_gradient_from_energy_predictor: bool = False,
    # Pitch Predictor
    pitch_predictor_layers: int = 2,
    pitch_predictor_chans: int = 384,
    pitch_predictor_kernel_size: int = 3,
    pitch_predictor_dropout: float = 0.5,
    pitch_embed_kernel_size: int = 9,
    pitch_embed_dropout: float = 0.5,
    stop_gradient_from_pitch_predictor: bool = False,
    # Training Related
    transformer_enc_dropout_rate: float = 0.1,
    transformer_enc_positional_dropout_rate: float = 0.1,
    transformer_enc_attn_dropout_rate: float = 0.1,
    transformer_dec_dropout_rate: float = 0.1,
    transformer_dec_positional_dropout_rate: float = 0.1,
    transformer_dec_attn_dropout_rate: float = 0.1,
    duration_predictor_dropout_rate: float = 0.1,
    postnet_dropout_rate: float = 0.5,
    init_type: str = "xavier_uniform",
    init_enc_alpha: float = 1.0,
    init_dec_alpha: float = 1.0,
    use_masking: bool = False,
    use_weighted_masking: bool = False,
    ):
        super(FastSpeech2, self).__init__()

        # Store Hyperparameters
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
        self.reduction_factor = reduction_factor
        self.stop_gradient_from_pitch_predictor = stop_gradient_from_pitch_predictor
        self.stop_gradient_from_energy_predictor = stop_gradient_from_energy_predictor
        self.use_scaled_pos_enc = use_scaled_pos_enc

        self.padding_idx = 0
        
        # Positional Encoding Class
        pos_enc_class = (ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding)

        # Encoder
        encoder_input_layer = torch.nn.Embedding(
            num_embeddings=idim, embedding_dim=adim, padding_idx=self.padding_idx
        )

        self.encoder = TransformerEncoder(
            idim=idim,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=eunits,
            num_blocks=elayers,
            input_layer=encoder_input_layer,
            dropout_rate=transformer_enc_dropout_rate,
            positional_dropout_rate=transformer_enc_positional_dropout_rate,
            attention_dropout_rate=transformer_enc_attn_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=encoder_normalize_before,
            concat_after=encoder_concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
        )

        # Duration Predictor
        self.duration_predictor = DurationPredictor(
            idim=adim,
            n_layers=duration_predictor_layers,
            n_chans=duration_predictor_chans,
            kernel_size=duration_predictor_kernel_size,
            dropout_rate=duration_predictor_dropout_rate,
        )

        # Pitch Predictor
        self.pitch_predictor = VariancePredictor(
            idim=adim,
            n_layers=pitch_predictor_layers,
            n_chans=pitch_predictor_chans,
            kernel_size=pitch_embed_kernel_size,
            dropout_rate=pitch_predictor_dropout
        )

        # Pitch Embeddings
        self.pitch_embed = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=adim,
                kernel_size=pitch_embed_kernel_size,
                padding=(pitch_embed_kernel_size - 1) // 2
            ),
            torch.nn.Dropout(pitch_embed_dropout)
        )

        # Energy Predictor
        self.energy_predictor = VariancePredictor(
            idim=adim,
            n_layers=energy_predictor_layers,
            n_chans=energy_predictor_chans,
            kernel_size=energy_predictor_kernel_size,
            dropout_rate=energy_predictor_dropout,
        )

        # Energy Embeddings
        self.energy_embed = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=adim,
                kernel_size=energy_embed_kernel_size,
                padding=(energy_embed_kernel_size - 1) // 2,
            ),
            torch.nn.Dropout(energy_embed_dropout),
        )

        # Length Regulator
        self.length_regulator = LengthRegulator()
        
        # Decoder
        self.decoder = TransformerEncoder(
            idim=0,
            attention_dim=adim,
            attention_heads=aheads,
            linear_units=dunits,
            num_blocks=dlayers,
            input_layer=None,
            dropout_rate=transformer_dec_dropout_rate,
            positional_dropout_rate=transformer_dec_positional_dropout_rate,
            attention_dropout_rate=transformer_dec_attn_dropout_rate,
            pos_enc_class=pos_enc_class,
            normalize_before=decoder_normalize_before,
            concat_after=decoder_concat_after,
            positionwise_layer_type=positionwise_layer_type,
            positionwise_conv_kernel_size=positionwise_conv_kernel_size,
        )

        # Final Projection Layer
        self.feat_out = torch.nn.Linear(adim, odim * reduction_factor)

        # PostNet
        self.postnet = (
            None
            if postnet_layers == 0
            else Postnet(
                idim=adim,
                odim=odim,
                n_layers=postnet_layers,
                n_chans=postnet_chans,
                n_filts=postnet_filts,
                use_batch_norm=use_batch_norm,
                dropout_rate=postnet_dropout_rate,
            )    
        )

        # Initialize Parameters
        # self._reset_parameters()

        # Criterions
        self.criterion = FastSpeech2Loss(
            use_masking=use_masking, use_weighted_masking=use_weighted_masking
        )

    
    def forward(
        self,
        text,
        text_lengths,
        speech,
        speech_lengths,
        durations,
        duration_lengths,
        pitch, 
        pitch_lengths,
        energy,
        energy_lengths
    ):
        text = text[:, : text_lengths.max()] # for data-parallel
        speech = speech[:, : speech_lengths.max()]
        durations = durations[:, : duration_lengths.max()]
        pitch = pitch[:, : pitch_lengths.max()]
        energy = energy[:, : energy_lengths.max()]

        batch_size = text.size(0)

        # Add EOS at the last of sequence
        xs = F.pad(text, [0, 1], "constant", self.padding_idx)
        for i, l in enumerate(text_lengths):
            xs[i, l] = self.eos
        ilens = text_lengths + 1

        ys, ds, ps, es = speech, durations, pitch, energy
        olens = speech_lengths

        # forward propogation
        before_outs, after_outs, d_outs, p_outs, e_outs = self._forward(
            xs, ilens, ys, olens, ds, ps, es, is_inference=False
        )

        # modify mod part for ground-truth
        if self.reduction_factor > 1:
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            max_olen = max(olens)
            ys = ys[:, :max_olen]
        
        # Calculate Loss
        if self.postnet is None:
            after_outs = None
        
        # Calculate Loss
        l1_loss, duration_loss, pitch_loss, energy_loss = self.criterion(
            after_outs=after_outs,
            before_outs=before_outs,
            d_outs=d_outs,
            p_outs=p_outs,
            e_outs=e_outs,
            ys=ys,
            ds=ds,
            ps=ps,
            es=es,
            ilens=ilens,
            olens=olens,
        )

        loss = l1_loss + duration_loss + pitch_loss + energy_loss

        return loss
    

    def _forward(
        self,
        xs,
        ilens,
        ys,
        olens,
        ds,
        ps,
        es,
        is_inference: bool = False,
        alpha: float = 1.0,
    ):
        
        x_masks = self._source_masks(ilens)
        hs, _ = self.encoder(xs, x_masks)

        # forward duration predictor and variance predictors
        d_masks = make_pad_mask(ilens).to(xs.device)
        if self.stop_gradient_from_pitch_predictor:
            p_outs = self.pitch_predictor(hs.detach(), d_masks.unsqueeze(-1))
        else:
            p_outs = self.pitch_predictor(hs, d_masks.unsqueeze(-1))
        
        if self.stop_gradient_from_energy_predictor:
            e_outs = self.energy_predictor(hs.detach(), d_masks.unsqueeze(-1))
        else:
            e_outs = self.energy_predictor(hs, d_masks.unsqueeze(-1))
        

        if is_inference:
            d_outs = self.duration_predictor.inference(hs, d_masks)
            p_embs = self.pitch_embed(p_outs.transpose(1, 2)).transpose(1, 2)
            e_embs = self.energy_embed(e_outs.transpose(1, 2)).transpose(1, 2)
            hs = hs + e_embs + p_embs
            hs = self.length_regulator(hs, d_outs, alpha) 
        else:
            d_outs = self.duration_predictor(hs, d_masks)

            # Use Ground-Truth in Training
            p_embs = self.pitch_embed(ps.transpose(1, 2)).transpose(1, 2)
            e_embs = self.energy_embed(es.transpose(1, 2)).transpose(1, 2)
            hs = hs + e_embs + p_embs
            hs = self.length_regulator(hs, ds)
        

        # forward decoder
        if olens is not None and not is_inference:
            if self.reduction_factor > 1:
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                olens_in = olens
            h_masks = self._source_masks(olens_in)
        else:
            h_masks = None
        zs, _ = self.decoder(hs, h_masks)
        before_outs = self.feat_out(zs).view(
            zs.size(0), -1, self.odim
        )

        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)
        
        return before_outs, after_outs, d_outs, p_outs, e_outs

    def inference(
        self,
        text,
        alpha: float = 1.0
    ):
        x = text

        # add eos at the end of sequence
        x = F.pad(x, [0, 1], "constant", self.eos)

        # Setup Batch Axis
        ilens = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        xs, ys = x.unsqueeze(0), None

        _, outs, *_ = self._forward(
            xs,
            ilens,
            ys,
            is_inference=True,
            alpha=alpha,
        )

        return outs[0], None, None
    
    def _source_masks(self):
        pass


