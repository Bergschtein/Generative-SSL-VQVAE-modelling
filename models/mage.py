import copy
import numpy as np
import math
from pathlib import Path
import tempfile
from typing import Union
from collections import deque
from torch import nn
import torch.nn.functional as F

import torch

from einops import repeat, rearrange
from typing import Callable

from models.transformers import AutoEncoderTransformer
from models.encoder_decoder import VQVAEEncoder, VQVAEDecoder
from models.vq import VectorQuantize

from utils import (
    compute_downsample_rate,
    get_root_dir,
    freeze,
    timefreq_to_time,
    time_to_timefreq,
    quantize,
    ssl_config_filename,
)


class MAGE(nn.Module):
    """
    references:

    """

    def __init__(
        self,
        input_length: int,
        choice_temperature: int,
        stochastic_sampling: int,
        T: int,
        config: dict,
        n_classes: int,
        **kwargs,
    ):
        super().__init__()
        self.choice_temperature = choice_temperature
        self.T = T
        self.config = config
        self.n_classes = n_classes

        self.mask_token_ids = config["VQVAE"]["codebook"]["size"]
        self.gamma = self.gamma_func("cosine")
        dataset_name = config["dataset"]["dataset_name"]

        # define encoder, decoder, vq_models
        dim = config["encoder"]["dim"]
        in_channels = config["dataset"]["in_channels"]
        downsampled_width = config["encoder"]["downsampled_width"]
        self.n_fft = config["VQVAE"]["n_fft"]
        downsample_rate = compute_downsample_rate(
            input_length, self.n_fft, downsampled_width
        )

        self.encoder = VQVAEEncoder(
            dim, 2 * in_channels, downsample_rate, config["encoder"]["n_resnet_blocks"]
        )
        self.decoder = VQVAEDecoder(
            dim, 2 * in_channels, downsample_rate, config["decoder"]["n_resnet_blocks"]
        )
        self.vq_model = VectorQuantize(
            dim, config["VQVAE"]["codebook"]["size"], **config["VQVAE"]
        )
        # load trained models for encoder, decoder, and vq_models
        stage1_ssl_method = config["SSL"]["stage1_method"]
        self.load(
            self.encoder,
            get_root_dir().joinpath("saved_models"),
            f"{ssl_config_filename(config, 'encoder')}-{dataset_name}.ckpt",
        )
        print(f"{stage1_ssl_method} encoder loaded")
        self.load(
            self.decoder,
            get_root_dir().joinpath("saved_models"),
            f"{ssl_config_filename(config, 'decoder')}-{dataset_name}.ckpt",
        )
        print(f"{stage1_ssl_method} decoder loaded")
        self.load(
            self.vq_model,
            get_root_dir().joinpath("saved_models"),
            f"{ssl_config_filename(config, 'vqmodel')}-{dataset_name}.ckpt",
        )
        print(f"{stage1_ssl_method} vqmodel loaded")

        # freeze the models for encoder, decoder, and vq_model
        freeze(self.encoder)
        freeze(self.decoder)
        freeze(self.vq_model)

        # evaluation model for encoder, decoder, and vq_model
        self.encoder.eval()
        self.decoder.eval()
        self.vq_model.eval()

        # token lengths
        self.num_tokens = self.encoder.num_tokens.item()

        # latent space dim
        self.H_prime = self.encoder.H_prime
        self.W_prime = self.encoder.W_prime

        # pretrained discrete tokens
        embed = nn.Parameter(copy.deepcopy(self.vq_model._codebook.embed))

        # Encoder Decoder Bidirectional Transformer
        self.autoencoder_transformer = AutoEncoderTransformer(
            self.num_tokens,
            config["VQVAE"]["codebook"]["size"],
            config["VQVAE"]["codebook"]["dim"],
            **config["MAGE"]["prior_model"],
            n_classes=n_classes,
            pretrained_tok_emb=embed,
        )

        # stochastic codebook sampling
        self.vq_model._codebook.sample_codebook_temp = stochastic_sampling

    def load(self, model, dirname, fname):
        """
        model: instance
        path_to_saved_model_fname: path to the ckpt file (i.e., trained model)
        """
        print(dirname)
        print(fname)
        try:
            model.load_state_dict(torch.load(dirname.joinpath(fname)))
        except FileNotFoundError:
            dirname = Path(tempfile.gettempdir())
            model.load_state_dict(torch.load(dirname.joinpath(fname)))

    @torch.no_grad()
    def encode_to_z_q(self, x, encoder: VQVAEEncoder, vq_model: VectorQuantize):
        """
        x: (B, C, L)
        """
        C = x.shape[1]
        xf = time_to_timefreq(x, self.n_fft, C)  # (B, C, H, W)
        z = encoder(xf)  # (b c h w)
        z_q, indices, vq_loss, perplexity = quantize(
            z, vq_model
        )  # (b c h w), (b (h w) h), ...
        return z_q, indices

    def forward(self, x, y, return_summaries: bool = False):
        """
        x: (B, C, L)
        y: (B, 1)
        """
        device = x.device
        _, s = self.encode_to_z_q(x, self.encoder, self.vq_model)  # (b n)

        # --- Creating Masked Tokens ---
        # randomly sample two 't' values
        t1 = np.random.uniform(0, 1)
        t2 = np.random.uniform(0, 1)

        n_masks1 = math.floor(self.gamma(t1) * s.shape[1])
        n_masks2 = math.floor(self.gamma(t2) * s.shape[1])

        rand1 = torch.rand(s.shape, device=device)  # (b n)
        rand2 = torch.rand(s.shape, device=device)  # (b n)

        mask1 = torch.zeros(s.shape, dtype=torch.bool, device=device)
        mask2 = torch.zeros(s.shape, dtype=torch.bool, device=device)

        mask1.scatter_(dim=1, index=rand1.topk(n_masks1, dim=1).indices, value=True)
        mask2.scatter_(dim=1, index=rand2.topk(n_masks2, dim=1).indices, value=True)

        # --- Encode-Decode transformers ---
        logits1, summary1 = self.autoencoder_transformer(s, y, masks=mask1)
        logits2, summary2 = self.autoencoder_transformer(s, y, masks=mask2)

        logits = [logits1, logits2]
        summaries = [summary1, summary2]
        target = s

        if return_summaries:
            return logits, summaries, target
        else:
            return logits, target

    @torch.no_grad()
    def summarize(self, x):
        _, s = self.encode_to_z_q(x, self.encoder, self.vq_model)
        summary = self.autoencoder_transformer.summarize(s)
        return summary

    @torch.no_grad()
    def summarize_dataloader(self, data_loader, device):
        summary_data = []
        for batch in data_loader:
            x, _ = batch[0].to(device), batch[1].to(device)
            summary = self.summarize(x)
            for s in summary:
                summary_data.append(s.tolist())
        return np.array(summary_data)

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r**2
        elif mode == "cubic":
            return lambda r: 1 - r**3
        else:
            raise NotImplementedError

    def create_input_tokens_normal(self, num, num_tokens, mask_token_ids, device):
        # Initialize blank tokens and create masked tokens by multiplying with mask_token_ids
        blank_tokens = torch.ones((num, num_tokens), device=device)
        masked_tokens = mask_token_ids * blank_tokens

        # Create a mask with all True values, indicating all positions are masked
        mask = torch.ones((num, num_tokens), dtype=torch.bool, device=device)

        return masked_tokens.to(torch.int64), mask

    def mask_by_random_topk(self, mask_len, probs, temperature=1.0, device="cpu"):
        """
        mask_len: (b 1)
        probs: (b n); also for the confidence scores

        This version keeps `mask_len` exactly.
        """

        def log(t, eps=1e-20):
            return torch.log(t.clamp(min=eps))

        def gumbel_noise(t):
            """
            Gumbel max trick: https://neptune.ai/blog/gumbel-softmax-loss-function-guide-how-to-implement-it-in-pytorch
            """
            noise = torch.zeros_like(t).uniform_(0, 1)
            return -log(-log(noise))

        confidence = torch.log(probs + 1e-5) + temperature * gumbel_noise(probs).to(
            device
        )  # Gumbel max trick; 1e-5 for numerical stability; (b n)
        mask_len_unique = int(mask_len.unique().item())
        masking_ind = torch.topk(
            confidence, k=mask_len_unique, dim=-1, largest=False
        ).indices  # (b k)
        masking = torch.zeros_like(confidence).to(device)  # (b n)
        for i in range(masking_ind.shape[0]):
            masking[i, masking_ind[i].long()] = 1.0
        masking = masking.bool()
        return masking

    def sample(
        self,
        s: torch.Tensor,
        unknown_number_in_the_beginning,
        class_condition: Union[torch.Tensor, None],
        init_masking,
        guidance_scale: float,
        gamma: Callable,
        device,
    ):
        masking = init_masking

        for t in range(self.T):
            logits, _ = self.autoencoder_transformer(
                embed_ind=s,
                class_condition=class_condition,
                masks=masking,
            )  # (b n codebook_size) == (b n K)
            if isinstance(class_condition, torch.Tensor):
                logits_null, _ = self.autoencoder_transformer(
                    embed_ind=s, class_condition=None, masks=masking
                )
                logits = logits_null + guidance_scale * (logits - logits_null)

            sampled_ids = torch.distributions.categorical.Categorical(
                logits=logits
            ).sample()  # (b n)
            unknown_map = (
                s == self.mask_token_ids
            )  # which tokens need to be sampled; (b n)
            sampled_ids = torch.where(
                unknown_map, sampled_ids, s
            )  # keep the previously-sampled tokens; (b n)

            # create masking according to `t`
            ratio = 1.0 * (t + 1) / self.T  # just a percentage e.g. 1 / 12
            mask_ratio = gamma(ratio)

            probs = F.softmax(logits, dim=-1)  # convert logits into probs; (b n K)
            selected_probs = torch.gather(
                probs, dim=-1, index=sampled_ids.unsqueeze(-1)
            ).squeeze()  # get probability for the selected tokens; p(\hat{s}(t) | \hat{s}_M(t)); (b n)

            _CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to(device)
            selected_probs = torch.where(
                unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS
            )  # assign inf probability to the previously-selected tokens; (b n)

            mask_len = torch.unsqueeze(
                torch.floor(unknown_number_in_the_beginning * mask_ratio), 1
            )  # number of tokens that are to be masked;  (b,)
            mask_len = torch.clip(
                mask_len, min=0.0
            )  # `mask_len` should be equal or larger than zero.

            # Adds noise for randomness
            masking = self.mask_by_random_topk(
                mask_len,
                selected_probs,
                temperature=self.choice_temperature * (1.0 - ratio),
                device=device,
            )

            # Masks tokens with lower confidence.
            s = torch.where(masking, self.mask_token_ids, sampled_ids)  # (b n)

        return s

    @torch.no_grad()
    def iterative_decoding(
        self,
        num=1,
        mode="cosine",
        class_index=None,
        device="cpu",
        guidance_scale: float = 1.0,
    ):
        """
        It performs the iterative decoding and samples token indices.
        :param num: number of samples
        :return: sampled token indices
        """
        s, init_masks = self.create_input_tokens_normal(
            num, self.num_tokens, self.mask_token_ids, device
        )  # (b n)

        unknown_number_in_the_beginning = torch.sum(
            s == self.mask_token_ids, dim=-1
        )  # (b,)

        gamma = self.gamma_func(mode)
        class_condition = (
            repeat(torch.Tensor([class_index]).int().to(device), "i -> b i", b=num)
            if class_index != None
            else None
        )  # (b 1)

        s = self.sample(
            s,
            unknown_number_in_the_beginning,
            class_condition,
            init_masks,
            guidance_scale,
            gamma,
            device,
        )

        return s

    def decode_token_ind_to_timeseries(
        self, s: torch.Tensor, return_representations: bool = False
    ):
        #
        # It takes token embedding indices and decodes them to time series.
        #:param s: token embedding index
        #:param return_representations:
        #:return:
        #

        vq_model = self.vq_model
        decoder = self.decoder

        quantize = F.embedding(s, vq_model._codebook.embed)  # (b n d)
        quantize = vq_model.project_out(quantize)  # (b n c)

        quantize = rearrange(quantize, "b n c -> b c n")  # (b c n) == (b c (h w))

        # print("quantize.shape before reshaping:", quantize.shape)
        # print(self.H_prime, self.W_prime)

        quantize = rearrange(
            quantize, "b c (h w) -> b c h w", h=self.H_prime, w=self.W_prime
        )

        uhat = decoder(quantize)

        xhat = timefreq_to_time(
            uhat, self.n_fft, self.config["dataset"]["in_channels"]
        )  # (B, C, L)

        if return_representations:
            return xhat, quantize
        else:
            return xhat

    def critical_reverse_sampling(
        self,
        s: torch.Tensor,
        unknown_number_in_the_beginning,
        class_condition: Union[torch.Tensor, None],
    ):
        """
        s: sampled token sequence from the naive iterative decoding.
        """

        mask_token_ids = self.mask_token_ids
        transformer = self.integrated_transformer
        vq_model = self.vq_model

        # compute the confidence scores for s_T
        # the scores are used for the step retraction by iteratively removing unrealistic tokens.
        confidence_scores = self.compute_confidence_score(
            s, mask_token_ids, vq_model, transformer, class_condition
        )  # (b n)

        # find s_{t*}
        # t* denotes the step where unrealistic tokens have been removed.
        t_star = 1
        s_star = None
        prev_error = None
        error_ratio_hist = deque(
            maxlen=round(self.T * self.config["MaskGIT"]["ESS"]["error_ratio_ma_rate"])
        )
        for t in range(1, self.T)[::-1]:
            # masking ratio according to the masking scheduler
            ratio_t = 1.0 * (t + 1) / self.T  # just a percentage e.g. 1 / 12
            ratio_tm1 = 1.0 * t / self.T  # tm1: t - 1
            mask_ratio_t = self.gamma(ratio_t)
            mask_ratio_tm1 = self.gamma(ratio_tm1)  # tm1: t - 1

            # mask length
            mask_len_t = torch.unsqueeze(
                torch.floor(unknown_number_in_the_beginning * mask_ratio_t), 1
            )
            mask_len_tm1 = torch.unsqueeze(
                torch.floor(unknown_number_in_the_beginning * mask_ratio_tm1), 1
            )

            # masking matrices: {True: masking, False: not-masking}
            masking_t = self.mask_by_random_topk(
                mask_len_t, confidence_scores, temperature=0.0, device=s.device
            )  # (b n)
            masking_tm1 = self.mask_by_random_topk(
                mask_len_tm1, confidence_scores, temperature=0.0, device=s.device
            )  # (b n)
            masking = ~(
                (masking_tm1.float() - masking_t.float()).bool()
            )  # (b n); True for everything except the area of interest with False.

            # if there's no difference between t-1 and t, ends the retraction.
            if masking_t.float().sum() == masking_tm1.float().sum():
                t_star = t
                s_star = torch.where(masking_t, mask_token_ids, s)  # (b n)
                print("no difference between t-1 and t.")
                break

            # predict s_t given s_{t-1}
            s_tm1 = torch.where(masking_tm1, mask_token_ids, s)  # (b n)
            logits = transformer(s_tm1, class_condition=class_condition)  # (b n K)
            s_t_hat = logits.argmax(dim=-1)  # (b n)

            # leave the tokens of interest -- i.e., ds/dt -- only at t
            s_t = torch.where(masking, mask_token_ids, s)  # (b n)
            s_t_hat = torch.where(masking, mask_token_ids, s_t_hat)  # (b n)

            # measure error: distance between z_q_t and z_q_t_hat
            z_q_t = F.embedding(s_t[~masking], vq_model._codebook.embed)  # (b n d)
            z_q_t_hat = F.embedding(
                s_t_hat[~masking], vq_model._codebook.embed
            )  # (b n d)
            error = ((z_q_t - z_q_t_hat) ** 2).mean().cpu().detach().item()

            # error ratio
            if t + 1 == self.T:
                error_ratio_ma = 0.0
                prev_error = error
            else:
                error_ratio = error / (prev_error + 1e-5)
                error_ratio_hist.append(error_ratio)
                error_ratio_ma = np.mean(error_ratio_hist)
                print(
                    f"t:{t} | error:{round(error, 6)} | error_ratio_ma:{round(error_ratio_ma, 6)}"
                )
                prev_error = error

            # stopping criteria
            stopping_threshold = 1.0
            if error_ratio_ma > stopping_threshold and (t + 1 != self.T):
                t_star = t
                s_star = torch.where(masking_t, mask_token_ids, s)  # (b n)
                print("stopped by `error_ratio_ma > threshold`.")
                break
            if t == 1:
                t_star = t
                s_star = torch.where(masking_t, mask_token_ids, s)  # (b n)
                print("t_star has reached t=1.")
                break
        print("t_star:", t_star)
        return t_star, s_star

    def iterative_decoding_with_self_token_critic(
        self,
        t_star,
        s_star,
        unknown_number_in_the_beginning,
        class_condition: Union[torch.Tensor, None],
        guidance_scale: float,
        device,
    ):
        mask_token_ids = self.mask_token_ids
        transformer = self.integrated_transformer
        vq_model = self.vq_model
        choice_temperature = self.choice_temperature

        s = s_star
        for t in range(t_star, self.T):
            logits = transformer(
                s, class_condition=class_condition
            )  # (b n codebook_size) == (b n K)
            if isinstance(class_condition, torch.Tensor):
                logits_null = transformer(s, class_condition=None)
                logits = logits_null + guidance_scale * (logits - logits_null)
            sampled_ids = torch.distributions.categorical.Categorical(
                logits=logits
            ).sample()  # (b n)

            # create masking according to `t`
            ratio = 1.0 * (t + 1) / self.T  # just a percentage e.g. 1 / 12
            mask_ratio = self.gamma(ratio)

            # compute the confidence scores for s_t
            confidence_scores = self.compute_confidence_score(
                sampled_ids, mask_token_ids, vq_model, transformer, class_condition
            )  # (b n)

            mask_len = torch.unsqueeze(
                torch.floor(unknown_number_in_the_beginning * mask_ratio), 1
            )  # number of tokens that are to be masked;  (b,)
            mask_len = torch.clip(
                mask_len, min=0.0
            )  # `mask_len` should be equal or larger than zero.

            # Adds noise for randomness
            masking = self.mask_by_random_topk(
                mask_len,
                confidence_scores,
                temperature=choice_temperature * (1.0 - ratio),
                device=device,
            )

            # Masks tokens with lower confidence.
            s = torch.where(masking, mask_token_ids, sampled_ids)  # (b n)
        return s

    def compute_confidence_score(
        self, s, mask_token_ids, vq_model, transformer, class_condition
    ):
        confidence_scores = torch.zeros_like(s).float()  # (b n)
        for n in range(confidence_scores.shape[-1]):
            s_m = copy.deepcopy(s)  # (b n)
            s_m[:, n] = (
                mask_token_ids  # (b n); masking the n-th token to measure the confidence score for that token.
            )
            logits = transformer(s_m, class_condition=class_condition)  # (b n K)
            logits = torch.nn.functional.softmax(logits, dim=-1)  # (b n K)

            true_tokens = s[:, n]  # (b,)
            logits = logits[:, n]  # (b, K)
            pred_tokens = logits.argmax(dim=-1)  # (b,)

            z_q_true = vq_model._codebook.embed[true_tokens]  # (b, dim)
            z_q_pred = vq_model._codebook.embed[pred_tokens]  # (b, dim)
            dist = torch.sum((z_q_true - z_q_pred) ** 2, dim=-1)  # (b,)
            confidence_scores[:, n] = -1 * dist  # confidence score for the n-th token
        confidence_scores = torch.nn.functional.softmax(
            confidence_scores, dim=-1
        )  # (b n)
        return confidence_scores
