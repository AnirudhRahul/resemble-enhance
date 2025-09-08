from ...hparams import HParams
from .base import Chain, Choice, Permutation
from .custom import RandomGaussianNoise, RandomRIR, RandomWhamNoise


class Distorter(Chain):
    def __init__(self, hp: HParams, training: bool = False, mode: str = "enhancer"):
        # Lazy import
        from .sox import RandomBandpassDistorter, RandomEqualizer, RandomLowpassDistorter, RandomOverdrive, RandomReverb

        if training:
            effects = []
            if hp.enable_rir:
                effects.append(RandomRIR(hp.rir_dir))
            if hp.enable_reverb:
                effects.append(RandomReverb())
            if hp.enable_gaussian_noise:
                effects.append(RandomGaussianNoise())
            if hp.enable_wham_noise:
                effects.append(RandomWhamNoise(hp.wham_noise_dir))
            if hp.enable_overdrive:
                effects.append(RandomOverdrive())
            if hp.enable_equalizer:
                effects.append(RandomEqualizer())
            if hp.enable_filters:
                effects.append(
                    Choice(
                        RandomLowpassDistorter(),
                        RandomBandpassDistorter(),
                    )
                )
            permutation = Permutation(*effects)
            if mode == "denoiser":
                super().__init__(permutation)
            else:
                # 80%: distortion, 20%: clean
                super().__init__(Choice(permutation, Chain(), p=[0.8, 0.2]))
        else:
            effects = []
            if hp.enable_rir:
                effects.append(RandomRIR(hp.rir_dir, deterministic=True))
            if hp.enable_reverb:
                effects.append(RandomReverb(deterministic=True))
            super().__init__(*effects)
