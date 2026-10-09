import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ============================================================
# SPECTRAL-PHASE FEATURE EXTRACTOR
# ============================================================

class SpectralPhase(nn.Module):
    """
    Spectral feature extractor untuk:

        Tahap 4:
            Channel 0 = Log-Mel
            Channel 1 = Phase-derived

        Tahap 5:
            Modified Group Delay (MGD)

    Default configuration:
        sample rate = 16000 Hz
        FFT         = 512
        hop         = 160
        Mel bins    = 80

    Tahap 4 forward():
        input  : [B,1,T]
        output : [B,2,80,frames]

    MGD:
        modified_group_delay()
        output : [B,80,frames]

    IMPORTANT:
        MGD belum dimasukkan ke forward().
        Hal ini disengaja agar Tahap 4 tetap frozen
        sampai MGD lolos numerical QC.
    """

    def __init__(
        self,
        sr=16000,
        n_fft=512,
        hop=160,
        n_mels=80,
        eps=1e-6,
    ):
        super().__init__()

        # ====================================================
        # BASIC CONFIG
        # ====================================================

        self.sr = int(sr)

        self.n_fft = int(n_fft)

        self.hop = int(hop)

        self.n_mels = int(n_mels)

        self.eps = float(eps)

        # ====================================================
        # MEL FILTERBANK
        # ====================================================
        #
        # torchaudio returns:
        #
        #     [freq, mel]
        #
        # Kita transpose menjadi:
        #
        #     [mel, freq]
        #
        # agar bisa:
        #
        # einsum("mf,bft->bmt")
        #
        # ====================================================

        fb = (
            torchaudio.functional
            .melscale_fbanks(
                n_freqs=(
                    self.n_fft
                    // 2
                    + 1
                ),

                f_min=0.0,

                f_max=(
                    self.sr
                    / 2.0
                ),

                n_mels=self.n_mels,

                sample_rate=self.sr,

                norm="slaney",

                mel_scale="slaney",
            )
            .T
            .contiguous()
        )

        self.register_buffer(
            "fb",
            fb,
        )

        # ====================================================
        # HANN WINDOW
        # ====================================================

        window = torch.hann_window(
            self.n_fft
        )

        self.register_buffer(
            "window",
            window,
        )

        # ====================================================
        # OPTIONAL NORMALIZATION
        # ====================================================
        #
        # Expected mean/std shape:
        #
        # [1,2,80,1]
        #
        # Default:
        # normalization OFF.
        #
        # ====================================================

        feature_mean = torch.zeros(
            1,
            2,
            self.n_mels,
            1,
            dtype=torch.float32,
        )

        feature_std = torch.ones(
            1,
            2,
            self.n_mels,
            1,
            dtype=torch.float32,
        )

        self.register_buffer(
            "feature_mean",
            feature_mean,
        )

        self.register_buffer(
            "feature_std",
            feature_std,
        )

        self.normalization_enabled = False


    # ========================================================
    # PREPARE WAVEFORM
    # ========================================================

    def _prepare_waveform(
        self,
        wav,
    ):
        """
        Accepted input:

            [T]
            [B,T]
            [B,1,T]

        Output:

            [B,1,T]

        Audio dipastikan:
            float
            finite
            minimum panjang n_fft
        """

        # ====================================================
        # CONVERT TO TENSOR
        # ====================================================

        if not torch.is_tensor(
            wav
        ):

            wav = torch.as_tensor(
                wav
            )

        # ====================================================
        # DIMENSION NORMALIZATION
        # ====================================================

        if wav.ndim == 1:

            # [T]
            # ->
            # [1,1,T]

            wav = (
                wav
                .unsqueeze(0)
                .unsqueeze(0)
            )

        elif wav.ndim == 2:

            # [B,T]
            # ->
            # [B,1,T]

            wav = (
                wav
                .unsqueeze(1)
            )

        elif wav.ndim == 3:

            # Expected:
            # [B,1,T]

            if (
                wav.shape[1]
                != 1
            ):

                raise ValueError(
                    "Expected mono waveform "
                    "with shape [B,1,T], "
                    f"got {tuple(wav.shape)}"
                )

        else:

            raise ValueError(
                "Waveform must have shape "
                "[T], [B,T], or [B,1,T]. "
                f"Got {tuple(wav.shape)}"
            )

        # ====================================================
        # FLOAT
        # ====================================================

        if not wav.is_floating_point():

            wav = wav.float()

        else:

            wav = wav.to(
                dtype=torch.float32
            )

        # ====================================================
        # FINITE CHECK
        # ====================================================

        if not torch.isfinite(
            wav
        ).all():

            raise ValueError(
                "Input waveform contains "
                "NaN or Inf."
            )

        # ====================================================
        # MINIMUM LENGTH
        # ====================================================

        num_samples = (
            wav.shape[
                -1
            ]
        )

        if (
            num_samples
            < self.n_fft
        ):

            pad_amount = (
                self.n_fft
                - num_samples
            )

            wav = F.pad(
                wav,
                (
                    0,
                    pad_amount,
                ),
            )

        return wav


    # ========================================================
    # PHASE UNWRAP
    # ========================================================

    @staticmethod
    def unwrap_phase(
        phase,
        dim=-1,
    ):
        """
        Phase unwrap manual menggunakan
        discontinuity correction ±pi.

        Input:
            wrapped phase

        Output:
            unwrapped phase

        Operasi dilakukan sepanjang
        dimensi waktu.
        """

        if (
            phase.shape[
                dim
            ]
            <= 1
        ):

            return phase

        # ====================================================
        # FIRST DIFFERENCE
        # ====================================================

        diff = torch.diff(
            phase,
            dim=dim,
        )

        two_pi = (
            2.0
            * math.pi
        )

        # ====================================================
        # MAP DIFFERENCE TO [-pi, pi)
        # ====================================================

        diff_mod = (
            torch.remainder(
                diff
                + math.pi,

                two_pi,
            )
            - math.pi
        )

        # Special handling agar +pi
        # tidak selalu dipetakan ke -pi.

        diff_mod = torch.where(
            (
                diff_mod
                == -math.pi
            )
            &
            (
                diff
                > 0
            ),

            torch.full_like(
                diff_mod,
                math.pi,
            ),

            diff_mod,
        )

        # ====================================================
        # CORRECTION
        # ====================================================

        correction = (
            diff_mod
            - diff
        )

        correction = torch.where(
            diff.abs()
            < math.pi,

            torch.zeros_like(
                correction
            ),

            correction,
        )

        # ====================================================
        # CUMULATIVE CORRECTION
        # ====================================================

        correction = torch.cumsum(
            correction,
            dim=dim,
        )

        # ====================================================
        # PREPEND ZERO
        # ====================================================

        first_shape = list(
            phase.shape
        )

        first_shape[
            dim
        ] = 1

        zero = torch.zeros(
            first_shape,

            dtype=phase.dtype,

            device=phase.device,
        )

        correction = torch.cat(
            [
                zero,
                correction,
            ],
            dim=dim,
        )

        return (
            phase
            + correction
        )


    # ========================================================
    # SHARED STFT
    # ========================================================

    def _shared_stft(
        self,
        wav,
    ):
        """
        Compute STFT sekali untuk
        Log-Mel dan phase-derived.

        Input:
            [B,1,T]

        Output:
            complex tensor
            [B,F,T_frames]
        """

        wav = (
            self._prepare_waveform(
                wav
            )
        )

        signal = (
            wav.squeeze(
                1
            )
        )

        window = self.window.to(
            device=signal.device,
            dtype=signal.dtype,
        )

        X = torch.stft(
            signal,

            n_fft=self.n_fft,

            hop_length=self.hop,

            win_length=self.n_fft,

            window=window,

            center=False,

            return_complex=True,
        )

        if not torch.isfinite(
            X.real
        ).all():

            raise RuntimeError(
                "STFT real part contains "
                "NaN/Inf."
            )

        if not torch.isfinite(
            X.imag
        ).all():

            raise RuntimeError(
                "STFT imaginary part contains "
                "NaN/Inf."
            )

        return X


    # ========================================================
    # LOG-MEL
    # ========================================================

    def _compute_logmel(
        self,
        X,
    ):
        """
        Magnitude STFT -> Mel -> log.

        Output:
            [B,80,T]
        """

        magnitude = (
            X.abs()
            .clamp_min(
                self.eps
            )
        )

        fb = self.fb.to(
            device=magnitude.device,
            dtype=magnitude.dtype,
        )

        mel = torch.einsum(
            "mf,bft->bmt",

            fb,

            magnitude,
        )

        logmel = torch.log(
            mel
            + self.eps
        )

        if not torch.isfinite(
            logmel
        ).all():

            raise RuntimeError(
                "Log-Mel contains NaN/Inf."
            )

        return logmel


    # ========================================================
    # PHASE-DERIVED
    # ========================================================

    def _compute_phase_derived(
        self,
        X,
    ):
        """
        Phase-derived representation:

            angle(X)
               ↓
            unwrap along time
               ↓
            temporal difference
               ↓
            absolute value
               ↓
            same Mel filterbank

        Output:
            [B,80,T]
        """

        # ====================================================
        # WRAPPED PHASE
        # ====================================================

        phase = torch.angle(
            X
        )

        # ====================================================
        # UNWRAP TIME
        # ====================================================

        unwrapped_phase = (
            self.unwrap_phase(
                phase,
                dim=-1,
            )
        )

        # ====================================================
        # TEMPORAL PHASE DIFFERENCE
        # ====================================================
        #
        # prepend first frame menyebabkan
        # temporal difference frame pertama = 0.
        #
        # ====================================================

        dphase = torch.diff(
            unwrapped_phase,

            dim=-1,

            prepend=(
                unwrapped_phase[
                    ...,
                    :1
                ]
            ),
        )

        dphase = (
            dphase
            .abs()
        )

        # ====================================================
        # MEL PROJECTION
        # ====================================================

        fb = self.fb.to(
            device=dphase.device,
            dtype=dphase.dtype,
        )

        phase_mel = torch.einsum(
            "mf,bft->bmt",

            fb,

            dphase,
        )

        if not torch.isfinite(
            phase_mel
        ).all():

            raise RuntimeError(
                "Phase-derived map contains "
                "NaN/Inf."
            )

        return phase_mel


    # ========================================================
    # MODIFIED GROUP DELAY
    # ========================================================

    def modified_group_delay(
        self,
        wav,
        alpha=0.4,
        gamma=0.9,
    ):
        """
        Modified Group Delay (MGD) starter.

        Tahap 5.

        Input:
            wav:
                [B,1,T]

        Output:
            mgd_mel:
                [B,80,T_frames]

        Pipeline:

            wav
             |
             +--> STFT X
             |
             +--> n*x[n]
                    |
                    +--> STFT Y

            X,Y
             |
             +--> group-delay numerator

            |X|
             |
             +--> frequency smoothing
             |
             +--> denominator

            numerator / denominator
             |
             +--> alpha compression
             |
             +--> absolute MGD
             |
             +--> Mel projection

        Notes:
            alpha default = 0.4
            gamma default = 0.9

        Formula ini adalah starter sesuai
        panduan dan nantinya tetap perlu
        literature validation / ablation.
        """

        # ====================================================
        # PARAMETER VALIDATION
        # ====================================================

        if alpha <= 0:

            raise ValueError(
                "alpha must be > 0."
            )

        if gamma <= 0:

            raise ValueError(
                "gamma must be > 0."
            )

        # ====================================================
        # PREPARE AUDIO
        # ====================================================

        wav = (
            self._prepare_waveform(
                wav
            )
        )

        signal = (
            wav.squeeze(
                1
            )
        )

        num_samples = int(
            signal.shape[
                -1
            ]
        )

        # ====================================================
        # WINDOW
        # ====================================================

        window = self.window.to(
            device=signal.device,
            dtype=signal.dtype,
        )

        # ====================================================
        # X = STFT{x[n]}
        # ====================================================

        X = torch.stft(
            signal,

            n_fft=self.n_fft,

            hop_length=self.hop,

            win_length=self.n_fft,

            window=window,

            center=False,

            return_complex=True,
        )

        # ====================================================
        # n VECTOR
        # ====================================================

        n = torch.arange(
            num_samples,

            device=signal.device,

            dtype=signal.dtype,
        ).unsqueeze(
            0
        )

        # ====================================================
        # n*x[n]
        # ====================================================

        weighted_signal = (
            signal
            * n
        )

        # ====================================================
        # Y = STFT{n*x[n]}
        # ====================================================

        Y = torch.stft(
            weighted_signal,

            n_fft=self.n_fft,

            hop_length=self.hop,

            win_length=self.n_fft,

            window=window,

            center=False,

            return_complex=True,
        )

        # ====================================================
        # SHAPE CHECK
        # ====================================================

        if (
            X.shape
            != Y.shape
        ):

            raise RuntimeError(
                "X/Y STFT shape mismatch: "
                f"X={tuple(X.shape)}, "
                f"Y={tuple(Y.shape)}"
            )

        # ====================================================
        # FINITE STFT CHECK
        # ====================================================

        if not torch.isfinite(
            X.real
        ).all():

            raise RuntimeError(
                "MGD X.real contains NaN/Inf."
            )

        if not torch.isfinite(
            X.imag
        ).all():

            raise RuntimeError(
                "MGD X.imag contains NaN/Inf."
            )

        if not torch.isfinite(
            Y.real
        ).all():

            raise RuntimeError(
                "MGD Y.real contains NaN/Inf."
            )

        if not torch.isfinite(
            Y.imag
        ).all():

            raise RuntimeError(
                "MGD Y.imag contains NaN/Inf."
            )

        # ====================================================
        # GROUP DELAY NUMERATOR
        # ====================================================
        #
        # Starter formula dari panduan:
        #
        # numerator =
        #     X.real * Y.real
        #     +
        #     X.imag * Y.imag
        #
        # ====================================================

        numerator = (
            X.real
            * Y.real
            +
            X.imag
            * Y.imag
        )

        # ====================================================
        # MAGNITUDE
        # ====================================================

        magnitude = (
            X.abs()
            .clamp_min(
                self.eps
            )
        )

        # ====================================================
        # SMOOTHED MAGNITUDE
        # ====================================================
        #
        # Smooth sepanjang frequency axis.
        #
        # magnitude:
        #     [B,F,T]
        #
        # unsqueeze:
        #     [B,1,F,T]
        #
        # avg_pool:
        #     kernel=(5,1)
        #
        # sehingga smoothing dilakukan pada
        # frequency bins, tidak pada waktu.
        #
        # ====================================================

        smooth = F.avg_pool2d(
            magnitude.unsqueeze(
                1
            ),

            kernel_size=(
                5,
                1,
            ),

            stride=1,

            padding=(
                2,
                0,
            ),
        ).squeeze(
            1
        )

        smooth = (
            smooth
            .clamp_min(
                self.eps
            )
        )

        # ====================================================
        # DENOMINATOR
        # ====================================================

        denominator = (
            smooth.pow(
                2.0
                * gamma
            )
            +
            self.eps
        )

        # ====================================================
        # TAU
        # ====================================================

        tau = (
            numerator
            /
            denominator
        )

        if not torch.isfinite(
            tau
        ).all():

            raise RuntimeError(
                "Raw MGD tau contains "
                "NaN/Inf."
            )

        # ====================================================
        # ALPHA COMPRESSION
        # ====================================================
        #
        # sign(tau) * |tau|^alpha
        #
        # ====================================================

        mgd = (
            tau.sign()
            *
            tau.abs()
            .clamp_min(
                self.eps
            )
            .pow(
                alpha
            )
        )

        if not torch.isfinite(
            mgd
        ).all():

            raise RuntimeError(
                "Compressed MGD contains "
                "NaN/Inf."
            )

        # ====================================================
        # ABSOLUTE MGD
        # ====================================================
        #
        # Panduan meminta absolute MGD
        # diproyeksikan ke Mel.
        #
        # ====================================================

        mgd_abs = (
            mgd.abs()
        )

        # ====================================================
        # MEL PROJECTION
        # ====================================================

        fb = self.fb.to(
            device=mgd_abs.device,
            dtype=mgd_abs.dtype,
        )

        mgd_mel = torch.einsum(
            "mf,bft->bmt",

            fb,

            mgd_abs,
        )

        # ====================================================
        # OUTPUT QC
        # ====================================================

        if (
            mgd_mel.ndim
            != 3
        ):

            raise RuntimeError(
                "MGD output must be 3D, "
                f"got {tuple(mgd_mel.shape)}"
            )

        if (
            mgd_mel.shape[
                1
            ]
            != self.n_mels
        ):

            raise RuntimeError(
                "MGD Mel dimension invalid: "
                f"{tuple(mgd_mel.shape)}"
            )

        if not torch.isfinite(
            mgd_mel
        ).all():

            raise RuntimeError(
                "MGD Mel map contains "
                "NaN/Inf."
            )

        return mgd_mel


    # ========================================================
    # SET NORMALIZATION
    # ========================================================

    def set_normalization(
        self,
        mean,
        std,
    ):
        """
        Mengaktifkan standardisasi Tahap 4.

        Accepted shapes:

            [2,80,1]
            [1,2,80,1]

        Normalization:
            (x - mean) / std
        """

        mean = torch.as_tensor(
            mean,
            dtype=torch.float32,
        )

        std = torch.as_tensor(
            std,
            dtype=torch.float32,
        )

        # ====================================================
        # CONVERT [2,80,1]
        # TO [1,2,80,1]
        # ====================================================

        if mean.ndim == 3:

            mean = (
                mean
                .unsqueeze(
                    0
                )
            )

        if std.ndim == 3:

            std = (
                std
                .unsqueeze(
                    0
                )
            )

        expected_shape = (
            1,
            2,
            self.n_mels,
            1,
        )

        if (
            tuple(
                mean.shape
            )
            != expected_shape
        ):

            raise ValueError(
                "Invalid mean shape. "
                f"Expected {expected_shape}, "
                f"got {tuple(mean.shape)}"
            )

        if (
            tuple(
                std.shape
            )
            != expected_shape
        ):

            raise ValueError(
                "Invalid std shape. "
                f"Expected {expected_shape}, "
                f"got {tuple(std.shape)}"
            )

        if not torch.isfinite(
            mean
        ).all():

            raise ValueError(
                "Mean contains NaN/Inf."
            )

        if not torch.isfinite(
            std
        ).all():

            raise ValueError(
                "Std contains NaN/Inf."
            )

        if torch.any(
            std <= 0
        ):

            raise ValueError(
                "All std values must be > 0."
            )

        # ====================================================
        # COPY TO REGISTERED BUFFERS
        # ====================================================

        self.feature_mean.copy_(
            mean.to(
                self.feature_mean.device
            )
        )

        self.feature_std.copy_(
            std.to(
                self.feature_std.device
            )
        )

        self.normalization_enabled = True


    # ========================================================
    # DISABLE NORMALIZATION
    # ========================================================

    def disable_normalization(
        self,
    ):
        """
        Menonaktifkan normalization.
        """

        self.normalization_enabled = False


    # ========================================================
    # NORMALIZE FEATURE
    # ========================================================

    def _normalize(
        self,
        feature,
    ):
        """
        Standardize feature Tahap 4.

        Input:
            [B,2,80,T]
        """

        if not self.normalization_enabled:

            return feature

        mean = self.feature_mean.to(
            device=feature.device,
            dtype=feature.dtype,
        )

        std = self.feature_std.to(
            device=feature.device,
            dtype=feature.dtype,
        )

        normalized = (
            feature
            - mean
        ) / std

        if not torch.isfinite(
            normalized
        ).all():

            raise RuntimeError(
                "Normalized feature contains "
                "NaN/Inf."
            )

        return normalized


    # ========================================================
    # FORWARD — TAHAP 4
    # ========================================================

    def forward(
        self,
        wav,
    ):
        """
        Tahap 4 forward.

        Output tetap:

            [B,2,80,T]

        Channel:
            0 = Log-Mel
            1 = Phase-derived

        MGD BELUM dimasukkan di sini.
        """

        # ====================================================
        # PREPARE INPUT
        # ====================================================

        wav = (
            self._prepare_waveform(
                wav
            )
        )

        # ====================================================
        # SHARED STFT
        # ====================================================

        X = self._shared_stft(
            wav
        )

        # ====================================================
        # LOG-MEL
        # ====================================================

        logmel = (
            self._compute_logmel(
                X
            )
        )

        # ====================================================
        # PHASE-DERIVED
        # ====================================================

        phase_mel = (
            self._compute_phase_derived(
                X
            )
        )

        # ====================================================
        # SYNCHRONIZATION CHECK
        # ====================================================

        if (
            logmel.shape
            != phase_mel.shape
        ):

            raise RuntimeError(
                "Log-Mel / Phase map "
                "shape mismatch: "
                f"logmel={tuple(logmel.shape)}, "
                f"phase={tuple(phase_mel.shape)}"
            )

        # ====================================================
        # STACK
        # ====================================================
        #
        # [B,80,T]
        # +
        # [B,80,T]
        #
        # ->
        #
        # [B,2,80,T]
        #
        # ====================================================

        feature = torch.stack(
            [
                logmel,
                phase_mel,
            ],

            dim=1,
        )

        # ====================================================
        # SHAPE CHECK
        # ====================================================

        if (
            feature.ndim
            != 4
        ):

            raise RuntimeError(
                "SpectralPhase output "
                "must be 4D."
            )

        if (
            feature.shape[
                1
            ]
            != 2
        ):

            raise RuntimeError(
                "SpectralPhase output "
                "must contain 2 channels."
            )

        if (
            feature.shape[
                2
            ]
            != self.n_mels
        ):

            raise RuntimeError(
                "SpectralPhase output "
                "must contain "
                f"{self.n_mels} Mel bins."
            )

        # ====================================================
        # OPTIONAL NORMALIZATION
        # ====================================================

        feature = (
            self._normalize(
                feature
            )
        )

        # ====================================================
        # FINITE CHECK
        # ====================================================

        if not torch.isfinite(
            feature
        ).all():

            raise RuntimeError(
                "SpectralPhase output contains "
                "NaN/Inf."
            )

        return feature


# ============================================================
# SELF TEST
# ============================================================

def _self_test():

    print(
        "=" * 75
    )

    print(
        "SPECTRAL-PHASE + MGD SELF TEST"
    )

    print(
        "=" * 75
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = SpectralPhase(
        sr=16000,
        n_fft=512,
        hop=160,
        n_mels=80,
    )

    model.eval()

    # ========================================================
    # RANDOM INPUT
    # ========================================================
    #
    # 4 seconds at 16 kHz:
    #
    # 64000 samples
    #
    # Batch = 2
    #
    # ========================================================

    wav = torch.randn(
        2,
        1,
        64000,
        dtype=torch.float32,
    )

    print()

    print(
        "Input shape:",
        tuple(
            wav.shape
        ),
    )

    # ========================================================
    # TAHAP 4 TEST
    # ========================================================

    with torch.no_grad():

        feature = model(
            wav
        )

    print()

    print(
        "Feature shape:",
        tuple(
            feature.shape
        ),
    )

    print(
        "Finite:",
        bool(
            torch.isfinite(
                feature
            ).all()
        ),
    )

    # ========================================================
    # EXPECTED FRAMES
    # ========================================================

    expected_frames = (
        (
            64000
            - model.n_fft
        )
        // model.hop
        + 1
    )

    expected_feature_shape = (
        2,
        2,
        80,
        expected_frames,
    )

    assert (
        tuple(
            feature.shape
        )
        == expected_feature_shape
    ), (
        "Unexpected Tahap 4 feature shape: "
        f"{tuple(feature.shape)}, "
        f"expected={expected_feature_shape}"
    )

    assert torch.isfinite(
        feature
    ).all()

    print()

    print(
        "TAHAP 4 SELF TEST: PASSED"
    )

    # ========================================================
    # MGD TEST
    # ========================================================

    with torch.no_grad():

        mgd = (
            model.modified_group_delay(
                wav,

                alpha=0.4,

                gamma=0.9,
            )
        )

    print()

    print(
        "MGD shape:",
        tuple(
            mgd.shape
        ),
    )

    print(
        "MGD finite:",
        bool(
            torch.isfinite(
                mgd
            ).all()
        ),
    )

    print(
        "MGD min:",
        float(
            mgd.min()
        ),
    )

    print(
        "MGD max:",
        float(
            mgd.max()
        ),
    )

    print(
        "MGD mean:",
        float(
            mgd.mean()
        ),
    )

    print(
        "MGD std:",
        float(
            mgd.std()
        ),
    )

    # ========================================================
    # MGD SHAPE CHECK
    # ========================================================

    expected_mgd_shape = (
        2,
        80,
        expected_frames,
    )

    assert (
        tuple(
            mgd.shape
        )
        == expected_mgd_shape
    ), (
        "Unexpected MGD shape: "
        f"{tuple(mgd.shape)}, "
        f"expected={expected_mgd_shape}"
    )

    assert torch.isfinite(
        mgd
    ).all()

    assert torch.all(
        mgd >= 0
    )

    print()

    print(
        "MGD SELF TEST: PASSED"
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()

    print(
        "=" * 75
    )

    print(
        "ALL SELF TESTS PASSED"
    )

    print(
        "=" * 75
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    _self_test()