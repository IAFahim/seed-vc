"""
Real-time mic -> converted voice -> headphones, in the terminal.

Headless adaptation of real-time-gui.py: same tiny DiT model, same chunked
inference + SOLA crossfade, same funasr VAD gate — only the Tk window is gone
(system Tk 9 is incompatible with FreeSimpleGUI).

Run from the repo root:
    .venv/bin/python realtime_loop.py                    # voice conversion
    .venv/bin/python realtime_loop.py --mode listen      # hear yourself (no model, lowest latency)
    .venv/bin/python realtime_loop.py --reference examples/reference/teio_0.wav
Ctrl+C (or pkill -f realtime_loop.py) to stop.
"""
import os
import sys
import warnings

warnings.simplefilter("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import importlib.util
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import sounddevice as sd
import torchaudio.transforms as tat

# Import load_models / custom_infer from real-time-gui.py (module name has a dash)
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_here)
_spec = importlib.util.spec_from_file_location("rtgui", os.path.join(_here, "real-time-gui.py"))
rtgui = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rtgui)


def parse_args():
    p = argparse.ArgumentParser(description="Seed-VC real-time mic loop")
    p.add_argument("--mode", choices=["vc", "listen"], default="vc",
                   help="vc = convert your voice to the reference speaker; listen = plain mic loopback")
    p.add_argument("--reference", default=os.path.join(_here, "examples/reference/trump_0.wav"),
                   help="reference wav supplying the target voice")
    p.add_argument("--block-time", type=float, default=0.25, help="inference block size in seconds")
    p.add_argument("--crossfade", type=float, default=0.05, help="crossfade length in seconds")
    p.add_argument("--extra-time-ce", type=float, default=2.5, help="content-encoder left context (s)")
    p.add_argument("--extra-time", type=float, default=0.5, help="DiT left context (s)")
    p.add_argument("--extra-time-right", type=float, default=0.02, help="right lookahead (s); raise for quality, latency grows")
    p.add_argument("--diffusion-steps", type=int, default=10)
    p.add_argument("--cfg-rate", type=float, default=0.7)
    p.add_argument("--max-prompt-length", type=float, default=3.0)
    p.add_argument("--input-device", default=None, help="input device name/index substring; default = system default")
    p.add_argument("--output-device", default=None, help="output device name/index substring; default = system default")
    p.add_argument("--fp16", default=True, type=lambda v: str(v).lower() in ("1", "true", "yes"))
    p.add_argument("--checkpoint-path", default=None,
                   help="custom first-stage DiT checkpoint (default: realtime tiny model)")
    p.add_argument("--config-path", default=None, help="config yml matching --checkpoint-path")
    p.add_argument("--sr-device", action="store_true",
                   help="run the audio stream at the device's native rate instead of the model rate "
                        "(avoids PipeWire resampling; output is resampled back to the model rate internally)")
    return p.parse_args()


def resolve_device(name, is_input):
    """Return a PortAudio device index for a name substring, or None for default."""
    if name is None or str(name).lower() in ("default", "none", ""):
        return None
    devices = sd.query_devices()
    try:  # plain integer index
        idx = int(name)
        if 0 <= idx < len(devices):
            return idx
    except ValueError:
        pass
    cand = [i for i, d in enumerate(devices)
            if name.lower() in d["name"].lower()
            and d["max_input_channels" if is_input else "max_output_channels"] > 0]
    if not cand:
        raise SystemExit(f"no {'input' if is_input else 'output'} device matching {name!r}; "
                         f"available: {[d['name'] for d in devices]}")
    return cand[0]


class RealtimeLoop:
    def __init__(self, args, model_set, vad, device):
        self.args = args
        self.model_set = model_set
        self.vad = vad
        self.device = device
        self.mode = args.mode

        self.sr_model = model_set[-1]["sampling_rate"]
        if args.sr_device:
            self.samplerate = int(sd.query_devices(
                (resolve_device(args.input_device, is_input=True) or sd.default.device[0])
            )["default_samplerate"])
        else:
            self.samplerate = self.sr_model

        in_idx = resolve_device(args.input_device, is_input=True)
        out_idx = resolve_device(args.output_device, is_input=False)
        max_in = sd.query_devices(in_idx or sd.default.device[0])["max_input_channels"]
        max_out = sd.query_devices(out_idx or sd.default.device[1])["max_output_channels"]
        self.channels = int(min(max_in, max_out, 2))
        self.device_pair = (in_idx, out_idx)

        a = self.args
        zc = self.samplerate // 50
        self.zc = zc
        self.block_frame = int(np.round(a.block_time * self.samplerate / zc)) * zc
        self.block_frame_16k = 320 * self.block_frame // zc
        self.crossfade_frame = int(np.round(a.crossfade * self.samplerate / zc)) * zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * zc)
        self.sola_search_frame = zc
        self.extra_frame = int(np.round(a.extra_time_ce * self.samplerate / zc)) * zc
        self.extra_frame_right = int(np.round(a.extra_time_right * self.samplerate / zc)) * zc
        self.skip_head = self.extra_frame // zc
        self.skip_tail = self.extra_frame_right // zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // zc

        self.input_wav = torch.zeros(
            self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame + self.extra_frame_right,
            device=device, dtype=torch.float32)
        self.input_wav_res = torch.zeros(320 * self.input_wav.shape[0] // zc, device=device, dtype=torch.float32)
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=device, dtype=torch.float32)
        self.fade_in = torch.sin(0.5 * np.pi * torch.linspace(0, 1, self.sola_buffer_frame, device=device)) ** 2
        self.fade_out = 1 - self.fade_in
        self.resampler = tat.Resample(self.samplerate, 16000, dtype=torch.float32).to(device)
        self.resampler2 = (tat.Resample(self.sr_model, self.samplerate, dtype=torch.float32).to(device)
                           if self.sr_model != self.samplerate else None)

        self.reference_wav, _ = librosa.load(a.reference, sr=self.sr_model)
        if self.reference_wav.ndim > 1:
            self.reference_wav = librosa.to_mono(self.reference_wav)

        # funasr VAD state (only used in vc mode)
        self.vad_cache = {}
        self.vad_chunk_size = min(500, 1000 * a.block_time)
        self.vad_speech_detected = False
        self.vad_clear_flag = False

        self._print_every = max(1, int(10.0 / a.block_time))  # stats ~ every 10 s
        self._block_count = 0
        self._infer_sum = 0.0
        self._overruns = 0

        if self.mode == "vc":
            if a.extra_time_ce - a.extra_time < 0:
                raise SystemExit("extra-time-ce must be >= extra-time")

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _infer_vc(self):
        infer_wav = rtgui.custom_infer(
            self.model_set,
            self.reference_wav,
            self.args.reference,
            self.input_wav_res,
            self.block_frame_16k,
            self.skip_head,
            self.skip_tail,
            self.return_length,
            int(self.args.diffusion_steps),
            self.args.cfg_rate,
            self.args.max_prompt_length,
            self.args.extra_time_ce - self.args.extra_time,
        )
        if self.resampler2 is not None:
            infer_wav = self.resampler2(infer_wav.float())
        return infer_wav

    def _vad_pass(self, indata):
        indata_16k = librosa.resample(indata, orig_sr=self.samplerate, target_sr=16000)
        res = self.vad.generate(input=indata_16k, cache=self.vad_cache,
                                is_final=False, chunk_size=self.vad_chunk_size)
        res_value = res[0]["value"]
        if len(res_value) % 2 == 1 and not self.vad_speech_detected:
            self.vad_speech_detected = True
        elif len(res_value) % 2 == 1 and self.vad_speech_detected:
            self.vad_clear_flag = True

    def audio_callback(self, indata, outdata, frames, times, status):
        t0 = time.perf_counter()
        if status:
            self._overruns += 1
        indata = librosa.to_mono(indata.T)

        # slide the context window forward, append the new block
        self.input_wav[: -indata.shape[0]] = self.input_wav[indata.shape[0]:].clone()
        self.input_wav[-indata.shape[0]:] = torch.from_numpy(indata).to(self.device)
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[self.block_frame_16k:].clone()
        self.input_wav_res[-320 * (indata.shape[0] // self.zc + 1):] = torch.from_numpy(
            librosa.resample(self.input_wav[-indata.shape[0] - 2 * self.zc:].cpu().numpy(),
                             orig_sr=self.samplerate, target_sr=16000)[320:])

        if self.mode == "vc":
            self._vad_pass(indata)
            infer_wav = self._infer_vc()
            if not self.vad_speech_detected:
                infer_wav = torch.zeros_like(self.input_wav[self.extra_frame:])
        else:  # listen: straight passthrough of the freshest frames
            infer_wav = self.input_wav[self.extra_frame:].clone()

        # SOLA crossfade (verbatim from real-time-gui.py / DDSP-SVC)
        conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(F.conv1d(conv_input ** 2,
                                      torch.ones(1, 1, self.sola_buffer_frame, device=self.device)) + 1e-8)
        tensor = cor_nom[0, 0] / cor_den[0, 0]
        sola_offset = (torch.argmax(tensor).item() if tensor.numel() > 1 else float(tensor))

        infer_wav = infer_wav[sola_offset:]
        infer_wav[: self.sola_buffer_frame] *= self.fade_in
        infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out
        self.sola_buffer[:] = infer_wav[self.block_frame: self.block_frame + self.sola_buffer_frame]
        outdata[:] = (infer_wav[: self.block_frame]
                      .repeat(self.channels, 1).t().cpu().numpy())

        # stats
        self._block_count += 1
        self._infer_sum += time.perf_counter() - t0
        if self._block_count % self._print_every == 0:
            n = self._print_every
            print(f"[{self.mode}] blocks={self._block_count}  avg processing={1000 * self._infer_sum / n:.0f} ms/block "
                  f"(budget {int(self.args.block_time * 1000)} ms)  xruns={self._overruns}"
                  + ("  ⚠ AUDIO GLITCHES" if self._overruns else ""))
            self._infer_sum = 0.0
            if self.vad_clear_flag:
                self.vad_speech_detected = False
                self.vad_clear_flag = False

    def run(self):
        stream = sd.Stream(callback=self.audio_callback,
                           blocksize=self.block_frame,
                           samplerate=self.samplerate,
                           channels=self.channels,
                           dtype="float32",
                           device=self.device_pair)
        stream.start()
        try:
            algo_latency = stream.latency[-1] + self.args.block_time + self.args.crossfade \
                + self.args.extra_time_right + 0.01
        except Exception:
            algo_latency = float("nan")
        print("=" * 62)
        print(f" mode            : {self.mode}" + ("  (plain mic loopback)" if self.mode == "listen" else ""))
        print(f" reference voice : {self.args.reference}")
        print(f" input device    : {sd.query_devices(self.device_pair[0] or sd.default.device[0])['name']}")
        print(f" output device   : {sd.query_devices(self.device_pair[1] or sd.default.device[1])['name']}")
        print(f" samplerate      : {self.samplerate} Hz   block: {int(self.args.block_time * 1000)} ms")
        print(f" ~latency        : {int(algo_latency * 1000)} ms (speak, then wait a beat)")
        print(" Ctrl+C to stop")
        print("=" * 62)
        try:
            while stream.active:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            stream.abort()
            stream.close()
            print("stopped.")


def main():
    args = parse_args()
    if not os.path.isfile(args.reference):
        raise SystemExit(f"reference not found: {args.reference}\n"
                         f"try one of: {os.listdir(os.path.join(_here, 'examples/reference'))}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rtgui.device = device  # module global used by load_models / custom_infer
    print("loading models (first run downloads checkpoints)...")
    ckpt, cfg = args.checkpoint_path, args.config_path
    if ckpt and not os.path.isfile(ckpt):
        from hf_utils import load_custom_model_from_hf
        ckpt, cfg = load_custom_model_from_hf("Plachta/Seed-VC", ckpt, cfg)
        print(f"resolved HF download: {ckpt}")
    model_set = rtgui.load_models(argparse.Namespace(
        checkpoint_path=ckpt, config_path=cfg, fp16=args.fp16))
    vad = None
    if args.mode == "vc":
        from funasr import AutoModel
        vad = AutoModel(model="fsmn-vad", model_revision="v2.0.4", disable_update=True)
    RealtimeLoop(args, model_set, vad, device).run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        sys.exit(1)
