"""Test vLLM ASR unified server on LibriSpeech test.clean.

Reports WER, inference time, RTFx, and avg per utterance.
Supports concurrent requests for batched server-side inference.
"""

import argparse
import asyncio
import base64
import io
import os
import time

import aiohttp
import datasets
import numpy as np
import soundfile as sf
from jiwer import wer
from tqdm import tqdm
from whisper_normalizer.english import EnglishTextNormalizer


def audio_to_base64(audio_arr: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio_arr, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


async def send_request(session, server_url, audio_b64):
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                    {"type": "text", "text": "Transcribe this audio."},
                ],
            }
        ],
        "max_tokens": 256,
        "temperature": 0.0,
    }
    t0 = time.time()
    async with session.post(
        f"{server_url}/v1/chat/completions", json=payload, timeout=aiohttp.ClientTimeout(total=300)
    ) as resp:
        data = await resp.json()
    client_latency_ms = (time.time() - t0) * 1000

    content = data["choices"][0]["message"]["content"]
    hyp = content.split("<debug_info>")[0].strip()

    server_time_ms = None
    import json as _json
    import re
    match = re.search(r"<debug_info>(.*?)</debug_info>", content, re.DOTALL)
    if match:
        try:
            info = _json.loads(match.group(1))
            server_time_ms = info.get("generation_time_ms")
        except _json.JSONDecodeError:
            pass

    return hyp, client_latency_ms, server_time_ms


async def run_batch(session, server_url, batch_b64s):
    tasks = [send_request(session, server_url, b64) for b64 in batch_b64s]
    results = await asyncio.gather(*tasks)
    hyps = [r[0] for r in results]
    client_latencies = [r[1] for r in results]
    server_times = [r[2] for r in results]
    return hyps, client_latencies, server_times


async def evaluate(server_url, ds, n, concurrency):
    all_refs = []
    all_hyps = []
    total_audio_secs = 0.0

    # Pre-encode all audio
    print(f"Encoding {n} audio samples...")
    audio_b64s = []
    for i in tqdm(range(n), desc="Encoding"):
        item = ds[i]
        audio_arr = item["audio"]["array"].astype(np.float32)
        sr = item["audio"]["sampling_rate"]
        total_audio_secs += len(audio_arr) / sr
        audio_b64s.append(audio_to_base64(audio_arr, sr))
        all_refs.append(item.get("text", ""))

    all_client_latencies = []
    all_server_times = []

    print(f"Sending {n} requests with concurrency={concurrency}...")
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        start = time.time()

        for batch_start in tqdm(range(0, n, concurrency), desc="Transcribing"):
            batch_end = min(batch_start + concurrency, n)
            batch = audio_b64s[batch_start:batch_end]
            hyps, client_lats, server_ts = await run_batch(session, server_url, batch)
            all_hyps.extend(hyps)
            all_client_latencies.extend(client_lats)
            all_server_times.extend(server_ts)

        elapsed = time.time() - start

    return all_refs, all_hyps, total_audio_secs, elapsed, all_client_latencies, all_server_times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--dataset", default="librispeech",
                        help="Dataset subset name (e.g., librispeech, ami, earnings22)")
    parser.add_argument("--split", default=None,
                        help="Split name (default: test.clean for librispeech, test for others)")
    args = parser.parse_args()

    split = args.split
    if split is None:
        split = "test.clean" if args.dataset == "librispeech" else "test"

    print(f"Loading {args.dataset} {split}...")
    ds = datasets.load_dataset(
        "hf-audio/esb-datasets-test-only-sorted",
        args.dataset,
        split=split,
    )

    n = args.max_samples or len(ds)

    all_refs, all_hyps, total_audio_secs, elapsed, client_lats, server_ts = asyncio.run(
        evaluate(args.server, ds, n, args.concurrency)
    )

    normalizer = EnglishTextNormalizer()
    refs_norm = [normalizer(r) for r in all_refs]
    hyps_norm = [normalizer(h) for h in all_hyps]

    # Filter out empty references (some datasets have them)
    valid = [(r, h) for r, h in zip(refs_norm, hyps_norm) if r.strip()]
    if len(valid) < len(refs_norm):
        print(f"  (filtered {len(refs_norm) - len(valid)} empty references)")
    refs_norm, hyps_norm = zip(*valid) if valid else ([], [])
    error_rate = wer(list(refs_norm), list(hyps_norm)) if refs_norm else 0.0

    rtfx = total_audio_secs / elapsed
    throughput = n / elapsed

    valid_server_ts = [t for t in server_ts if t is not None]
    avg_server_ms = sum(valid_server_ts) / len(valid_server_ts) if valid_server_ts else 0
    avg_client_ms = sum(client_lats) / len(client_lats)

    results = {
        "dataset": args.dataset,
        "split": split,
        "concurrency": args.concurrency,
        "utterances": n,
        "total_audio_secs": round(total_audio_secs, 1),
        "wer_pct": round(error_rate * 100, 2),
        "wall_clock_secs": round(elapsed, 2),
        "rtfx": round(rtfx, 1),
        "throughput_utt_per_s": round(throughput, 1),
        "avg_client_latency_ms": round(avg_client_ms, 1),
        "avg_wall_per_utt_ms": round(elapsed / n * 1000, 1),
    }

    print(f"\n{'=' * 70}")
    print(f"{args.dataset} {split} — vLLM ASR Unified Server (concurrency={args.concurrency})")
    print(f"{'=' * 70}")
    print(f"  {'Utterances:':<30s} {n}")
    print(f"  {'Total audio:':<30s} {total_audio_secs:.1f}s ({total_audio_secs/3600:.2f}h)")
    print(f"  {'WER:':<30s} {error_rate * 100:.2f}%")
    print(f"  {'Wall clock time:':<30s} {elapsed:.2f}s")
    print(f"  {'RTFx:':<30s} {rtfx:.1f}x")
    print(f"  {'Throughput:':<30s} {throughput:.1f} utt/s")
    print(f"  {'Avg client latency:':<30s} {avg_client_ms:.1f}ms  (HTTP round-trip + inference)")
    print(f"  {'Avg wall clock per utt:':<30s} {elapsed / n * 1000:.1f}ms  (wall / N)")
    print(f"{'=' * 70}")

    print("\nSample outputs:")
    for i in [0, min(n // 2, n - 1), n - 1]:
        print(f"  [{i}] ref: {all_refs[i][:80]}")
        print(f"  [{i}] hyp: {all_hyps[i][:80]}")
        print()

    # Save results to file
    import json
    os.makedirs("/home/dongjig/results/vllm_asr_eval", exist_ok=True)
    out_path = f"/home/dongjig/results/vllm_asr_eval/{args.dataset}_{split.replace('.', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
