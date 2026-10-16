"""
Autopilot's voice: Kokoro-82M (open weights, Apache 2.0) behind a small OpenAI-compatible API.

Made for Google Cloud Run (free amount, 2 vCPU; see deploy.sh). On 2 CPUs Kokoro makes speech
about 2.3x as fast as it plays, so it streams: the first clause comes back as MP3 in about half a
second, and the rest arrives while that one plays.

  POST /v1/audio/speech   { input, voice?, speed?, response_format?: segments|mp3|wav|pcm, stream?: true }

response_format "segments" (what Autopilot uses) streams each piece as its own small MP3, framed as
[1 byte: the pause after it (0 none, 1 short, 2 a full stop)][4 bytes: length, big-endian][MP3].
The app decodes and schedules each one the moment it arrives (Web Audio). Browsers playing one long
streamed MP3 through an <audio> element wait for ~40 KB before starting (~5 s of speech at 64 kbps),
which would undo everything here. Content-Type: application/vnd.autopilot.voice
  GET  /v1/audio/voices   { voices: [{ id }] }
  GET  /health            { ok, model, voices }

API_KEY is required (the app sends "Authorization: Bearer <API_KEY>"); without it the server won't
start, because a Cloud Run URL is public. ALLOW_NO_KEY=1 turns that off for local testing.
"""
import asyncio
import hmac
import io
import json
import os
import re
import threading
import time
import wave
from collections import OrderedDict
from contextlib import asynccontextmanager

import lameenc
import numpy as np
import onnxruntime as rt
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from kokoro_onnx import Kokoro

MODEL = os.environ.get('KOKORO_MODEL', 'kokoro-v1.0.fp16.onnx')
VOICES = os.environ.get('KOKORO_VOICES', 'voices-v1.0.bin')
API_KEY = os.environ.get('API_KEY', '').strip()
ALLOW_NO_KEY = os.environ.get('ALLOW_NO_KEY', '') == '1'
MAX_CHARS = int(os.environ.get('MAX_CHARS', '1500'))
MAX_BODY = 32 * 1024                        # a request is a line of text: anything bigger is refused
SAMPLE_RATE = 24000
FRAMES = 'application/vnd.autopilot.voice'
PAUSE_S = (0.0, 0.13, 0.28)                 # the app's quiet after a piece, by pause byte (lib/voiceFrames.js)

kokoro = None
ready = threading.Event()
lock = threading.Lock()                     # one synthesis at a time: it already uses every core
cache = OrderedDict()                       # (text, voice, speed) -> mp3 bytes; greetings repeat
CACHE_MAX = 300


def cpu_count():
    """
    Threads for the model. ORT_THREADS wins (deploy.sh sets it to the --cpu it deploys with). Without
    it: the container's CPU quota, then the CPUs it may run on. Not os.cpu_count(): in a container that
    can be the whole host, and on Cloud Run startup CPU boost briefly doubles the quota, so a pool sized
    while booting would be twice too big once the boost ends (threads fighting over 2 CPUs).
    """
    env = os.environ.get('ORT_THREADS', '').strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    try:
        with open('/sys/fs/cgroup/cpu.max') as f:
            quota, period = f.read().split()
        if quota != 'max':
            return max(1, int(int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


THREADS = cpu_count()


def load():
    global kokoro
    t = time.perf_counter()
    so = rt.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.inter_op_num_threads = 1
    so.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    session = rt.InferenceSession(MODEL, so, providers=['CPUExecutionProvider'])
    t_session = time.perf_counter() - t
    kokoro = Kokoro.from_session(session, VOICES)
    kokoro.create('Ready.', voice='af_heart', speed=1.0, lang='en-us')    # first run allocates: do it now
    ready.set()
    print(f'[voice] {MODEL} ready in {time.perf_counter() - t:.1f}s (session {t_session:.1f}s), '
          f'{THREADS} threads, {len(kokoro.get_voices())} voices, onnxruntime {rt.__version__}', flush=True)


@asynccontextmanager
async def lifespan(_app):
    if not API_KEY and not ALLOW_NO_KEY:
        raise RuntimeError('API_KEY is not set. Set it (deploy.sh does), or ALLOW_NO_KEY=1 for local testing.')
    # Load before the port opens (uvicorn binds after this returns). On Cloud Run, CPU is only given
    # during startup and while a request is being handled, so a model loading in a background thread
    # after the port opened would crawl. This way the instance takes traffic only once it can speak,
    # and requests that arrive while it starts simply wait for it.
    load()
    yield


app = FastAPI(title='Autopilot voice (Kokoro-82M)', docs_url=None, redoc_url=None, lifespan=lifespan)


def check_key(request: Request):
    if not API_KEY:
        return
    got = request.headers.get('authorization', '')
    if not hmac.compare_digest(got.encode(), f'Bearer {API_KEY}'.encode()):
        raise HTTPException(status_code=401, detail='bad key')


# ---- splitting a reply so it starts fast and never stalls -------------------------------------------
#
# Making speech costs a fixed ~0.25 s per piece plus ~0.36 s per second of speech (2 vCPU, measured).
# So: say a short first piece right away (~0.5 s), then size each next piece by how much speech is
# already queued, so it's ready before that runs out. RATE is learned from every piece, so this holds
# on whatever CPU the server lands on. Cuts go between clauses; a cut inside a clause happens only when
# waiting for the whole clause would leave an audible gap, and goes before a word like "to" or "and".

FIXED = 0.25                     # seconds per piece, whatever its length
RATE = 0.40                      # seconds of work per second of speech (learned below)
CHAR_S = 0.058                   # seconds of speech per character at speed 1
FIRST_MAX = 48                   # a first clause longer than this is cut, so speech still starts fast
PIECE_MAX = 300
GAP_OK = 0.4                     # a pause this short at a comma or full stop sounds natural
PHRASE = frozenset('to and but so or because which that who when while if then with for from at in on '
                   'about after before right just since until than as by into over'.split())
# a piece never ends on one of these: "Here's the | plan" sounds broken, "Here's the plan | for" doesn't
DANGLE = frozenset("the a an your my his her their our its this these those to of for with and or but i you we "
                   "they he she it is are was were be been am not no some any every each very just really have has "
                   "had do does did will would can could should i'll i'm you're we're let's at in on by from as than "
                   "about into over".split())
CLAUSE = re.compile(r'(?<=[.!?;:,\u2014])\s+')


def weight(s):
    """How long a piece is to say, in characters: a digit is said as a word ("7" -> "seven")."""
    return len(s) + 4 * sum(ch.isdigit() for ch in s)


def learn(took, audio_s):
    """Fold one piece's real speed into RATE (ignoring tiny pieces, where the fixed cost dominates)."""
    global RATE
    if audio_s >= 1.0:
        seen = max(0.05, (took - FIXED) / audio_s)
        RATE = 0.8 * RATE + 0.2 * min(seen, 3.0)


def cut_at_word(c, limit):
    """
    Where to cut clause c so its first part is about `limit` characters: after a word that isn't left
    dangling ("the", "your", "to"...), preferably right before a phrase word ("to", "and", "which"...).
    When no clean cut fits under the limit, the nearest clean cut past it (but well before the end):
    a slightly longer piece beats waiting for the whole clause. None if there's no clean cut at all.
    """
    limit = max(12, int(limit))
    word = lambda x: x.lower().strip(',.;:!?"\'')
    ok = [m.start() for m in re.finditer(' ', c)
          if m.start() >= 8 and word(c[:m.start()].rsplit(' ', 1)[-1]) not in DANGLE]
    if not ok:
        return None
    nice = lambda i: word(c[i + 1:].split(' ', 1)[0]) in PHRASE
    under = [i for i in ok if i <= limit]
    if under:
        good = [i for i in under if nice(i) and i >= 0.5 * under[-1]]
        i = (good or under)[-1]
    else:
        over = [i for i in ok if i <= 0.7 * len(c)]
        if not over:
            return None
        good = [i for i in over if nice(i) and i <= 2 * over[0]]
        i = (good or over)[0]
    return c[:i], c[i + 1:]


class Planner:
    """Hands out the next piece of a reply, sized by how much speech is queued ahead of the listener."""

    def __init__(self, text, speed=1.0):
        self.left = [c.strip() for c in CLAUSE.split(text) if c.strip()] or [text]
        self.speed = speed
        self.ahead = 0.0         # seconds of speech made but not yet heard
        self.first = True

    def chars_for(self, seconds):
        return seconds / CHAR_S * self.speed

    def next(self):
        """-> (piece, joins_mid_clause) or None when done."""
        if not self.left:
            return None
        c = self.left.pop(0)
        if self.first:                                                # the first piece: fast
            self.first = False
            if len(c) > FIRST_MAX and (cut := cut_at_word(c, 32)):
                self.left.insert(0, cut[1])
                return cut[0], True
            return self.trim(c)
        budget = self.chars_for(max(0.0, self.ahead * 0.85 - FIXED) / RATE)
        piece = c
        while self.left and weight(piece) + 1 + weight(self.left[0]) <= budget and len(piece) + 1 + len(self.left[0]) <= PIECE_MAX:
            piece += ' ' + self.left.pop(0)
        if weight(piece) <= budget or len(piece) <= 12:
            return self.trim(piece)
        # the next clause alone is too long to be ready in time: how long would they wait?
        wait = FIXED + RATE * weight(piece) * CHAR_S / self.speed - self.ahead
        limit = min(budget * len(piece) / weight(piece), PIECE_MAX)      # the budget, back in characters
        if wait <= GAP_OK or not (cut := cut_at_word(piece, limit)):
            return self.trim(piece)
        self.left.insert(0, cut[1])
        return cut[0], True

    def trim(self, piece):
        if len(piece) > PIECE_MAX:                    # one huge clause: cut it at a word, or anywhere
            a, b = cut_at_word(piece, PIECE_MAX) or hard_cut(piece, PIECE_MAX)
            self.left.insert(0, b)
            return a, True
        return piece, False

    def made(self, took, audio_s):
        """A piece is ready (`took` seconds to make, `audio_s` long): it's now queued for the listener."""
        self.ahead = max(0.0, self.ahead - took) + audio_s


def hard_cut(c, size):
    """Last resort for text with no usable space (a pasted link, "aaaa..."): cut at a space if any, else anywhere."""
    i = c.rfind(' ', 0, size + 1)
    return (c[:i], c[i + 1:]) if i > 0 else (c[:size], c[size:])


def blocks(text, size=PIECE_MAX):
    """Whole-reply (non-streamed) requests: clauses packed into pieces of up to `size` characters."""
    out = []
    for c in (c.strip() for c in CLAUSE.split(text)):
        while len(c) > size:
            a, b = cut_at_word(c, size) or hard_cut(c, size)
            out.append(a); c = b.strip()
        if not c:
            continue
        if out and len(out[-1]) + 1 + len(c) <= size:
            out[-1] += ' ' + c
        else:
            out.append(c)
    return out or [text]


def tighten(audio, lead=True, trail=True, keep=0.03):
    """Shave the silence Kokoro leaves at a piece's edges (for joins inside a clause, so they don't hitch)."""
    loud = np.flatnonzero(np.abs(audio) > 0.01)
    if not len(loud):
        return audio
    pad = int(keep * SAMPLE_RATE)
    a = max(0, loud[0] - pad) if lead else 0
    b = min(len(audio), loud[-1] + pad) if trail else len(audio)
    return audio[a:b]


def style_for(voice: str):
    """A voice id, or a mix like "af_heart(2)+af_bella(1)"."""
    names = kokoro.get_voices()
    if '+' not in voice:
        if voice not in names:
            raise HTTPException(status_code=400, detail=f'unknown voice {voice}')
        return voice
    total, mix = 0.0, None
    for part in voice.split('+'):
        m = re.match(r'^\s*([a-z]{2}_[a-z]+)\s*(?:\((\d+(?:\.\d+)?)\))?\s*$', part)
        if not m or m.group(1) not in names:
            raise HTTPException(status_code=400, detail=f'unknown voice {part}')
        w = float(m.group(2) or 1)
        s = kokoro.get_voice_style(m.group(1)) * w
        mix = s if mix is None else mix + s
        total += w
    return mix / total


def synth(text, voice, speed):
    """-> (audio, seconds of actual work). Waiting for the lock (someone else's line) isn't work."""
    with lock:
        t = time.perf_counter()
        audio, _ = kokoro.create(text, voice=voice, speed=speed, lang='en-us')
        work = time.perf_counter() - t
    return np.clip(audio, -1, 1), work


def log_line(text, done, started, first, audio_s, cut=False):
    """One line per reply in the logs (Cloud Run > Logs): how fast it really is on this hardware."""
    took = time.perf_counter() - started
    rtf = f', {took / audio_s:.2f}x real time' if audio_s else ''
    first_s = f'{first:.2f}s' if first is not None else '-'
    print(f'[voice] {len(text)} chars, {done} pieces{" (cut in)" if cut else ""}: first audio {first_s}, '
          f'{audio_s:.1f}s of speech in {took:.1f}s{rtf} (rate {RATE:.2f})', flush=True)


def pcm16(audio):
    return (audio * 32767).astype('<i2').tobytes()


def pause_after(piece, mid):
    """0: the next piece continues this clause; 2: a sentence ended; 1: a comma or the like."""
    if mid:
        return 0
    return 2 if piece.rstrip('"\'\u201d)').endswith(('.', '!', '?')) else 1


def mp3_encoder():
    e = lameenc.Encoder()
    e.set_bit_rate(64); e.set_in_sample_rate(SAMPLE_RATE); e.set_channels(1); e.set_quality(5)
    return e


def wav_bytes(pcm: bytes):
    b = io.BytesIO()
    with wave.open(b, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE); w.writeframes(pcm)
    return b.getvalue()


@app.get('/')
def root():
    return {'ok': True, 'service': 'Autopilot voice (Kokoro-82M)', 'ready': ready.is_set()}


@app.get('/health')
def health():
    return JSONResponse({'ok': ready.is_set(), 'model': MODEL, 'voices': len(kokoro.get_voices()) if ready.is_set() else 0},
                        status_code=200 if ready.is_set() else 503)


@app.get('/v1/audio/voices')
def voices(request: Request):
    check_key(request)
    if not ready.wait(timeout=60):
        raise HTTPException(status_code=503, detail='still loading')
    return {'voices': [{'id': v} for v in kokoro.get_voices()]}


@app.post('/v1/audio/speech')
async def speech(request: Request):
    check_key(request)
    try:
        if int(request.headers.get('content-length') or 0) > MAX_BODY:
            raise HTTPException(status_code=413, detail='too big')
    except ValueError:
        raise HTTPException(status_code=400, detail='bad content-length')
    raw = await request.body()
    if len(raw) > MAX_BODY:
        raise HTTPException(status_code=413, detail='too big')
    try:
        body = json.loads(raw)
        assert isinstance(body, dict)
    except Exception:
        raise HTTPException(status_code=400, detail='send JSON: { input, voice?, speed? }')
    text = re.sub(r'\s+', ' ', str(body.get('input') or '')).strip()[:MAX_CHARS]
    if not text:
        raise HTTPException(status_code=400, detail='input is empty')
    fmt = body.get('response_format') or 'mp3'
    if fmt not in ('segments', 'mp3', 'wav', 'pcm'):
        raise HTTPException(status_code=400, detail='response_format: segments, mp3, wav, or pcm')
    try:
        speed = min(1.6, max(0.6, float(body.get('speed') or 1.0)))
    except (TypeError, ValueError):
        speed = 1.0
    if not await asyncio.to_thread(ready.wait, 90):
        raise HTTPException(status_code=503, detail='still loading')
    voice = style_for(str(body.get('voice') or 'af_heart'))
    key = (text, str(body.get('voice') or 'af_heart'), round(speed, 2), fmt)
    framed = fmt == 'segments'
    if framed or fmt == 'mp3':
        if key in cache:
            cache.move_to_end(key)
            return Response(cache[key], media_type=FRAMES if framed else 'audio/mpeg', headers={'x-cache': 'hit'})
    if fmt in ('wav', 'pcm') or (fmt == 'mp3' and body.get('stream') is False):
        pcm = b''.join([pcm16((await asyncio.to_thread(synth, p, voice, speed))[0]) for p in blocks(text)])
        if fmt == 'pcm':
            return Response(pcm, media_type='audio/pcm')
        if fmt == 'wav':
            return Response(wav_bytes(pcm), media_type='audio/wav')
        enc = mp3_encoder()
        out = bytes(enc.encode(pcm)) + bytes(enc.flush())
        return Response(out, media_type='audio/mpeg')

    async def stream():
        enc = mp3_encoder()
        whole = bytearray()
        plan = Planner(text, speed)
        started = time.perf_counter()
        first = None
        audio_s = 0.0
        count = 0
        joined = False                                    # the last piece ended inside a clause
        finished = False
        try:
            while (step := plan.next()) is not None:
                piece, mid = step
                if await request.is_disconnected():      # they cut in: stop working on it
                    return
                t = time.perf_counter()
                audio, work = await asyncio.to_thread(synth, piece, voice, speed)
                learn(work, len(audio) / SAMPLE_RATE)     # how fast this CPU is
                if framed:
                    # the app trims each piece's edges and adds its own pause: plan with what it will
                    # actually play, or the planner thinks more is queued than the listener has
                    pause = pause_after(piece, mid)
                    audio = tighten(audio, keep=0.012)
                    heard = len(audio) / SAMPLE_RATE + PAUSE_S[pause]
                else:
                    if mid or joined:
                        audio = tighten(audio, lead=joined, trail=mid)
                    heard = len(audio) / SAMPLE_RATE
                joined = mid
                plan.made(time.perf_counter() - t, heard)   # the time incl. waiting: what they hear
                audio_s += len(audio) / SAMPLE_RATE
                count += 1
                if framed:                                # its own MP3, so it can be decoded alone
                    e = mp3_encoder()
                    mp3 = bytes(e.encode(pcm16(audio))) + bytes(e.flush())
                    chunk = bytes([pause]) + len(mp3).to_bytes(4, 'big') + mp3
                else:
                    chunk = bytes(enc.encode(pcm16(audio)))
                whole += chunk
                if chunk:
                    if first is None:
                        first = time.perf_counter() - started
                    yield chunk
            tail = b'' if framed else bytes(enc.flush())
            whole += tail
            if tail:
                yield tail
            finished = True
            cache[key] = bytes(whole)
            while len(cache) > CACHE_MAX:
                cache.popitem(last=False)
        finally:                                          # also runs when the listener hangs up mid-reply
            log_line(text, count, started, first, audio_s, cut=not finished)

    return StreamingResponse(stream(), media_type=FRAMES if framed else 'audio/mpeg', headers={'cache-control': 'no-store', 'x-accel-buffering': 'no'})
