#!/usr/bin/env python3
"""Generates all LudoMax audio (background music + sound effects) from scratch.

All sounds are synthesised here, so there are no third-party audio licences to worry about.
Requires: numpy, ffmpeg (only for the optional .ogg conversion of the music).

Usage:  python3 scripts/generate_audio.py
Output: app/src/main/res/raw/*.wav and music_loop.ogg
"""
import os, subprocess, wave
import numpy as np

SR = 44100
OUT = os.path.join(os.path.dirname(__file__), "..", "app", "src", "main", "res", "raw")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)


def save_wav(name, x, peak=0.85):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(np.abs(x)) or 1.0
    x = x / m * peak
    pcm = (x * 32767).astype("<i2")
    with wave.open(os.path.join(OUT, name + ".wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def t_(dur):
    return np.arange(int(SR * dur)) / SR


def env_exp(n, decay):
    return np.exp(-np.arange(n) / SR * decay)


def fade(x, ms=4):
    n = int(SR * ms / 1000)
    x = x.copy()
    x[:n] *= np.linspace(0, 1, n)
    x[-n:] *= np.linspace(1, 0, n)
    return x


def bell(freq, dur, decay=7.0, bright=1.0):
    t = t_(dur)
    parts = [(1, 1.0), (2.01, 0.5 * bright), (3.0, 0.25 * bright), (4.2, 0.12 * bright)]
    x = sum(a * np.sin(2 * np.pi * freq * k * t) for k, a in parts)
    return fade(x * np.exp(-t * decay))


def place(buf, x, at):
    i = int(at * SR)
    n = min(len(x), len(buf) - i)
    if n > 0:
        buf[i:i + n] += x[:n]


def noise_burst(dur, decay, hp=True):
    n = int(SR * dur)
    x = rng.standard_normal(n)
    if hp:
        x = np.diff(x, prepend=0)
    return x * env_exp(n, decay)


# ------------------------------------------------------------------ SFX
def sfx_click():
    t = t_(0.06)
    return fade(np.sin(2 * np.pi * 1400 * t) * np.exp(-t * 70) + 0.4 * np.sin(2 * np.pi * 2100 * t) * np.exp(-t * 90))


def sfx_dice():
    buf = np.zeros(int(SR * 0.9))
    # clicks that slow down like a die settling
    times, tt, gap = [], 0.0, 0.035
    while tt < 0.75:
        times.append(tt)
        tt += gap
        gap *= 1.16
    for i, at in enumerate(times):
        amp = 0.5 + 0.5 * rng.random()
        k = noise_burst(0.05, 90) * amp
        th = np.sin(2 * np.pi * (180 + 60 * rng.random()) * t_(0.05)) * np.exp(-t_(0.05) * 60) * 0.7
        place(buf, k + th, at)
    return buf


def sfx_step():
    t = t_(0.09)
    f = 520 + 380 * np.exp(-t * 40)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return fade(np.sin(ph) * np.exp(-t * 38))


def sfx_out():
    t = t_(0.45)
    f = 380 + 700 * (t / 0.45) ** 1.5
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = (np.sin(ph) + 0.4 * np.sin(2 * ph)) * np.exp(-t * 5) * (1 - np.exp(-t * 200))
    buf = np.zeros(len(t) + int(SR * 0.1))
    place(buf, x, 0)
    place(buf, bell(1760, 0.3, 9) * 0.35, 0.22)
    return buf


def sfx_capture():
    t = t_(0.6)
    f = 760 * np.exp(-t * 4.2) + 90
    ph = 2 * np.pi * np.cumsum(f) / SR
    x = np.sign(np.sin(ph)) * 0.35 + np.sin(ph) * 0.6
    x = x * np.exp(-t * 4.5)
    x[: int(SR * 0.12)] += noise_burst(0.12, 28)[: int(SR * 0.12)] * 0.9
    return fade(x)


def sfx_home():
    buf = np.zeros(int(SR * 1.2))
    for i, f in enumerate([523.25, 659.25, 783.99, 1046.5]):
        place(buf, bell(f, 0.7, 5.5), 0.11 * i)
    return buf


def sfx_six():
    buf = np.zeros(int(SR * 0.6))
    place(buf, bell(783.99, 0.4, 8), 0.0)
    place(buf, bell(1046.5, 0.45, 7), 0.1)
    return buf


def sfx_nomove():
    buf = np.zeros(int(SR * 0.4))
    for i, f in enumerate([220, 165]):
        t = t_(0.17)
        x = (np.sign(np.sin(2 * np.pi * f * t)) * 0.5 + np.sin(2 * np.pi * f * t)) * np.exp(-t * 9)
        place(buf, fade(x), 0.17 * i)
    return buf


def sfx_win():
    buf = np.zeros(int(SR * 2.6))
    seq = [(523.25, 0.0), (659.25, 0.14), (783.99, 0.28), (1046.5, 0.42)]
    for f, at in seq:
        place(buf, bell(f, 0.5, 6), at)
    for f in (523.25, 659.25, 783.99, 1046.5, 1318.5):
        place(buf, bell(f, 1.9, 2.4, 0.8) * 0.8, 0.62)
    for i in range(8):
        place(buf, bell(1568 + 120 * i, 0.3, 11) * 0.3, 0.9 + 0.11 * i)
    return buf


# ------------------------------------------------------------------ Music
def note_freq(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def synth_note(freq, dur, kind="lead", vol=1.0):
    n = int(SR * dur)
    t = np.arange(n) / SR
    vib = 1 + 0.003 * np.sin(2 * np.pi * 5.5 * t)
    ph = 2 * np.pi * freq * t * vib
    if kind == "lead":      # soft bell-ish pluck
        x = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph)
        e = np.exp(-t * 5.5) * (1 - np.exp(-t * 400))
    elif kind == "pad":
        x = np.sin(ph) + 0.5 * np.sin(ph * 1.003) + 0.25 * np.sin(2 * ph)
        e = np.minimum(1, t * 6) * np.minimum(1, (dur - t) * 5)
    else:                   # bass
        x = np.sin(ph) + 0.4 * np.sin(2 * ph)
        e = np.exp(-t * 3.2) * (1 - np.exp(-t * 300))
    return x * e * vol


def saw(freq, t, harm=10):
    """Band-limited-ish saw wave (additive)."""
    x = np.zeros_like(t)
    for k in range(1, harm + 1):
        if freq * k > 9000:
            break
        x += np.sin(2 * np.pi * freq * k * t) / k
    return x * 0.6


def pluck_saw(freq, dur, vol, decay=9.0, detune=True):
    t = t_(dur)
    x = saw(freq, t)
    if detune:
        x = x + saw(freq * 1.006, t) * 0.8
    return fade(x * np.exp(-t * decay) * (1 - np.exp(-t * 500)) * vol, 3)


def make_music():
    """Energetic 140 BPM dance/arcade loop: kick, clap, hats, driving bass, chord stabs, arp + hook."""
    bpm = 140
    beat = 60 / bpm
    step = beat / 4                      # 16th note
    bars = 16
    total = bars * 4 * beat
    tail = 2.0
    buf = np.zeros(int(SR * (total + tail)))
    # Am - F - C - G  (roots in MIDI, triads an octave above)
    chords = [
        (45, [57, 60, 64]),   # Am
        (41, [57, 60, 65]),   # F
        (48, [55, 60, 64]),   # C
        (43, [55, 59, 62]),   # G
    ]
    # catchy 2-bar hook in A minor pentatonic (midi, length in 16ths), repeated; bars 4-15
    hook = [(81, 2), (0, 1), (81, 1), (84, 2), (81, 2), (79, 2), (76, 4),
            (0, 2), (79, 2), (81, 2), (84, 2), (88, 2), (84, 2), (81, 4), (0, 2), (76, 2), (79, 2), (81, 4)]
    # normalise hook to exactly 32 sixteenths (2 bars)
    acc, fixed = 0, []
    for m, l in hook:
        if acc + l > 32:
            l = 32 - acc
        if l <= 0:
            break
        fixed.append((m, l)); acc += l
    if acc < 32:
        fixed.append((0, 32 - acc))
    hook = fixed

    for bar in range(bars):
        root, triad = chords[bar % 4]
        t0 = bar * 4 * beat
        big = bar >= 4                    # first 4 bars: lighter intro, then full energy

        # kick on every beat
        for b in range(4):
            t = t_(0.22)
            f = 150 * np.exp(-t * 28) + 48
            k = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 11) * 0.95
            place(buf, k, t0 + b * beat)
        # off-beat open hat + 16th closed hats
        for i in range(16):
            amp = 0.20 if i % 4 == 2 else (0.09 if i % 2 == 0 else 0.05)
            place(buf, noise_burst(0.05 if i % 4 != 2 else 0.11, 90 if i % 4 != 2 else 45) * amp, t0 + i * step)
        # clap on 2 and 4
        if big or bar >= 2:
            for b in (1, 3):
                place(buf, noise_burst(0.16, 26) * 0.34, t0 + b * beat)
        # driving bass: eighth notes with octave jumps
        for i in range(8):
            m = root + (12 if i % 4 == 3 else 0)
            place(buf, synth_note(note_freq(m), beat * 0.48, "bass", 0.55), t0 + i * beat / 2)
        # chord stabs on the "and" of 2 and the 1 + 3.5 syncopation
        if big:
            for at in (0.0, 1.5, 2.5, 3.0):
                for m in triad:
                    place(buf, pluck_saw(note_freq(m), beat * 0.45, 0.075, 10.0), t0 + at * beat)
        # 16th arpeggio
        pattern = [0, 1, 2, 1, 2, 1, 0, 1, 2, 1, 0, 2, 1, 2, 0, 1]
        for i in range(16):
            m = triad[pattern[i]] + 12
            place(buf, pluck_saw(note_freq(m), step * 1.8, 0.07 if big else 0.05, 14.0, False), t0 + i * step)
        # hook melody
        if big:
            pos = 0
            off = (bar % 2) * 16
            # find where this bar sits inside the 2-bar hook
            cur = 0
            for m, l in hook:
                st, en = cur, cur + l
                cur = en
                if en <= off or st >= off + 16:
                    continue
                if m:
                    place(buf, pluck_saw(note_freq(m), max(l * step * 0.95, step), 0.17, 5.0), t0 + (max(st, off) - off) * step)
        # riser snare roll before bar 8 and bar 16 ends
        if bar in (7, 15):
            for i in range(8):
                amp = 0.10 + 0.04 * i
                place(buf, noise_burst(0.07, 60) * amp, t0 + 3 * beat + i * step / 2)

    # light echo for width
    d = int(SR * beat * 0.75)
    wet = buf.copy()
    wet[d:] += buf[:-d] * 0.18
    buf = wet
    # fold tail onto the start for a seamless loop
    n = int(SR * total)
    loop = buf[:n].copy()
    tail_part = buf[n:]
    loop[: len(tail_part)] += tail_part[: len(loop)]
    # gentle soft clip so it is loud but not harsh
    loop = np.tanh(loop * 1.2)
    return loop


if __name__ == "__main__":
    sfx = dict(sfx_click=sfx_click, sfx_dice=sfx_dice, sfx_step=sfx_step, sfx_out=sfx_out,
               sfx_capture=sfx_capture, sfx_home=sfx_home, sfx_six=sfx_six,
               sfx_nomove=sfx_nomove, sfx_win=sfx_win)
    for name, fn in sfx.items():
        save_wav(name, fn())
        print("wrote", name)
    m = make_music()
    tmp = os.path.join(OUT, "_music_tmp.wav")
    m = m / np.max(np.abs(m)) * 0.7
    with wave.open(tmp, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((m * 32767).astype("<i2").tobytes())
    ogg = os.path.join(OUT, "music_loop.ogg")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:a", "libvorbis", "-q:a", "3", ogg])
    if r.returncode == 0:
        os.remove(tmp)
        print("wrote music_loop.ogg")
    else:
        os.replace(tmp, os.path.join(OUT, "music_loop.wav"))
        print("ffmpeg/libvorbis missing -> wrote music_loop.wav (rename R.raw.music_loop is unchanged)")
