//! Audio effects a mod can put on the whole game's sound (capability `render`, `set-audio-fx`): pitch with
//! wow/flutter, a low-pass, echo, a small reverb, a tremolo, a master gain and a strict limiter.
//!
//! Everything is slewed: the targets a mod sends are approached with a time constant (1.2 s), the master gain
//! changes at most 3 dB a second, the effects' amounts are interpolated across each buffer, so nothing clicks or
//! jumps. `None` (the mod cleared it, stopped, or unloaded) means neutral, reached quickly (0.12 s; the gain still
//! at most 3 dB a second, and only ever going back UP to 0 dB from a quieter level, never louder than neutral).
//! The limiter ceiling applies while any effect is on: no peak above it, whatever the effects add up to.
//!
//! No dependency: the file is tested on its own (`rustc --edition 2024 --test audio_fx.rs`).

/// Floats a mod sends. Index: 0 pitch ratio, 1 wow depth, 2 flutter depth, 3 low-pass Hz (>= 20000: off),
/// 4 echo wet, 5 echo time (s), 6 reverb wet, 7 tremolo depth, 8 tremolo rate (Hz), 9 gain (dB, <= 0),
/// 10 limiter ceiling (0.3 to 1).
pub const PARAMS: usize = 12;

pub const NEUTRAL: [f32; PARAMS] = [1.0, 0.0, 0.0, 20_000.0, 0.0, 0.30, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0];

const MAX_ECHO_S: f32 = 0.7;
const TAU_NORMAL: f32 = 1.2;
const TAU_RESET: f32 = 0.12;
/// The master gain moves at most this many dB a second.
pub const GAIN_DB_PER_S: f32 = 3.0;
const WOW_HZ: f32 = 0.45;
const FLUTTER_HZ: f32 = 6.3;
const ECHO_FEEDBACK: f32 = 0.38;

/// Clamps what a mod sent to what is safe and meaningful.
pub fn sanitize(p: &[f32; PARAMS]) -> [f32; PARAMS] {
    let f = |i: usize| if p[i].is_finite() { p[i] } else { NEUTRAL[i] };
    [
        f(0).clamp(0.7, 1.3),
        f(1).clamp(0.0, 0.04),
        f(2).clamp(0.0, 0.012),
        f(3).clamp(700.0, 20_000.0),
        f(4).clamp(0.0, 0.5),
        f(5).clamp(0.08, MAX_ECHO_S - 0.05),
        f(6).clamp(0.0, 0.5),
        f(7).clamp(0.0, 0.3),
        f(8).clamp(0.3, 2.5),
        f(9).clamp(-9.0, 0.0),
        f(10).clamp(0.3, 1.0),
        0.0,
    ]
}

struct Comb {
    buf: Vec<f32>,
    i: usize,
    damp: f32,
}
struct Allpass {
    buf: Vec<f32>,
    i: usize,
}

/// One channel's state.
struct Chan {
    lp1: f32,
    lp2: f32,
    echo: Vec<f32>,
    echo_i: usize,
    combs: [Comb; 4],
    aps: [Allpass; 2],
}

pub struct AudioFx {
    target: [f32; PARAMS],
    cur: [f32; PARAMS],
    fast: bool,
    chans: Vec<Chan>,
    rate: u32,
    t: f64,
    t0: f64,
    lim_peak: f32,
    active: bool,
}

impl Default for AudioFx {
    fn default() -> Self {
        Self { target: NEUTRAL, cur: NEUTRAL, fast: false, chans: Vec::new(), rate: 0, t: 0.0, t0: 0.0, lim_peak: 1.0, active: false }
    }
}

impl AudioFx {
    /// What the mod asks for now; `None` is neutral, reached fast.
    pub fn set_target(&mut self, p: Option<&[f32; PARAMS]>) {
        match p {
            Some(p) => {
                self.target = sanitize(p);
                self.fast = false;
            }
            None => {
                self.target = NEUTRAL;
                self.fast = true;
            }
        }
    }

    /// The values applied now (the smoothed ones), for the status and the logs.
    pub fn applied(&self) -> [f32; PARAMS] {
        self.cur
    }
    pub fn is_active(&self) -> bool {
        self.active
    }

    fn at_neutral(&self) -> bool {
        self.cur.iter().zip(NEUTRAL.iter()).take(11).all(|(c, n)| (c - n).abs() < 1e-3 * n.abs().max(1.0))
            && self.target.iter().zip(NEUTRAL.iter()).take(11).all(|(c, n)| (c - n).abs() < 1e-6)
    }

    fn build(&mut self, channels: usize, rate: u32) {
        let r = rate as f32 / 44_100.0;
        let mk = |n: usize| ((n as f32 * r) as usize).max(8);
        self.chans = (0..channels.max(1))
            .map(|c| {
                let spread = c * 23;
                Chan {
                    lp1: 0.0,
                    lp2: 0.0,
                    echo: vec![0.0; (MAX_ECHO_S * rate as f32) as usize + 2],
                    echo_i: 0,
                    combs: [1116, 1188, 1277, 1356].map(|n| Comb { buf: vec![0.0; mk(n + spread)], i: 0, damp: 0.0 }),
                    aps: [556, 441].map(|n| Allpass { buf: vec![0.0; mk(n + spread)], i: 0 }),
                }
            })
            .collect();
        self.rate = rate;
    }

    /// Called once per buffer BEFORE the voices are mixed: advances the smoothing and returns the pitch ratio
    /// (with wow and flutter) the voices play at for these `frames`.
    pub fn begin(&mut self, frames: usize, rate: u32) -> f32 {
        let dt = frames as f32 / rate as f32;
        let tau = if self.fast { TAU_RESET } else { TAU_NORMAL };
        let k = 1.0 - (-dt / tau).exp();
        for i in 0..PARAMS {
            if i == 9 {
                let step = GAIN_DB_PER_S * dt;
                self.cur[9] += (self.target[9] - self.cur[9]).clamp(-step, step);
            } else if i == 3 {
                // the cutoff glides in octaves
                let (c, t) = (self.cur[3].max(1.0).ln(), self.target[3].max(1.0).ln());
                self.cur[3] = (c + (t - c) * k).exp();
            } else {
                self.cur[i] += (self.target[i] - self.cur[i]) * k;
            }
        }
        self.active = !self.at_neutral();
        if !self.active {
            // fully back to neutral: snap (differences under the thresholds) and forget the tails
            self.cur = NEUTRAL;
            if self.chans.iter().any(|c| c.echo_i != 0 || c.lp1 != 0.0) {
                self.chans.clear();
            }
            self.lim_peak = 1.0;
            return 1.0;
        }
        self.t0 = self.t;
        self.t += f64::from(dt);
        let mid = self.t0 + f64::from(dt) * 0.5;
        let tw = std::f64::consts::TAU;
        let wow = self.cur[1] * (tw * f64::from(WOW_HZ) * mid).sin() as f32;
        let flut = self.cur[2] * (tw * f64::from(FLUTTER_HZ) * mid + 1.3).sin() as f32;
        self.cur[0] * (1.0 + wow + flut)
    }

    /// Called once per buffer AFTER the voices are mixed (interleaved, `channels` per frame).
    /// `prev` is what `cur` was at the start of the buffer, `self.cur` the end: amounts are interpolated.
    pub fn process(&mut self, data: &mut [f32], channels: usize, rate: u32, prev: &[f32; PARAMS]) {
        if !self.active {
            return;
        }
        if self.chans.len() != channels.max(1) || self.rate != rate {
            self.build(channels, rate);
        }
        let frames = data.len() / channels.max(1);
        if frames == 0 {
            return;
        }
        let sr = rate as f32;
        let tw = std::f32::consts::TAU;
        let (c0, c1) = (prev, &self.cur);
        let echo_len = self.chans[0].echo.len();
        // a lower ceiling holds at once (a limiter's attack); a higher one is approached with the rest
        let ceiling = c1[10].min(self.target[10]);
        for n in 0..frames {
            let u = (n + 1) as f32 / frames as f32;
            let mix = |i: usize| c0[i] + (c1[i] - c0[i]) * u;
            let fc = (mix(3).max(1.0).ln()).exp().min(sr * 0.45);
            let a = 1.0 - (-tw * fc / sr).exp();
            let echo_w = mix(4);
            let rev_w = mix(6);
            let trem = mix(7);
            let g = 10f32.powf(mix(9) / 20.0);
            let delay = ((mix(5) * sr) as usize).clamp(1, echo_len - 1);
            let phase = (self.t0 + (n as f64) / f64::from(sr)) as f32;
            let trem_gain = 1.0 - trem * 0.5 * (1.0 - (tw * mix(8) * phase).cos());
            // a mono feed for the reverb
            let mut mono = 0.0;
            for c in 0..channels {
                mono += data[n * channels + c];
            }
            mono /= channels as f32;
            for c in 0..channels {
                let ch = &mut self.chans[c];
                let x = data[n * channels + c];
                // low-pass, two poles
                ch.lp1 += a * (x - ch.lp1);
                ch.lp2 += a * (ch.lp1 - ch.lp2);
                let dry = ch.lp2;
                // echo: feedback delay, crossing channels a little
                let read = (ch.echo_i + echo_len - delay) % echo_len;
                let e = ch.echo[read];
                ch.echo[ch.echo_i] = (dry + e * ECHO_FEEDBACK).clamp(-4.0, 4.0);
                ch.echo_i = (ch.echo_i + 1) % echo_len;
                // reverb: four combs and two allpasses on the mono feed
                let mut r = 0.0;
                let input = mono * 0.25;
                for cb in &mut ch.combs {
                    let o = cb.buf[cb.i];
                    cb.damp = o * 0.6 + cb.damp * 0.4;
                    cb.buf[cb.i] = (input + cb.damp * 0.80).clamp(-4.0, 4.0);
                    cb.i = (cb.i + 1) % cb.buf.len();
                    r += o;
                }
                for ap in &mut ch.aps {
                    let b = ap.buf[ap.i];
                    let o = -r + b;
                    ap.buf[ap.i] = r + b * 0.5;
                    ap.i = (ap.i + 1) % ap.buf.len();
                    r = o;
                }
                data[n * channels + c] = (dry + e * echo_w + r * rev_w) * trem_gain * g;
            }
        }
        // strict limiter: instant attack on the buffer's peak, release over 250 ms, then a hard clamp
        let peak = data.iter().fold(ceiling, |p, s| p.max(s.abs()));
        self.lim_peak = self.lim_peak.max(peak).max(ceiling);
        let release = (-1.0 / (0.25 * sr)).exp();
        for frame in data.chunks_mut(channels) {
            let gain = ceiling / self.lim_peak;
            for s in frame {
                *s = (*s * gain).clamp(-ceiling, ceiling);
            }
            self.lim_peak = (self.lim_peak * release).max(peak.min(self.lim_peak)).max(ceiling);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;
    const BUF: usize = 512;

    fn run(fx: &mut AudioFx, secs: f32, mut src: impl FnMut(usize) -> f32, mut watch: impl FnMut(&[f32])) {
        let mut n = 0usize;
        let buffers = (secs * RATE as f32) as usize / BUF;
        for _ in 0..buffers {
            let prev = fx.applied();
            let _pitch = fx.begin(BUF, RATE);
            let mut data = vec![0.0f32; BUF * 2];
            for f in 0..BUF {
                let s = src(n);
                data[f * 2] = s;
                data[f * 2 + 1] = s;
                n += 1;
            }
            fx.process(&mut data, 2, RATE, &prev);
            watch(&data);
        }
    }
    fn sine(f: f32, a: f32) -> impl FnMut(usize) -> f32 {
        move |n| a * (std::f32::consts::TAU * f * n as f32 / RATE as f32).sin()
    }
    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32).sqrt()
    }
    const WILD: [f32; PARAMS] = [0.9, 0.03, 0.01, 900.0, 0.4, 0.35, 0.4, 0.25, 2.0, -3.0, 0.6, 0.0];

    #[test]
    fn neutral_is_a_passthrough() {
        let mut fx = AudioFx::default();
        let mut data = vec![0.5f32; BUF * 2];
        let prev = fx.applied();
        assert_eq!(fx.begin(BUF, RATE), 1.0);
        fx.process(&mut data, 2, RATE, &prev);
        assert!(data.iter().all(|s| *s == 0.5));
        assert!(!fx.is_active());
    }

    #[test]
    fn nothing_exceeds_the_ceiling_even_with_a_huge_input() {
        let mut fx = AudioFx::default();
        fx.set_target(Some(&WILD));
        let mut worst = 0.0f32;
        run(&mut fx, 6.0, sine(220.0, 6.0), |d| worst = d.iter().fold(worst, |p, s| p.max(s.abs())));
        assert!(worst <= 0.6 + 1e-6, "peak {worst}");
    }

    #[test]
    fn gain_moves_at_most_3_db_a_second_and_never_jumps() {
        let mut fx = AudioFx::default();
        let mut p = NEUTRAL;
        p[9] = -9.0;
        fx.set_target(Some(&p));
        let mut last = 0.0f32;
        let mut worst = 0.0f32;
        for _ in 0..(4 * RATE as usize / BUF) {
            fx.begin(BUF, RATE);
            let db = fx.applied()[9];
            worst = worst.max((db - last).abs() / (BUF as f32 / RATE as f32));
            last = db;
        }
        assert!(worst <= GAIN_DB_PER_S + 1e-3, "{worst} dB/s");
        assert!(fx.applied()[9] < -8.0 + 0.1 || fx.applied()[9] < -5.0);
    }

    #[test]
    fn clearing_returns_to_an_exact_neutral_quickly() {
        let mut fx = AudioFx::default();
        fx.set_target(Some(&WILD));
        run(&mut fx, 5.0, sine(330.0, 0.5), |_| {});
        assert!(fx.is_active());
        fx.set_target(None);
        // the gain (at -3 dB, 1 dB a second slower than the rest) is the last one home: 1 s at most
        run(&mut fx, 1.3, sine(330.0, 0.5), |_| {});
        assert!(!fx.is_active(), "still active: {:?}", fx.applied());
        assert_eq!(fx.applied(), NEUTRAL);
    }

    #[test]
    fn reset_never_goes_louder_than_neutral() {
        let mut fx = AudioFx::default();
        let mut p = NEUTRAL;
        p[9] = -3.0;
        p[4] = 0.3;
        p[7] = 0.2;
        fx.set_target(Some(&p));
        run(&mut fx, 4.0, sine(330.0, 0.4), |_| {});
        fx.set_target(None);
        let mut worst = 0.0f32;
        run(&mut fx, 1.5, sine(330.0, 0.4), |d| worst = d.iter().fold(worst, |p, s| p.max(s.abs())));
        assert!(worst <= 0.4 * 1.3, "peak {worst}");
    }

    #[test]
    fn lowpass_takes_the_highs_away() {
        let mut fx = AudioFx::default();
        let mut p = NEUTRAL;
        p[3] = 1000.0;
        fx.set_target(Some(&p));
        let mut high = Vec::new();
        run(&mut fx, 8.0, sine(6000.0, 0.5), |d| high.extend_from_slice(d));
        let tail = &high[high.len() - 8000..];
        assert!(rms(tail) < 0.02, "rms {}", rms(tail));
        let mut fx = AudioFx::default();
        fx.set_target(Some(&p));
        let mut low = Vec::new();
        run(&mut fx, 8.0, sine(200.0, 0.5), |d| low.extend_from_slice(d));
        assert!(rms(&low[low.len() - 8000..]) > 0.3);
    }

    #[test]
    fn tremolo_never_gets_louder_than_the_source() {
        let mut fx = AudioFx::default();
        let mut p = NEUTRAL;
        p[7] = 0.3;
        p[8] = 2.0;
        fx.set_target(Some(&p));
        let mut worst = 0.0f32;
        run(&mut fx, 6.0, sine(300.0, 0.5), |d| worst = d.iter().fold(worst, |p, s| p.max(s.abs())));
        assert!(worst <= 0.5 + 1e-4, "{worst}");
    }

    #[test]
    fn pitch_ratio_follows_the_target_with_a_wobble() {
        let mut fx = AudioFx::default();
        let mut p = NEUTRAL;
        p[0] = 0.9;
        p[1] = 0.02;
        fx.set_target(Some(&p));
        let mut seen = Vec::new();
        for _ in 0..(10 * RATE as usize / BUF) {
            seen.push(fx.begin(BUF, RATE));
        }
        let last = &seen[seen.len() - 200..];
        let (lo, hi) = last.iter().fold((9.0f32, 0.0f32), |(l, h), v| (l.min(*v), h.max(*v)));
        assert!(lo > 0.86 && hi < 0.94 && hi - lo > 0.01, "{lo} {hi}");
    }
}
