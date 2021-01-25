use rustfft::num_traits::Zero;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

#[derive(Clone)]
pub struct KPP {
    r0_at_zenith: f64,
    oscale: f64,
    zenith_angle: f64,
    wavelength: f64,
}
impl Default for KPP {
    fn default() -> Self {
        KPP {
            r0_at_zenith: 0.16,
            oscale: 25.0,
            zenith_angle: 30_f64.to_radians(),
            wavelength: 0.5e-6,
        }
    }
}
impl KPP {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn wavelength(self, wavelength: f64) -> Self {
        Self {
            wavelength,
            ..self
        }
    }
    pub fn pssn(
        self,
        pupil_size: f64,
        pupil_sampling: usize,
        telescope_pupil: &[f64],
    ) -> PSSn {
        let r0 = (self.r0_at_zenith.powf(-5_f64 / 3_f64) / self.zenith_angle.cos())
            .powf(-3_f64 / 5_f64)
            * (self.wavelength / 0.5e-6_f64).powf(1.2_f64);
        let n_otf = 2 * pupil_sampling - 1;
        let mut planner = FftPlanner::new();
        let forward = planner.plan_fft_forward(n_otf);
        let inverse = planner.plan_fft_inverse(n_otf);
        let d = pupil_size / (pupil_sampling - 1) as f64;
        let mut pssn = PSSn {
            r0: r0,
            oscale: self.oscale,
            wavelength: self.wavelength,
            pupil: pupil_size,
            n_pupil: pupil_sampling,
            n_otf,
            cpx_amplitude: vec![Complex::zero(); n_otf * n_otf],
            reference_telescope_otf: Default::default(),
            atmosphere_otf: PSSn::atmosphere_transfer_function(r0, self.oscale, d, n_otf),
            denom: 0f64,
            fft_forward: forward,
            fft_inverse: inverse,
            scratch: vec![Complex::zero(); n_otf * n_otf],
        };
        pssn.init(telescope_pupil);
        pssn
    }
}

pub struct PSSn {
    pub r0: f64,
    pub oscale: f64,
    wavelength: f64,
    pub pupil: f64,
    n_pupil: usize,
    n_otf: usize,
    cpx_amplitude: Vec<Complex<f64>>,
    pub reference_telescope_otf: Vec<Complex<f64>>,
    pub atmosphere_otf: Vec<f64>,
    denom: f64,
    fft_forward: Arc<dyn Fft<f64>>,
    fft_inverse: Arc<dyn Fft<f64>>,
    scratch: Vec<Complex<f64>>,
}
impl PSSn {
    fn atmosphere_transfer_function(r0: f64, big_l0: f64, d: f64, n_otf: usize) -> Vec<f64> {
        let mut atmosphere_otf: Vec<f64> = vec![0f64; n_otf * n_otf];
        for i in 0..n_otf {
            let q = i as i32 - n_otf as i32 / 2;
            let x = q as f64 * d;
            let ii = if q < 0i32 {
                (q + n_otf as i32) as usize
            } else {
                q as usize
            };
            for j in 0..n_otf {
                let q = j as i32 - n_otf as i32 / 2;
                let y = q as f64 * d;
                let jj = if q < 0i32 {
                    (q + n_otf as i32) as usize
                } else {
                    q as usize
                };
                let r = x.hypot(y);
                let kk = ii * n_otf + jj;
                atmosphere_otf[kk] = optust::phase::transfer_function(r, r0, big_l0);
            }
        }
        atmosphere_otf
    }
    fn cpx_amplitude_padding(&mut self, pupil: &[f64], phase: Option<&[f64]>) {
        self.cpx_amplitude =  vec![Complex::zero(); self.n_otf * self.n_otf];
        let wavenumber = 2. * std::f64::consts::PI / self.wavelength;
        for i in 0..self.n_pupil {
            let q = i as i32 - self.n_pupil as i32 / 2;
            let ii = if q < 0i32 {
                (q + self.n_otf as i32) as usize
            } else {
                q as usize
            };
            for j in 0..self.n_pupil {
                let q = j as i32 - self.n_pupil as i32 / 2;
                let jj = if q < 0i32 {
                    (q + self.n_otf as i32) as usize
                } else {
                    q as usize
                };
                let k = i* self.n_pupil +j;
                let kk = ii * self.n_otf + jj;
                match phase {
                    Some(ref phase) => {
                        let (s, c) = (wavenumber * phase[k]).sin_cos();
                        self.cpx_amplitude[kk].re = pupil[k] * c;
                        self.cpx_amplitude[kk].im = pupil[k] * s;
                    }
                    None => {
                        self.cpx_amplitude[kk].re = pupil[k];
                    }
                }
            }
        }
    }
    pub fn optical_transfer_function(
        &mut self,
        pupil: &[f64],
        phase: Option<&[f64]>
    ) {
        self.cpx_amplitude_padding(pupil, phase);
        self.fft_forward
            .process_with_scratch(&mut self.cpx_amplitude, &mut self.scratch);
        self.cpx_amplitude = (0..self.n_otf)
            .flat_map(|k| {
                self.cpx_amplitude
                    .iter()
                    .skip(k)
                    .step_by(self.n_otf)
                    .cloned()
                    .collect::<Vec<Complex<f64>>>()
            })
            .collect();
        self.fft_forward
            .process_with_scratch(&mut self.cpx_amplitude, &mut self.scratch);
        self.cpx_amplitude = self
            .cpx_amplitude
            .iter()
            .map(|z| Complex {
                re: z.norm_sqr() / (self.n_otf * self.n_otf) as f64,
                im: 0f64,
            })
            .collect();
        self.fft_inverse
            .process_with_scratch(&mut self.cpx_amplitude, &mut self.scratch);
        self.cpx_amplitude = (0..self.n_otf)
            .flat_map(|k| {
                self.cpx_amplitude
                    .iter()
                    .skip(k)
                    .step_by(self.n_otf)
                    .cloned()
                    .collect::<Vec<Complex<f64>>>()
            })
            .collect();
        self.fft_inverse
            .process_with_scratch(&mut self.cpx_amplitude, &mut self.scratch);
    }
    pub fn init(&mut self, pupil: &[f64]) {
        self.optical_transfer_function(pupil, None);
        self.reference_telescope_otf = self.cpx_amplitude.clone();
        self.denom = self
            .atmosphere_otf
            .iter()
            .zip(self.reference_telescope_otf.iter())
            .fold(0., |a, (c, o)| a + c * c * o.norm_sqr());
    }
    pub fn estimate(&mut self, pupil: &[f64], phase: Option<&[f64]>) -> f64 {
        self.optical_transfer_function(pupil, phase);
        let num = self
            .atmosphere_otf
            .iter()
            .zip(self.cpx_amplitude.iter())
            .fold(0., |a, (c, o)| a + c * c * o.norm_sqr());
        if self.denom == 0.0 {
            panic!("PSSn not initialized!")
        }
        num / self.denom
    }
}
