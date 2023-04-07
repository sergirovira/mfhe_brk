#![allow(deprecated)]

pub mod rlwe;
pub mod rgsw;
pub mod lwe;

use std::ops::Neg;

use concrete_core::backends::core::private as ccore;
use ccore::math::polynomial::Polynomial;
use ccore::math::random::RandomGenerator;
use ccore::math::tensor::{AsMutSlice, Tensor, AsMutTensor, AsRefTensor};
use ccore::crypto::encoding::PlaintextList;
use ccore::crypto::secret::generators::{EncryptionRandomGenerator, SecretRandomGenerator};
use ccore::crypto::glwe::GlweCiphertext;
use concrete_commons::parameters::{DecompositionBaseLog, DecompositionLevelCount, GlweSize, MonomialDegree, PlaintextCount, PolynomialSize};
use concrete_commons::dispersion::{LogStandardDev, StandardDev};
use concrete_core::backends::core::private::crypto::bootstrap::FourierBuffers;
use concrete_core::backends::core::private::math::fft::{Complex64, FourierPolynomial, AlignedVec};
use concrete_core::backends::core::private::math::polynomial::PolynomialList;
use num_traits::{One, Zero};
use crate::rgsw::RGSWCiphertext;
use crate::rlwe::*;

pub type Scalar = u64;
pub type SignedScalar = i64;

/// The context structure holds the TFHE parameters and
/// random number generators.
pub struct Context {
    pub random_generator: RandomGenerator,
    pub secret_generator: SecretRandomGenerator,
    pub encryption_generator: EncryptionRandomGenerator,
    pub std: LogStandardDev,
    //pub std_lwe: StandardDev,
    pub std_lwe: LogStandardDev,
    pub std_rlwe: StandardDev,
    pub poly_size: PolynomialSize,
    pub base_log: DecompositionBaseLog,
    pub level_count: DecompositionLevelCount,
    pub ks_base_log: DecompositionBaseLog,
    pub ks_level_count: DecompositionLevelCount,
    pub rgsw_base_log: DecompositionBaseLog,
    pub rgsw_level_count: DecompositionLevelCount,
    pub negs_base_log: DecompositionBaseLog,
    pub negs_level_count: DecompositionLevelCount,
    pub m: usize,
    pub glwe_size: GlweSize,
    pub k: usize,
}

impl Context {
    pub fn default() -> Context {
        let random_generator = RandomGenerator::new(None);
        let secret_generator = SecretRandomGenerator::new(None);
        let encryption_generator = EncryptionRandomGenerator::new(None);
        let std = LogStandardDev::from_log_standard_dev(-55.);
        let std_lwe = LogStandardDev::from_log_standard_dev(-47.); //std = 2^17 / 2^64 
        let std_rlwe = StandardDev::from_standard_dev(9.76908e-16); //std = 1.85*2^(4.2) / 2^64
        let poly_size = PolynomialSize(2048);
        let base_log = DecompositionBaseLog(4);
        let level_count = DecompositionLevelCount(15);
        let ks_base_log = DecompositionBaseLog(13);
        let ks_level_count = DecompositionLevelCount(3);
        let rgsw_base_log = DecompositionBaseLog(10);
        let rgsw_level_count = DecompositionLevelCount(5);
        let negs_base_log = DecompositionBaseLog(30);
        let negs_level_count = DecompositionLevelCount(2);
        let m = 3*64;
        let glwe_size = GlweSize(495);
        let k = 4;
        Context {
            random_generator,
            secret_generator,
            encryption_generator,
            std,
            std_lwe,
            std_rlwe,
            poly_size,
            base_log,
            level_count,
            ks_base_log,
            ks_level_count,
            rgsw_base_log,
            rgsw_level_count,
            negs_base_log,
            negs_level_count,
            m,
            glwe_size,
            k
        }
    }

    /// Output the plaintext count.
    pub fn plaintext_count(&self) -> PlaintextCount {
        PlaintextCount(self.poly_size.0)
    }

    /// Generate a binary plaintext.
    pub fn gen_binary_pt(&mut self) -> PlaintextList<Vec<Scalar>> {
        let cnt = self.plaintext_count();
        let mut ptxt = PlaintextList::allocate(Scalar::zero(), cnt);
        self.random_generator.fill_tensor_with_random_uniform_binary(ptxt.as_mut_tensor());
        ptxt
    }

    /// Generate a ternay plaintext.
    pub fn gen_ternary_ptxt(&mut self) -> PlaintextList<Vec<Scalar>> {
        let cnt = self.plaintext_count();
        let mut ptxt = PlaintextList::allocate(Scalar::zero(), cnt);
        self.random_generator.fill_tensor_with_random_uniform_ternary(ptxt.as_mut_tensor());
        ptxt
    }

    /// Generate a unit plaintext (all coefficients are 0 except the constant term is 1).
    pub fn gen_unit_pt(&self) -> PlaintextList<Vec<Scalar>> {
        let mut ptxt = PlaintextList::allocate(Scalar::zero(), self.plaintext_count());
        *ptxt.as_mut_polynomial().get_mut_monomial(MonomialDegree(0)).get_mut_coefficient() = Scalar::one();
        ptxt
    }

    /// Generate a plaintext where all the coefficients are 0.
    pub fn gen_zero_pt(&self) -> PlaintextList<Vec<Scalar>> {
        PlaintextList::allocate(Scalar::zero(), self.plaintext_count())
    }

    /// Generate a RLWE secret key.
    pub fn gen_rlwe_sk(&mut self) -> RLWESecretKey {
        RLWESecretKey::generate_binary(self.poly_size, &mut self.secret_generator)
    }

    /// Allocate and return buffers that are used for FFT.
    pub fn gen_fourier_buffers(&self) -> FourierBuffers<Scalar> {
        FourierBuffers::new(self.poly_size, GlweSize(2))
    }
}

/// Multiply a polynomial as tensor by a scalar
pub(crate) fn mul_const<C>(poly: &mut Tensor<C>, c: Scalar)
    where C: AsMutSlice<Element=Scalar>
{
    for coeff in poly.iter_mut() {
        *coeff = coeff.wrapping_mul(c);
    }
}

/// Returns a plaintext with all zeros but a 1 in the index position
pub fn plaintext_index(index: usize, ctnt: Scalar, ctx: &mut Context) -> PlaintextList<Vec<Scalar>> {
    let mut ptxt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
    for (i, ptxt) in ptxt.as_mut_polynomial().coefficient_iter_mut().enumerate() {
        if i == index {
            ptxt.set_one();
            *ptxt = ptxt.wrapping_mul(ctnt);
        };
    };
    ptxt
}

/// Encode binary x as x*(q/2)
pub fn binary_encode(x: &mut Scalar) {
    assert!(*x == 0 || *x == 1);
    *x = *x << (Scalar::BITS - 1)
}

pub fn binary_decode(x: &mut Scalar) {
    let lower = Scalar::MAX as Scalar >> 2;
    let upper = lower + (Scalar::MAX as Scalar >> 1);
    //println!("{:?},{:?},{:?}",lower,upper);
    if *x >= lower && *x < upper {
        *x = 1;
    } else {
        *x = 0;
    }
}

/// encode x as 2^64 / 8 or - 2^64 / 8 depending on sign
pub fn encode_accumulator(x: &mut Scalar) {
    let shift = (Scalar::BITS as usize) - 3;
    let y = *x as SignedScalar;

    if y < 0 {
        *x = 16140901064495857664;
        //*x = 3758096384;
    } else {
        *x = 1 << shift; 
    }
}

/// encode x as 2^32 / 8 or - 2^32 / 8 depending on sign
pub fn encode_accumulator32(x: &mut u32) {
    let shift = 32 - 3;
    let y = *x as i32;
    if y < 0 {
        *x = 3758096384;
    } else {
        *x = 1 << shift; 
    }
}

pub fn encode(x: &mut Scalar, ctx: &Context){
    let shift = (Scalar::BITS as usize) - ctx.rgsw_base_log.0;
    *x = *x*(1 << shift);
}

/// encode x as x * 2^64 / 8 or - x * 2^64 / 8 depending on sign 
pub fn encode_gate(x: &mut Scalar){
    let shift = (Scalar::BITS as usize) - 3;
    if *x != Scalar::zero() {
        *x = x.wrapping_mul(1 << shift);
    } else {
        *x = 16140901064495857664;
    }    
}

/// encode x as x * 2^32 / 8 or - x * 2^32 / 8 depending on sign
pub fn encode_gate32(x: &mut u32){
    let shift = 32 - 3;
    if *x != 0u32 {
        *x = x.wrapping_mul(1 << shift);
    } else {
        *x = 3758096384;
    }    
}


pub fn decode_gate(x: &mut Scalar){
    let shift = ((Scalar::BITS as usize) - 3) as Scalar;

    if *x == (1 << shift) {
        *x = Scalar::one();
    } else {
        *x = Scalar::zero();
    }
    
}

pub fn decode_gate32(x: &mut u32){
    let shift = 32 - 3 as u32;

    if *x == (1 << shift) {
        *x = 1u32;
    } else {
        *x = 0u32;
    }
    
}

pub fn decode(x: &mut Scalar, ctx: &Context){
    let shift = (Scalar::BITS as usize) - ctx.rgsw_base_log.0;
    *x = ((((*x as f64)/((1u128 << shift) as f64)).round() as f64) as Scalar)%(1 << ctx.rgsw_base_log.0);
}

pub fn decode_acc(x: &mut Scalar, ctx: &Context){
    *x = *x % (1 << ctx.rgsw_base_log.0);
}


/// Encode ternary x as x*(q/3)
pub fn ternary_encode(x: &mut Scalar) {
    const THIRD: Scalar = (Scalar::MAX as f64 / 3.0) as Scalar;
    if *x == 0 {
        *x = 0;
    } else if *x == 1 {
        *x = THIRD;
    } else if *x == Scalar::MAX {
        *x = 2*THIRD;
    } else {
        panic!("not a ternary scalar")
    }
}

pub fn ternary_decode(x: &mut Scalar) {
    const SIXTH: Scalar = (Scalar::MAX as f64 / 6.0) as Scalar;
    const THIRD: Scalar = SIXTH + SIXTH;
    const HALF: Scalar = Scalar::MAX / 2;
    if *x > SIXTH && *x <= HALF {
        *x = 1;
    } else if *x > HALF && *x <= HALF + THIRD {
        *x = Scalar::MAX;
    } else {
        *x = 0;
    }
}

/// Encode a binary polynomial.
pub fn poly_binary_encode<C>(xs: &mut Polynomial<C>)
    where C: AsMutSlice<Element=Scalar>
{
    for coeff in xs.coefficient_iter_mut() {
        binary_encode(coeff);
    }
}

pub fn poly_encode<C>(xs : &mut Polynomial<C>, ctx: &Context)
    where C: AsMutSlice<Element = Scalar>
{
    for coeff in xs.coefficient_iter_mut() {
        encode(coeff, ctx);
    }
}

pub fn poly_encode_accumulator<C>(xs : &mut Polynomial<C>)
    where C: AsMutSlice<Element = Scalar>
{
    for coeff in xs.coefficient_iter_mut() {
        encode_accumulator(coeff);
    }
}

pub fn poly_decode<C>(xs : &mut Polynomial<C>, ctx: &Context)
    where C: AsMutSlice<Element = Scalar>
{
    for coeff in xs.coefficient_iter_mut() {
        decode(coeff, ctx);
    }
}

pub fn poly_decode_gate<C>(xs : &mut Polynomial<C>)
    where C: AsMutSlice<Element = Scalar>
{
    for coeff in xs.coefficient_iter_mut() {
        decode_gate(coeff);
    }
}


pub fn poly_decode_acc<C>(xs : &mut Polynomial<C>, ctx: &Context)
    where C: AsMutSlice<Element = Scalar>
{
    for coeff in xs.coefficient_iter_mut() {
        decode_acc(coeff, ctx);
    }
}

pub fn poly_binary_decode<C>(xs: &mut Polynomial<C>)
    where C: AsMutSlice<Element=Scalar>
{
    for coeff in xs.coefficient_iter_mut() {
        binary_decode(coeff);
    }
}

/// Encode a ternary polynomial.
pub fn poly_ternary_encode<C>(xs: &mut Polynomial<C>)
    where C: AsMutSlice<Element=Scalar>
{
    for coeff in xs.coefficient_iter_mut() {
        ternary_encode(coeff);
    }
}

pub fn poly_ternary_decode<C>(xs: &mut Polynomial<C>)
    where C: AsMutSlice<Element=Scalar>
{
    for coeff in xs.coefficient_iter_mut() {
        ternary_decode(coeff);
    }
}

pub fn decomposed_rlwe_to_rgsw(cs: &Vec<RLWECiphertext>, neg_s: &RGSWCiphertext, ctx: &Context) -> RGSWCiphertext {
    let mut out = RGSWCiphertext::allocate(ctx.poly_size, ctx.rgsw_base_log,ctx.rgsw_level_count);
    let mut buffers = FourierBuffers::new(ctx.poly_size, GlweSize(2));
    for (i, mut c) in out.0.as_mut_glwe_list().ciphertext_iter_mut().enumerate() {
        let k = i / 2;
        if i % 2 == 0 {
            neg_s.external_product_with_buf_glwe(&mut c, &cs[k], &mut buffers);
        } else {
            c.as_mut_tensor().fill_with_copy(&cs[k].0.as_tensor());
        }
    }
    out
}

///Multiplies poly by X^degree mod X^N + 1
pub fn update_with_product_monomial(poly: &mut Polynomial<&mut [Complex64]>, degree: MonomialDegree) {
    let full_cycles_count = degree.0 / poly.as_tensor().len();
    if full_cycles_count % 2 != 0 {
        poly.as_mut_tensor()
            .as_mut_slice()
            .iter_mut()
            .for_each(|a| *a = a.neg());
    }
    let remaining_degree = degree.0 % poly.as_tensor().len();
    poly.as_mut_tensor()
        .as_mut_slice()
        .rotate_right(remaining_degree);
    poly.as_mut_tensor()
        .as_mut_slice()
        .iter_mut()
        .take(remaining_degree)
        .for_each(|a| *a = a.neg());
}