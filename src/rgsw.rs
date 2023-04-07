use std::fmt::Debug;

use concrete_commons::parameters::{CiphertextCount, DecompositionBaseLog, DecompositionLevelCount, GlweSize, MonomialDegree, PolynomialSize};
use concrete_core::backends::core::private as ccore;
use ccore::crypto::ggsw::StandardGgswCiphertext;
use concrete_core::backends::core::private::crypto::bootstrap::{FourierBuffers};
use concrete_core::backends::core::private::crypto::ggsw::FourierGgswCiphertext;
use concrete_core::backends::core::private::crypto::glwe::FourierGlweCiphertext;
use concrete_core::backends::core::private::math::fft::{Complex64, AlignedVec, FourierPolynomial};
use get_size::GetSize;
use num_traits::identities::Zero;
use crate::rlwe::{RLWECiphertext};
use crate::*;

#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// An RGSW ciphertext.
/// It is a wrapper around `StandardGgswCiphertext` from concrete.
pub struct RGSWCiphertext(pub(crate) StandardGgswCiphertext<Vec<Scalar>>);

impl RGSWCiphertext {
    pub fn allocate(poly_size: PolynomialSize, decomp_base_log: DecompositionBaseLog, decomp_level: DecompositionLevelCount) -> RGSWCiphertext {
        RGSWCiphertext(
            StandardGgswCiphertext::allocate(
                Scalar::zero(),
                poly_size,
                GlweSize(2),
                decomp_level,
                decomp_base_log,
            )
        )
    }

    pub fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }

    pub fn decomposition_level_count(&self) -> DecompositionLevelCount {
        self.0.decomposition_level_count()
    }

    pub fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.0.decomposition_base_log()
    }

    pub fn ciphertext_count(&self) -> CiphertextCount {
        self.0.as_glwe_list().ciphertext_count()
    }

    pub fn ciphertext_size(&self) -> usize {
        self.0.as_tensor().get_size()
    }

    pub(crate) fn external_product_with_buf_glwe<C>(&self, out: &mut GlweCiphertext<C>, d: &RLWECiphertext, buffers: &mut FourierBuffers<Scalar>)
        where C: AsMutSlice<Element=Scalar>
    {
        let mut transformed = FourierGgswCiphertext::allocate(
            Complex64::new(0., 0.),
            self.polynomial_size(),
            GlweSize(2),
            self.decomposition_level_count(),
            self.decomposition_base_log(),
        );
        transformed.fill_with_forward_fourier(&self.0, buffers);

        let mut buffers = FourierBuffers::new(self.polynomial_size(), GlweSize(2));

        transformed.external_product(out, &d.0, &mut buffers);
    }

    pub fn external_product_with_buf(&self, out: &mut RLWECiphertext, d: &RLWECiphertext, buffers: &mut FourierBuffers<Scalar>) {
        self.external_product_with_buf_glwe(&mut out.0, d, buffers);
    }

    pub fn external_product(&self, out: &mut RLWECiphertext, d: &RLWECiphertext) {
        let mut buffers = FourierBuffers::new(self.polynomial_size(), GlweSize(2));
        self.external_product_with_buf(out, d, &mut buffers);
    }
    
    pub fn internal_product(&self, out: &mut RGSWCiphertext, ctxt: &RGSWCiphertext, ctx: &mut Context){
        for element in 0 .. out.ciphertext_count().0 {
            let mut aux = RLWECiphertext::allocate(ctx.poly_size);
            let mut buffers = FourierBuffers::new(self.polynomial_size(), GlweSize(2));
            let rlwe = ctxt.get_nth_row(element);
            self.external_product_with_buf(&mut aux, &rlwe, &mut buffers);
            (out.0.as_mut_glwe_list().ciphertext_iter_mut().nth(element).unwrap()).as_mut_tensor().fill_with_copy(&aux.get_glwe_copy(ctx).as_tensor());
        }
    }

    ///Multiplies out by X^degree. TODO: remove self
    pub fn product_monomial(&self, out: &mut RGSWCiphertext, degree: MonomialDegree){
        for element in 0 .. self.ciphertext_count().0 {
            (out.0.as_mut_glwe_list().ciphertext_iter_mut().nth(element).unwrap()).as_mut_polynomial_list().update_with_wrapping_monic_monomial_mul(degree);
        }
    }

    ///Substract ctxt from self
    pub fn sub_ciphertext(&mut self, ctxt: &RGSWCiphertext) {
        for (i, mut ct) in self.0.as_mut_glwe_list().ciphertext_iter_mut().enumerate() {
                ct.get_mut_mask().as_mut_polynomial_list().get_mut_polynomial(0).update_with_wrapping_sub(
                    &ctxt.0.as_glwe_list().ciphertext_iter().nth(i).unwrap().get_mask().as_polynomial_list().get_polynomial(0));
                ct.get_mut_body().as_mut_polynomial().update_with_wrapping_sub(
                    &ctxt.0.as_glwe_list().ciphertext_iter().nth(i).unwrap().get_body().as_polynomial());
        }
    }

    ///Add ctxt to self
    pub fn add_ciphertext(&mut self, ctxt: &RGSWCiphertext) {
        for (i, mut ct) in self.0.as_mut_glwe_list().ciphertext_iter_mut().enumerate() {
                ct.get_mut_mask().as_mut_polynomial_list().get_mut_polynomial(0).update_with_wrapping_add(
                    &ctxt.0.as_glwe_list().ciphertext_iter().nth(i).unwrap().get_mask().as_polynomial_list().get_polynomial(0));
                ct.get_mut_body().as_mut_polynomial().update_with_wrapping_add(
                    &ctxt.0.as_glwe_list().ciphertext_iter().nth(i).unwrap().get_body().as_polynomial());
        }
    }


    pub fn get_last_row(&self) -> RLWECiphertext {
        self.get_nth_row(self.decomposition_level_count().0 * 2 - 1)
    }

    pub fn get_nth_row(&self, n: usize) -> RLWECiphertext {
        let mut glwe_ct = GlweCiphertext::allocate(Scalar::zero(), self.polynomial_size(), GlweSize(2));
        glwe_ct.as_mut_tensor().fill_with_copy(self.0.as_glwe_list().ciphertext_iter().nth(n).unwrap().as_tensor());
        RLWECiphertext(glwe_ct)
    }

    pub fn get_last_row_nocast(&self) -> GlweCiphertext<Vec<Scalar>> {
        return self.get_nth_row_nocast(self.decomposition_level_count().0 * 2 - 1)
    }

    pub fn get_nth_row_nocast(&self, n: usize) -> GlweCiphertext<Vec<Scalar>> {
        let mut glwe_ct = GlweCiphertext::allocate(Scalar::zero(), self.polynomial_size(), GlweSize(2));
        glwe_ct.as_mut_tensor().fill_with_copy(self.0.as_glwe_list().ciphertext_iter().nth(n).unwrap().as_tensor());
        return glwe_ct;
    }
}

pub fn compute_noise_rgsw(gsw_ct: &RGSWCiphertext , ptxt: &PlaintextList<Vec<Scalar>>, sk: &RLWESecretKey, ctx: &Context) -> f64 {
    let mut error_pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
    sk.decrypt_wrapping_rlwe(&mut error_pt, &gsw_ct.get_nth_row(1));

    error_pt.as_mut_polynomial().update_with_wrapping_sub(&ptxt.as_polynomial());

    let mut max_e = 0f64;
    for x in error_pt.as_tensor().iter() {
        // convert x to signed
        let z = (*x as SignedScalar).abs() as f64;
        //println!("z: {}",z);
        if z > max_e {
            max_e = z;
        }
    }
    max_e.log2()
}

#[derive(Debug, Clone)]
/// An RGSW ciphertext.
/// It is a wrapper around `StandardGgswCiphertext` from concrete.
pub struct FourierRGSWCiphertext(pub(crate) FourierGgswCiphertext<AlignedVec<Complex64>, Scalar>);

impl FourierRGSWCiphertext {

    pub fn allocate(poly_size: PolynomialSize, decomp_base_log: DecompositionBaseLog, decomp_level: DecompositionLevelCount) -> FourierRGSWCiphertext {
        FourierRGSWCiphertext(
            FourierGgswCiphertext::allocate(
                Complex64::new(0., 0.),
                poly_size,
                GlweSize(2),
                decomp_level,
                decomp_base_log,
            )
        )
    }
    
    pub fn ciphertext_count(&self) -> CiphertextCount {
        self.0.as_glwe_list().ciphertext_count()
    }

    pub fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }

    pub fn decomposition_base_log(&self) -> DecompositionBaseLog {
        self.0.decomposition_base_log()
    }

    ///Add ctxt to self
    pub fn add_ciphertext(&mut self, ctxt: &FourierRGSWCiphertext) {
        for (i, mut ct) in self.0.as_mut_glwe_list().ciphertext_iter_mut().enumerate() {
            let (mut body, mut mask) = ct.get_mut_body_and_mask();
            //let now = Instant::now();
            mask.as_mut_polynomial_list().get_mut_polynomial(0).as_mut_tensor().update_with_add(ctxt.0.as_glwe_list().ciphertext_iter().nth(i).unwrap().get_mask().as_polynomial_list().get_polynomial(0).as_tensor());
            body.as_mut_tensor().update_with_add(&ctxt.0.as_glwe_list().ciphertext_iter().nth(i).unwrap().get_body().as_tensor());
        }
    }

    ///Substract ctxt from self
    pub fn sub_ciphertext(&mut self, ctxt: &FourierRGSWCiphertext) {
        for (i, mut ct) in self.0.as_mut_glwe_list().ciphertext_iter_mut().enumerate() {
            let (mut body, mut mask) = ct.get_mut_body_and_mask();
            mask.as_mut_polynomial_list().get_mut_polynomial(0).as_mut_tensor().update_with_add(ctxt.0.as_glwe_list().ciphertext_iter().nth(i).unwrap().get_mask().as_polynomial_list().get_polynomial(0).as_tensor());
            body.as_mut_tensor().update_with_sub(&ctxt.0.as_glwe_list().ciphertext_iter().nth(i).unwrap().get_body().as_tensor());
        }
    }

    

    pub fn fill_with_forward_fourier(&mut self, ctxt: &RGSWCiphertext, buffers: &mut FourierBuffers<Scalar>) {
        self.0.fill_with_forward_fourier(&ctxt.0, buffers);
    }

    pub fn get_nth_row(&self, n: usize) -> FourierRLWECiphertext {
        let mut glwe_ct = FourierGlweCiphertext::allocate(Complex64::new(0., 0.), self.polynomial_size(), GlweSize(2));
        glwe_ct.as_mut_tensor().fill_with_copy(self.0.as_glwe_list().ciphertext_iter().nth(n).unwrap().as_tensor());
        FourierRLWECiphertext(glwe_ct)


    }
    
    ///Multiply self by monomial, placing the result in out
    pub fn product_monomial(&self, out: &mut FourierRGSWCiphertext, monomial: &FourierPolynomial<AlignedVec<Complex64>>){
        
        for element in 0 .. self.ciphertext_count().0 {
            for (_i, mut poly) in (out.0.as_mut_glwe_list().ciphertext_iter_mut().nth(element).unwrap()).as_mut_polynomial_list().polynomial_iter_mut().enumerate() {
                let mut p = FourierPolynomial::allocate(Complex64::new(0., 0.), poly.polynomial_size());
                
                for (i,coeff) in p.coefficient_iter_mut().enumerate() {
                    *coeff = *poly.get_monomial(MonomialDegree(i)).get_coefficient();
                }

                let mut res = FourierPolynomial::allocate(Complex64::new(0., 0.), poly.polynomial_size());
                res.update_with_multiply_accumulate(&p, &monomial);

                for (i,coeff) in poly.coefficient_iter_mut().enumerate() {
                    *coeff = *res.coefficient_iter().nth(i).unwrap();
                }
                
                
                
            }
        }

    }

    pub fn external_product(&self, out: &mut RLWECiphertext, d: &RLWECiphertext) {
        let mut buffers = FourierBuffers::new(self.polynomial_size(), GlweSize(2));
        self.0.external_product(&mut out.0, &d.0, &mut buffers);
    }

}
