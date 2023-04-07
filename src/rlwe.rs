use concrete_commons::dispersion::DispersionParameter;
use concrete_commons::key_kinds::BinaryKeyKind;
use concrete_commons::parameters::{GlweDimension, GlweSize, PolynomialSize, PolynomialCount, ModulusSwitchOffset, LutCountLog};
use concrete_core::backends::core::private as ccore;
use ccore::crypto::encoding::PlaintextList;
use ccore::crypto::glwe::{GlweBody, GlweCiphertext, GlweMask};
use ccore::crypto::secret::generators::{EncryptionRandomGenerator, SecretRandomGenerator};
use ccore::crypto::secret::GlweSecretKey;
use ccore::crypto::encoding::Plaintext;
use ccore::math::polynomial::{MonomialDegree, Polynomial};
use ccore::math::tensor::{AsMutTensor, AsRefSlice, AsRefTensor, Tensor};
use concrete_core::backends::core::private::crypto::glwe::FourierGlweCiphertext;
use concrete_core::backends::core::private::crypto::secret::LweSecretKey;
use concrete_core::backends::core::private::math::fft::{AlignedVec, Complex64};
use concrete_core::backends::core::private::math::torus::UnsignedTorus;
use num_traits::identities::{One, Zero};
use crate::*;

#[cfg(feature = "serde_serialize")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde_serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
/// An RLWE ciphertext.
/// It is a wrapper around `GlweCiphertext` from concrete.
pub struct RLWECiphertext(pub(crate) GlweCiphertext<Vec<Scalar>>);

impl RLWECiphertext {
    pub fn allocate(poly_size: PolynomialSize) -> RLWECiphertext {
        RLWECiphertext(GlweCiphertext::from_container(vec![Scalar::zero(); poly_size.0 * 2], poly_size))
    }

    pub fn get_glwe_copy(&self, ctx: &mut Context) -> GlweCiphertext<Vec<Scalar>> {
        let mut glwe = GlweCiphertext::allocate(0 as Scalar, ctx.poly_size, GlweSize(100));
        glwe.clone_from(&self.0);
        return glwe;
    }

    pub fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }

    pub fn get_body(&self) -> GlweBody<&[Scalar]> {
        self.0.get_body()
    }

    pub fn get_mask(&self) -> GlweMask<&[Scalar]> {
        self.0.get_mask()
    }

    pub fn get_mut_mask(&mut self) -> GlweMask<&mut [Scalar]> {
        self.0.get_mut_mask()
    }

    pub fn get_mut_body(&mut self) -> GlweBody<&mut [Scalar]> {
        self.0.get_mut_body()
    }

    pub fn as_tensor(& self) -> &Tensor<Vec<Scalar>>{
        self.0.as_tensor()
    }

    pub fn clear(&mut self) {
        self.0.as_mut_tensor().fill_with(|| Scalar::zero());
    }

    pub fn fill_with_copy(&mut self, other: &RLWECiphertext) {
        self.0.as_mut_tensor().fill_with_copy(other.0.as_tensor());
    }

    pub fn update_mask_with_add<C>(&mut self, other: &Polynomial<C>)
        where C: AsRefSlice<Element=Scalar>
    {
        self.get_mut_mask().as_mut_polynomial_list().get_mut_polynomial(0)
            .update_with_wrapping_add(other);
    }

    pub fn update_mask_with_sub<C>(&mut self, other: &Polynomial<C>)
        where C: AsRefSlice<Element=Scalar>
    {
        self.get_mut_mask().as_mut_polynomial_list().get_mut_polynomial(0)
            .update_with_wrapping_sub(other);
    }

    pub fn update_body_with_add<C>(&mut self, other: &Polynomial<C>)
        where C: AsRefSlice<Element=Scalar>
    {
        self.get_mut_body().as_mut_polynomial()
            .update_with_wrapping_add(other);
    }

    pub fn update_body_with_sub<C>(&mut self, other: &Polynomial<C>)
        where C: AsRefSlice<Element=Scalar>
    {
        self.get_mut_body().as_mut_polynomial()
            .update_with_wrapping_sub(other);
    }

    pub fn update_with_add(&mut self, other: &RLWECiphertext) {
        self.update_mask_with_add(&other.get_mask().as_polynomial_list().get_polynomial(0));
        self.update_body_with_add(&other.get_body().as_polynomial());
    }

    pub fn update_with_sub(&mut self, other: &RLWECiphertext) {
        self.update_mask_with_sub(&other.get_mask().as_polynomial_list().get_polynomial(0));
        self.update_body_with_sub(&other.get_body().as_polynomial());
    }

    pub fn update_with_monomial_div(&mut self, m: MonomialDegree) {
        self.get_mut_body().as_mut_polynomial().update_with_wrapping_unit_monomial_div(m);
        self.get_mut_mask().as_mut_polynomial_list().get_mut_polynomial(0).update_with_wrapping_unit_monomial_div(m);
    }
}


#[derive(Debug, Clone)]
/// An RLWE ciphertext.
/// It is a wrapper around `GlweCiphertext` from concrete.
pub struct RLWEPublicKey(pub(crate) Vec<GlweCiphertext<Vec<Scalar>>>);

impl RLWEPublicKey {
    pub fn allocate(poly_size: PolynomialSize, glwe_size: GlweSize, m: usize) -> RLWEPublicKey {
        let mut glwe_list: Vec<GlweCiphertext<Vec<Scalar>>> = Vec::new();
        for _i in 0..m {
            let ct = GlweCiphertext::from_container(vec![Scalar::zero(); poly_size.0 * glwe_size.0], poly_size);
            glwe_list.push(ct);
        }
        RLWEPublicKey(glwe_list)
    }

    pub fn get_size(&self) -> usize {
        self.0.len()
    }

    pub fn update_with_global_mask(&mut self, global_pk: &RLWEPublicKey, ctx: &Context) {
        for i in 0..ctx.m {
            self.0.get_mut(i).unwrap().get_mut_mask().as_mut_tensor().fill_with_copy(global_pk.0.get(i).unwrap().get_mask().as_tensor());
        }
    }

    pub fn update_body_with_pk(&mut self, pk: &RLWEPublicKey, ctx: &Context) {
        for i in 0..ctx.m {
            self.0.get_mut(i).unwrap().get_mut_body().as_mut_tensor().update_with_wrapping_add(pk.0.get(i).unwrap().get_body().as_tensor());
        }
    }

    pub fn encrypt_rlwe(&self, encrypted: &mut RLWECiphertext, pt: &PlaintextList<Vec<Scalar>>, generator: &mut SecretRandomGenerator, m: usize) {

        assert_eq!(m,self.get_size());

        if m == Scalar::one() as usize{
            encrypted.0.get_mut_mask().as_mut_tensor().update_with_wrapping_add(self.0.get(0).unwrap().get_mask().as_tensor());
            encrypted.0.get_mut_body().as_mut_polynomial().update_with_wrapping_add(&self.0.get(0).unwrap().get_body().as_polynomial());
        } else {
            let r = RLWESecretKey::generate_binary(PolynomialSize(m), generator);
            //println!("Random vector = {:?}", r.as_tensor());
        
            for (i, val) in r.0.as_polynomial_list().as_tensor().iter().enumerate() {
                if *val == Scalar::one() {
                    encrypted.0.get_mut_mask().as_mut_tensor().update_with_wrapping_add(self.0.get(i).unwrap().get_mask().as_tensor());
                    encrypted.0.get_mut_body().as_mut_polynomial().update_with_wrapping_add(&self.0.get(i).unwrap().get_body().as_polynomial());
                }
            }
        }
        encrypted.0.get_mut_body().as_mut_polynomial().update_with_wrapping_add(&pt.as_polynomial());

    }

    pub fn compare_mask(&self, pk: &RLWEPublicKey, m: usize) {
        for i in 0..m {
            assert_eq!(self.0.get(i).unwrap().get_mask().as_polynomial_list().as_tensor(),pk.0.get(i).unwrap().get_mask().as_polynomial_list().as_tensor());
        }
    }

    pub fn get_numberof_mask(&self) -> PolynomialCount {
        self.0.get(0).unwrap().get_mask().as_polynomial_list().polynomial_count()
    }

}


#[derive(Debug, Clone)]
/// An RLWE secret key.
pub struct RLWESecretKey(pub(crate) GlweSecretKey<BinaryKeyKind, Vec<Scalar>>);

impl RLWESecretKey {
    /// Generate a secret key where the coefficients are binary.
    pub fn generate_binary(
        poly_size: PolynomialSize,
        generator: &mut SecretRandomGenerator,
    ) -> Self {
        RLWESecretKey(
            GlweSecretKey::generate_binary(GlweDimension(1), poly_size, generator)
        )
    }

    pub fn rotate(&mut self, degree: MonomialDegree){
        self.0.as_mut_polynomial_list().update_with_wrapping_monic_monomial_mul(degree);
    }

    pub fn to_lwe_secretkey(&mut self, ctx: &mut Context) -> LweSecretKey<BinaryKeyKind, Vec<Scalar>> {
        let mut sk = RLWESecretKey::generate_binary(ctx.poly_size, &mut ctx.secret_generator);
        sk.clone_from(self);
        return sk.0.into_lwe_secret_key();
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

    pub fn plaintext_index2(index: usize, ctnt: Scalar, count: PlaintextCount) -> PlaintextList<Vec<Scalar>> {
        let mut ptxt = PlaintextList::allocate(Scalar::zero(), count);
        for (i, ptxt) in ptxt.as_mut_polynomial().coefficient_iter_mut().enumerate() {
            if i == index {
                ptxt.set_one();
                *ptxt = ptxt.wrapping_mul(ctnt);
            };
        };
        ptxt
    }

    /// Generate a trivial secret key where the coefficients are all zero.
    pub fn zero(poly_size: PolynomialSize) -> Self {
        RLWESecretKey(
            GlweSecretKey::binary_from_container(vec![Scalar::zero(); poly_size.0], poly_size)
        )
    }

    pub fn fill_with_copy<C>(&mut self, t: &Tensor<C>)
        where Tensor<C>: AsRefSlice<Element=Scalar> {
        self.0.as_mut_tensor().fill_with_copy(t);
    }

    /// Encode and then encrypt the plaintext pt.
    pub fn binary_encrypt_rlwe(&self, encrypted: &mut RLWECiphertext, pt: &PlaintextList<Vec<Scalar>>,
                               ctx: &mut Context)
    {
        let mut binary_encoded = pt.clone();
        poly_binary_encode(&mut binary_encoded.as_mut_polynomial());
        self.encrypt_rlwe(encrypted, &binary_encoded, ctx.std, &mut ctx.encryption_generator);
    }

    pub fn binary_encrypt_rlwe_with_noise(&self, encrypted: &mut RLWECiphertext, pt: &PlaintextList<Vec<Scalar>>,
        ctx: &mut Context, noise: impl DispersionParameter) {
        let mut binary_encoded = pt.clone();
        poly_binary_encode(&mut binary_encoded.as_mut_polynomial());
        self.encrypt_rlwe(encrypted, &binary_encoded, noise, &mut ctx.encryption_generator);
    }


    /// Encode and then encrypt the plaintext pt.
    pub fn ternary_encrypt_rlwe(&self, encrypted: &mut RLWECiphertext, pt: &PlaintextList<Vec<Scalar>>,
                                ctx: &mut Context)
    {
        let mut ternary_encoded = pt.clone();
        poly_ternary_encode(&mut ternary_encoded.as_mut_polynomial());
        self.encrypt_rlwe(encrypted, &ternary_encoded, ctx.std, &mut ctx.encryption_generator);
    }

    /// Encrypt a plaintext pt.
    // TODO change API to use Context
    pub fn encrypt_rlwe(&self, encrypted: &mut RLWECiphertext, pt: &PlaintextList<Vec<Scalar>>,
                        noise_parameter: impl DispersionParameter, generator: &mut EncryptionRandomGenerator) {
        self.0.encrypt_glwe(&mut encrypted.0, pt, noise_parameter, generator);
    }

    
    pub fn fill_rlwe_pk(&self, encrypted: &mut RLWEPublicKey,
        noise_parameter: impl DispersionParameter, generator: &mut EncryptionRandomGenerator, m: usize) {
        for i in 0..m {
            self.0.encrypt_glwe_mask(&mut encrypted.0.get_mut(i).unwrap(), noise_parameter, generator);
        }
    }

    /// Encrypt a scalar.
    pub fn encrypt_constant_rlwe(&self, encrypted: &mut RLWECiphertext, pt: &Plaintext<Scalar>, ctx: &mut Context) {
        let mut encoded = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        *encoded.as_mut_polynomial().get_mut_monomial(MonomialDegree(0)).get_mut_coefficient() = pt.0;
        self.0.encrypt_glwe(&mut encrypted.0, &encoded, ctx.std, &mut ctx.encryption_generator);
    }

    pub fn generate_mask(&self, encrypted: &mut RLWEPublicKey, ctx: &mut Context) {
        for i in 0..ctx.m {
            self.0.fill_mask(encrypted.0.get_mut(i).unwrap(), &mut ctx.encryption_generator);
        }
    }

    /// Decrypt a RLWE ciphertext.
    pub fn decrypt_rlwe(
        &self,
        pt: &mut PlaintextList<Vec<Scalar>>,
        encrypted: &RLWECiphertext,
    ) {
        
        self.0.decrypt_glwe(pt, &encrypted.0);
    }

    /// Decrypt a RLWE ciphertext.
    pub fn decrypt_wrapping_rlwe(
        &self,
        pt: &mut PlaintextList<Vec<Scalar>>,
        encrypted: &RLWECiphertext,
    ) {
        
        self.0.decrypt_wrapping_glwe(pt, &encrypted.0);
    }

    /// Decrypt a RLWE ciphertext and then decode.
    pub fn binary_decrypt_rlwe(
        &self,
        pt: &mut PlaintextList<Vec<Scalar>>,
        encrypted: &RLWECiphertext,
    ) {
        self.decrypt_rlwe(pt, encrypted);
        poly_binary_decode(&mut pt.as_mut_polynomial());
    }

    /// Decrypt a RLWE ciphertext and then decode.
    pub fn ternary_decrypt_rlwe(
        &self,
        pt: &mut PlaintextList<Vec<Scalar>>,
        encrypted: &RLWECiphertext,
    ) {
        self.decrypt_rlwe(pt, encrypted);
        poly_ternary_decode(&mut pt.as_mut_polynomial());
    }

    /// Create an RGSW ciphertext of a constant.
    pub fn encrypt_constant_rgsw(&self, out: &mut RGSWCiphertext,
                                 pt: &Plaintext<Scalar>,
                                 ctx: &mut Context) {
        self.0.encrypt_constant_ggsw(&mut out.0, pt, ctx.std, &mut ctx.encryption_generator)
        // NOTE:for debugging we can use
        //self.0.trivial_encrypt_constant_ggsw(&mut out.0, encoded, ctx.std, &mut ctx.encryption_generator)
    }

    pub fn encrypt_constant_wrapping_rgsw(&self, out: &mut RGSWCiphertext,
        pt: &Plaintext<Scalar>,
        ctx: &mut Context) {
        self.0.encrypt_constant_wrapping_ggsw(&mut out.0, pt, ctx.std_rlwe, &mut ctx.encryption_generator)
    }

    /* 
    pub fn trivial_encrypt_constant_wrapping_rgsw(&self, out: &mut RGSWCiphertext,
        pt: &Plaintext<Scalar>,
        ctx: &mut Context) {
        self.0.trivial_encrypt_constant_wrapping_ggsw(&mut out.0, pt, ctx.std_rlwe, &mut ctx.encryption_generator)
    }
    */

    /// Create an RGSW ciphertext of a polynomial.
    pub fn encrypt_rgsw(&self, out: &mut RGSWCiphertext, encoded: &PlaintextList<Vec<Scalar>>, ctx: &mut Context) {
        // first create a constant encryption of 0, then add the decomposed encoded value to it
        self.encrypt_constant_rgsw(out, &Plaintext(Scalar::zero()), ctx);
        let mut buf = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        for (i, mut m) in out.0.as_mut_glwe_list().ciphertext_iter_mut().enumerate() {
            let level = (i / 2) + 1;
            let shift: usize = (Scalar::BITS as usize) - ctx.rgsw_base_log.0 * level;
            buf.as_mut_tensor().fill_with_copy(encoded.as_tensor());
            mul_const(&mut buf.as_mut_tensor(), 1 << shift);
            //println!("Shift when encrypting GSW: {:?}",1 << shift);
            if i % 2 == 0 {
                // in this case we're in the "top half" of the ciphertext
                m.get_mut_mask().as_mut_polynomial_list().get_mut_polynomial(0).update_with_wrapping_add(&buf.as_polynomial());
            } else {
                // this is the "bottom half"
                m.get_mut_body().as_mut_polynomial().update_with_wrapping_add(&buf.as_polynomial());
            }
        }
    }

    pub fn encrypt_wrapping_rgsw(&self, out: &mut RGSWCiphertext, encoded: &PlaintextList<Vec<Scalar>>, ctx: &mut Context) {
        // first create a constant encryption of 0, then add the decomposed encoded value to it
        self.encrypt_constant_wrapping_rgsw(out, &Plaintext(Scalar::zero()), ctx);
        let mut buf = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        for (i, mut m) in out.0.as_mut_glwe_list().ciphertext_iter_mut().enumerate() {
            let level = (i / 2) + 1;
            let shift: usize = (Scalar::BITS as usize) - ctx.rgsw_base_log.0 * level;
            buf.as_mut_tensor().fill_with_copy(encoded.as_tensor());
            mul_const(&mut buf.as_mut_tensor(), 1 << shift);
            //println!("Shift when encrypting GSW: {:?}",1 << shift);
            if i % 2 == 0 {
                // in this case we're in the "top half" of the ciphertext
                //println!("MASK = {:?}",m.get_mut_mask().as_mut_polynomial_list().get_mut_polynomial(0));
                m.get_mut_mask().as_mut_polynomial_list().get_mut_polynomial(0).update_with_wrapping_add(&buf.as_polynomial());
            } else {
                // this is the "bottom half"
                m.get_mut_body().as_mut_polynomial().update_with_wrapping_add(&buf.as_polynomial());
            }
        }
    }

    /// Create a vector of RGSW ciphertexts of a polynomial.
    pub fn encrypt_constant_rgsw_vec(&self, v: &Vec<Plaintext<Scalar>>, ctx: &mut Context) -> Vec<RGSWCiphertext> {
        v.iter().map(|pt| {
            let mut rgsw_ct = RGSWCiphertext::allocate(ctx.poly_size, ctx.rgsw_base_log, ctx.rgsw_level_count);
            self.encrypt_constant_rgsw(&mut rgsw_ct, pt, ctx);
            rgsw_ct
        }).collect()
    }

    pub fn polynomial_size(&self) -> PolynomialSize {
        self.0.polynomial_size()
    }

    pub fn as_mut_tensor(&mut self) -> &mut Tensor<Vec<Scalar>>{
        self.0.as_mut_tensor()
    }

    pub fn as_tensor(& self) -> &Tensor<Vec<Scalar>>{
        self.0.as_tensor()
    }

    /// Compute RGSW(-s), where s is self
    pub fn neg_gsw(&self, ctx: &mut Context) -> RGSWCiphertext {
        let neg_sk = {
            let mut pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
            for (x, y) in pt.as_mut_tensor().iter_mut().zip(self.0.as_tensor().iter()) {
                *x = y * Scalar::MAX;
                //println!("(x,y) = {:?},{:?}",x,y);
            }
            //println!("pt = {:?}",pt);
            pt
        };
        
        let mut neg_sk_ct = RGSWCiphertext::allocate(ctx.poly_size, ctx.negs_base_log, ctx.negs_level_count);
        self.encrypt_rgsw(&mut neg_sk_ct, &neg_sk, ctx);
        neg_sk_ct
    }

    pub fn neg_gsw2(list: PlaintextList<Vec<u32>>, ctx: &mut Context) -> PlaintextList<Vec<u64>> {
        let neg_sk = {
            let mut pt = PlaintextList::allocate(0u64, ctx.plaintext_count());
            for (x, y) in pt.as_mut_tensor().iter_mut().zip(list.as_tensor().iter()) {
                *x = (*y as u64) * u64::MAX;
                //println!("(x,y) = {:?},{:?}",x,y);
            }
            //println!("pt = {:?}",pt);
            pt
        };
        neg_sk
    }

}

pub fn pbs_modulus_switch<Scalar>(
    input: Scalar,
    poly_size: PolynomialSize,
    offset: ModulusSwitchOffset,
    lut_count_log: LutCountLog,
) -> MonomialDegree
where
    Scalar: UnsignedTorus,
{
    // First, do the left shift (we discard the offset msb)
    let mut output = input << offset.0;
    // Start doing the right shift
    output >>= Scalar::BITS - poly_size.log2().0 - 2 + lut_count_log.0;
    // Do the rounding
    output += output & Scalar::ONE;
    // Finish the right shift
    output >>= 1;
    // Apply the lsb padding
    output <<= lut_count_log.0;
    MonomialDegree(output.cast_into() as usize)
}




pub fn compute_noise<C>(sk: &RLWESecretKey, ct: &RLWECiphertext, encoded_ptxt: &PlaintextList<C>) -> f64
    where C: AsRefSlice<Element=Scalar>
{
    // pt = b - a*s = Delta*m + e
    let mut pt = PlaintextList::allocate(Scalar::zero(), encoded_ptxt.count());
    sk.decrypt_rlwe(&mut pt, ct);

    // pt = pt - Delta*m = e (encoded_ptxt is Delta*m)
    pt.as_mut_polynomial().update_with_wrapping_sub(&encoded_ptxt.as_polynomial());

    //println!("Error: {:?}", pt);

    let mut max_e = 0f64;
    for x in pt.as_tensor().iter() {
        // convert x to signed
        let z = (*x as SignedScalar).abs() as f64;
        //println!("z: {}",z);
        if z > max_e {
            max_e = z;
        }
    }
    println!("Real noise: {:?}",max_e / ((1u128 << Scalar::BITS) as f64));
    max_e.log2()
    //max_e
    //max_e / ((1u128 << Scalar::BITS) as f64)
}

pub fn compute_wrapping_noise<C>(sk: &RLWESecretKey, ct: &RLWECiphertext, encoded_ptxt: &PlaintextList<C>) -> f64
    where C: AsRefSlice<Element=Scalar>
{
    // pt = b - a*s = Delta*m + e
    let mut pt = PlaintextList::allocate(Scalar::zero(), encoded_ptxt.count());
    sk.decrypt_wrapping_rlwe(&mut pt, ct);

    // pt = pt - Delta*m = e (encoded_ptxt is Delta*m)
    pt.as_mut_polynomial().update_with_wrapping_sub(&encoded_ptxt.as_polynomial());

    //println!("Error: {:?}", pt);

    let mut max_e = 0f64;
    for x in pt.as_tensor().iter() {
        // convert x to signed
        let z = (*x as SignedScalar).abs() as f64;
        //println!("z: {}",z);
        if z > max_e {
            max_e = z;
        }
    }
    println!("Real noise: {:?}",max_e / ((1u128 << Scalar::BITS) as f64));
    max_e.log2()
    //max_e
    //max_e / ((1u128 << Scalar::BITS) as f64)
}

pub fn compute_noise_binary<C>(sk: &RLWESecretKey, ct: &RLWECiphertext, ptxt: &PlaintextList<C>) -> f64
    where C: AsRefSlice<Element=Scalar>
{
    
    let mut tmp = PlaintextList::allocate(Scalar::zero(), ptxt.count());
    tmp.as_mut_tensor().fill_with_copy(ptxt.as_tensor());
    poly_binary_encode(&mut tmp.as_mut_polynomial());
    
    // pt = b - a*s = Delta*m + e
    let mut pt = PlaintextList::allocate(Scalar::zero(), tmp.count());
    sk.decrypt_rlwe(&mut pt, ct);

    // pt = pt - Delta*m = e (encoded_ptxt is Delta*m)
    pt.as_mut_polynomial().update_with_wrapping_sub(&tmp.as_polynomial());

    //println!("Error: {:?}", pt_poly);

    let mut max_e = 0f64;
    for x in pt.as_tensor().iter() {
        // convert x to signed
        let z = (*x as SignedScalar).abs() as f64;
        //println!("z: {}",z);
        if z > max_e {
            max_e = z;
        }
    }
    println!("Real noise: {:?}",max_e / ((1u128 << Scalar::BITS) as f64));
    max_e.log2()
}

pub fn compute_noise_ternary<C>(sk: &RLWESecretKey, ct: &RLWECiphertext, ptxt: &PlaintextList<C>) -> f64
    where C: AsRefSlice<Element=Scalar>
{
    let mut tmp = PlaintextList::allocate(Scalar::zero(), ptxt.count());
    tmp.as_mut_tensor().fill_with_copy(ptxt.as_tensor());
    poly_ternary_encode(&mut tmp.as_mut_polynomial());
    compute_noise(sk, ct, &tmp)
}


#[derive(Debug, Clone)]
/// An RGSW ciphertext.
/// It is a wrapper around `StandardGgswCiphertext` from concrete.
pub struct FourierRLWECiphertext(pub(crate) FourierGlweCiphertext<AlignedVec<Complex64>, Scalar>);

impl FourierRLWECiphertext {

    pub fn allocate(poly_size: PolynomialSize) -> FourierRLWECiphertext {
        FourierRLWECiphertext(
            FourierGlweCiphertext::allocate(
                Complex64::new(0., 0.),
                poly_size,
                GlweSize(2),
            )
        )
    }

    pub fn fill_with_backward_fourier(&mut self, rlwe: &mut RLWECiphertext, buffers: &mut FourierBuffers<Scalar>) {
        self.0.fill_with_backward_fourier(&mut rlwe.0, buffers);
    }

}