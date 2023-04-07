

use concrete_commons::dispersion::DispersionParameter;
use concrete_commons::key_kinds::BinaryKeyKind;
use concrete_commons::parameters::{PolynomialSize, LweDimension, ModulusSwitchOffset, LutCountLog};
use concrete_core::backends::core::private as ccore;
use ccore::crypto::secret::generators::{EncryptionRandomGenerator, SecretRandomGenerator};
use ccore::crypto::encoding::Plaintext;
use ccore::math::decomposition::SignedDecomposer;
use ccore::math::polynomial::{MonomialDegree, Polynomial};
use ccore::math::tensor::{AsMutTensor, AsRefTensor, Tensor};
use concrete_core::backends::core::private::crypto::encoding::Cleartext;
use concrete_core::backends::core::private::crypto::lwe::{LweCiphertext, LweKeyswitchKey};
use concrete_core::backends::core::private::crypto::secret::LweSecretKey;
use num_traits::identities::{One, Zero};
use crate::*;
use crate::rgsw::FourierRGSWCiphertext;
use ccore::crypto::lwe::{LweBody, LweMask};
use concrete_commons::parameters::{LweSize};

#[derive(Debug, Clone)]
/// A LWE ciphertext.
/// It is a wrapper around `LweCiphertext` from concrete.
pub struct LWECiphertext(pub(crate) LweCiphertext<Vec<Scalar>>);


impl LWECiphertext {
    pub fn allocate(size: LweSize) -> LWECiphertext {
        LWECiphertext(LweCiphertext::allocate(Scalar::zero(), size))
    }

    /// Return the length of the mask + 1 for the body.
    pub fn lwe_size(&self) -> LweSize {
        self.0.lwe_size()
    }

    pub fn get_body(&self) -> &LweBody<Scalar> {
        self.0.get_body()
    }

    pub fn get_mask(&self) -> LweMask<&[Scalar]> {
        self.0.get_mask()
    }

    pub fn get_mut_mask(&mut self) -> LweMask<&mut [Scalar]> {
        self.0.get_mut_mask()
    }

    pub fn get_mut_body(&mut self) -> &mut LweBody<Scalar> {
        self.0.get_mut_body()
    }

    pub fn clear(&mut self) {
        self.0.as_mut_tensor().fill_with(|| Scalar::zero());
    }

    pub fn fill_with_sample_extract(&mut self, c: &RLWECiphertext, n_th: MonomialDegree) {
        self.0.fill_with_glwe_sample_extraction(&c.0, n_th);
    }

    pub fn fill_with_const_sample_extract(&mut self, c: &RLWECiphertext) {
        self.0
            .fill_with_glwe_sample_extraction(&c.0, MonomialDegree(0));
    }

    pub fn update_with_scalar_mul(&mut self, value: Scalar){
        self.0.update_with_scalar_mul(Cleartext(value));
    }

    pub fn update_with_add(&mut self, ct: LWECiphertext) {
        self.0.update_with_add(&ct.0);
    }

    pub fn update_with_sub(&mut self, ct: LWECiphertext) {
        self.0.update_with_sub(&ct.0);
    }

    pub fn update_with_neg(&mut self) {
        self.0.update_with_neg();
    }

}


#[derive(Debug, Clone)]
/// An LWE ciphertext in base 2^32
/// It is a wrapper around `lweCiphertext` from concrete.
pub struct LWECiphertext32(pub(crate) LweCiphertext<Vec<u32>>);

impl LWECiphertext32 {
    pub fn allocate(size: LweSize) -> LWECiphertext32 {
        LWECiphertext32(LweCiphertext::allocate(0u32, size))
    }

    /// Return the length of the mask + 1 for the body.
    pub fn lwe_size(&self) -> LweSize {
        self.0.lwe_size()
    }

    pub fn get_body(&self) -> &LweBody<u32> {
        self.0.get_body()
    }

    pub fn get_mask(&self) -> LweMask<&[u32]> {
        self.0.get_mask()
    }

    pub fn get_mut_mask(&mut self) -> LweMask<&mut [u32]> {
        self.0.get_mut_mask()
    }

    pub fn get_mut_body(&mut self) -> &mut LweBody<u32> {
        self.0.get_mut_body()
    }

    pub fn clear(&mut self) {
        self.0.as_mut_tensor().fill_with(|| 0u32);
    }

    pub fn update_with_scalar_mul(&mut self, value: u32){
        self.0.update_with_scalar_mul(Cleartext(value));
    }
    pub fn update_with_add(&mut self, ct: LWECiphertext32) {
        self.0.update_with_add(&ct.0);
    }

    pub fn update_with_sub(&mut self, ct: LWECiphertext32) {
        self.0.update_with_sub(&ct.0);
    }

    pub fn update_with_neg(&mut self) {
        self.0.update_with_neg();
    }

}


#[derive(Debug, Clone)]
/// A LWE secret key
/// It is a wrapper around `LweSecretKey` from concrete.
pub struct LWESecretKey(pub(crate) LweSecretKey<BinaryKeyKind, Vec<Scalar>>);

impl LWESecretKey {
    /// Generate a secret key where the coefficients are binary.
    pub fn generate_binary(
        lwe_dimension: LweDimension,
        generator: &mut SecretRandomGenerator,
    ) -> Self {
        LWESecretKey(LweSecretKey::generate_binary(lwe_dimension, generator))
    }

    pub fn encrypt_lwe(
        &self,
        output: &mut LWECiphertext,
        pt: &Plaintext<Scalar>,
        noise_parameters: impl DispersionParameter,
        generator: &mut EncryptionRandomGenerator,
    ) {
        self.0
            .encrypt_lwe(&mut output.0, pt, noise_parameters, generator);
    }

    pub fn binary_encrypt_lwe(
        &self,
        output: &mut LWECiphertext,
        pt: &Plaintext<Scalar>,
        ctx: &mut Context,
    ) {
        let mut encoded_pt = pt.clone();
        binary_encode(&mut encoded_pt.0);
        self.encrypt_lwe(output, &encoded_pt, ctx.std, &mut ctx.encryption_generator);
    }

    pub fn decrypt_lwe(&self, output: &mut Plaintext<Scalar>, ct: &LWECiphertext) {
        self.0.decrypt_lwe(output, &ct.0);
    }

    pub fn decrypt_wrapping_lwe(
        &self,
        pt: &mut Plaintext<Scalar>,
        encrypted: &LWECiphertext,
    ) {
        
        self.0.decrypt_wrapping_lwe(pt, &encrypted.0);
    }

    /// Decrypt a LWE ciphertext and then decode.
    pub fn binary_decrypt_lwe(&self, pt: &mut Plaintext<Scalar>, encrypted: &LWECiphertext) {
        self.decrypt_lwe(pt, encrypted);
        binary_decode(&mut pt.0);
    }

    pub fn to_rlwe_sk(&self) -> RLWESecretKey {
        let mut sk = RLWESecretKey::zero(PolynomialSize(self.0.key_size().0));
        sk.fill_with_copy(self.0.as_tensor());
        sk
    }

    pub fn from_concrete_sk(
        concrete_sk: LweSecretKey<BinaryKeyKind,Vec<Scalar>>) -> LWESecretKey {
            LWESecretKey(concrete_sk)
    }

    pub fn key_size(&self) -> LweDimension {
        self.0.key_size()
    }

    pub fn as_mut_tensor(&mut self) -> &mut Tensor<Vec<Scalar>>{
        self.0.as_mut_tensor()
    }

    pub fn as_tensor(& self) -> &Tensor<Vec<Scalar>>{
        self.0.as_tensor()
    }

    pub fn zero(glwe_size: usize) -> Self {
        LWESecretKey(
            LweSecretKey::binary_from_container(vec![Scalar::zero(); glwe_size])
        )
    }

    pub fn generate_mask(&self, encrypted: &mut LWEPublicKey, ctx: &mut Context) {
        for i in 0..ctx.m {
            self.0.fill_mask(encrypted.0.get_mut(i).unwrap(), &mut ctx.encryption_generator);
        }
    }

    /// fills the mask of self with the mask of encrypted
    pub fn fill_lwe_pk(&self, encrypted: &mut LWEPublicKey,
        noise_parameter: impl DispersionParameter, generator: &mut EncryptionRandomGenerator, m: usize) {
        for i in 0..m {
            self.0.encrypt_lwe_mask(&mut encrypted.0.get_mut(i).unwrap(), noise_parameter, generator);
        }
    }

}
#[derive(Debug, Clone)]
pub struct LWEPublicKey(pub(crate) Vec<LweCiphertext<Vec<Scalar>>>);

impl LWEPublicKey {
    pub fn allocate(n: usize, m: usize) -> LWEPublicKey {
        let mut lwe_list: Vec<LweCiphertext<Vec<Scalar>>> = Vec::new();
        for _i in 0..m {
            let ct = LweCiphertext::from_container(vec![Scalar::zero(); n]);
            lwe_list.push(ct);
        }
        LWEPublicKey(lwe_list)
    }

    pub fn get_size(&self) -> usize {
        self.0.len()
    }
    
    pub fn update_with_global_mask(&mut self, global_pk: &LWEPublicKey, ctx: &Context) {
        for i in 0..ctx.m {
            self.0.get_mut(i).unwrap().get_mut_mask().as_mut_tensor().fill_with_copy(global_pk.0.get(i).unwrap().get_mask().as_tensor());
            //println!("update_with_global_mask: {:?}",global_pk.0.get(i).unwrap().get_mask().as_tensor().len());
        }
    }

    pub fn update_body_with_pk(&mut self, pk: &LWEPublicKey, ctx: &Context) {
        
        for i in 0..ctx.m {
            //println!("update_body_with_pk {:?}", self.0.get_mut(i).unwrap().get_mut_body().0);
            //println!("update_body_with_pk addition term {:?}", pk.0.get(i).unwrap().get_body().0);
            let sum = self.0.get_mut(i).unwrap().get_mut_body().0.wrapping_add(pk.0.get(i).unwrap().get_body().0);
            self.0.get_mut(i).unwrap().get_mut_body().0 = sum;
            //println!("update_body_with_pk 2 {:?}", self.0.get_mut(i).unwrap().get_mut_body().0);
        }
    }

    pub fn get_body(&mut self, m: usize) -> Vec<Scalar> {
        let mut body: Vec<Scalar> = Vec::new();
        for i in 0..m {
            body.push(self.0.get_mut(i).unwrap().get_body().0);

        }
        body
    }

    pub fn encrypt_lwe(&self, encrypted: &mut LWECiphertext, pt: &Plaintext<Scalar>, generator: &mut SecretRandomGenerator, m: usize) {

        assert_eq!(m,self.get_size());

        //println!("Encrypted before: {:?}", encrypted);

        if m == Scalar::one() as usize{
            encrypted.0.get_mut_mask().as_mut_tensor().update_with_wrapping_add(self.0.get(0).unwrap().get_mask().as_tensor());
            //encrypted.0.get_mut_body().as_mut_polynomial().update_with_wrapping_add(&self.0.get(0).unwrap().get_body().as_polynomial());
            let sum = encrypted.0.get_body().0.wrapping_add(self.0.get(0).unwrap().get_body().0); 
            encrypted.0.get_mut_body().0 = sum;
        } else {
            let r = LWESecretKey::generate_binary(LweDimension(self.get_size() - 1), generator);
            //println!("Random vector = {:?}", r.as_tensor());
        
            let mut sum = Scalar::zero();
            
            for (i, val) in r.0.as_tensor().iter().enumerate() {
                if *val == Scalar::one() {
                    encrypted.0.get_mut_mask().as_mut_tensor().update_with_wrapping_add(self.0.get(i).unwrap().get_mask().as_tensor());
                    sum = sum.wrapping_add(encrypted.0.get_body().0.wrapping_add(self.0.get(i).unwrap().get_body().0)); 
                }
            }

            encrypted.0.get_mut_body().0 = sum;

        }

        //println!("Encrypted after adding mask: {:?}", encrypted);

        let sum = encrypted.0.get_body().0.wrapping_add(pt.0);
        encrypted.0.get_mut_body().0 = sum;

        //println!("Encrypted after adding body: {:?}", encrypted);

    }


    pub fn compare_mask(&self, pk: &LWEPublicKey, m: usize) {
        for i in 0..m {
            assert_eq!(self.0.get(i).unwrap().get_mask(),pk.0.get(i).unwrap().get_mask());
        }
    }

    pub fn get_numberof_mask(&self) -> usize {
        self.0.get(0).unwrap().get_mask().as_tensor().len()
    }



}


#[derive(Debug, Clone)]
/// An LWE secret key.
pub struct LWEKeyswitchKey(pub(crate) LweKeyswitchKey<Vec<Scalar>>);

impl LWEKeyswitchKey {

    pub fn allocate(input_size: LweDimension, output_size: LweDimension, ctx: &Context) -> LWEKeyswitchKey {
        LWEKeyswitchKey(LweKeyswitchKey::allocate(0 as Scalar,ctx.ks_level_count,ctx.ks_base_log,input_size,output_size))
    }

    pub fn fill_with_keyswitching_key(&mut self, input_key: &LWESecretKey, output_key: &LWESecretKey, ctx: &mut Context) {
        self.0.fill_with_keyswitch_key(&input_key.0, &output_key.0, ctx.std_rlwe, &mut ctx.encryption_generator);
    }
    
    pub fn keyswitch_ciphertext(&self, after: &mut LWECiphertext, before: &LWECiphertext) {
        self.0.keyswitch_ciphertext(&mut after.0, &before.0);
    }

}

pub fn constant_sample_extract<>(
    lwe: &mut LWECiphertext,
    glwe: &RLWECiphertext,
) {
    // We extract the mask  and body of both ciphertexts
    let (mut body_lwe, mut mask_lwe) = lwe.0.get_mut_body_and_mask();
    let (body_glwe, mask_glwe) = glwe.0.get_body_and_mask();

    //println!("MASK SIZE: {:?}",mask_lwe.mask_size());
    // We construct a polynomial list from the lwe mask

    let mut mask_lwe_poly = PolynomialList::from_container(
        mask_lwe.as_mut_tensor().as_mut_slice(),
        glwe.polynomial_size(),
    );
    
    
    // We copy the mask values with the proper ordering and sign
    for (mut mask_lwe_polynomial, mask_glwe_polynomial) in mask_lwe_poly
        .polynomial_iter_mut()
        .zip(mask_glwe.as_polynomial_list().polynomial_iter())
    {
        for (lwe_coeff, glwe_coeff) in mask_lwe_polynomial
            .coefficient_iter_mut()
            .zip(mask_glwe_polynomial.coefficient_iter().rev())
        {
            *lwe_coeff = Scalar::zero().wrapping_sub(*glwe_coeff);
        }
    }

    mask_lwe_poly.update_with_wrapping_monic_monomial_mul(MonomialDegree(1));

    // We set the body
    body_lwe.0 = *body_glwe.as_tensor().get_element(0);
    
}



#[derive(Debug, Clone)]
/// An LWE to RLWE key switching key.
pub struct LWEtoRLWEKeyswitchKey {
    // TODO At the moment it's a list of full RGSW ciphertexts,
    // we should remove half of the rows.
    pub(crate) inner: Vec<RGSWCiphertext>,
}

impl LWEtoRLWEKeyswitchKey {
    pub fn allocate(ctx: &Context) -> LWEtoRLWEKeyswitchKey {
        LWEtoRLWEKeyswitchKey {
            inner: vec![
                RGSWCiphertext::allocate(ctx.poly_size, ctx.rgsw_base_log, ctx.rgsw_level_count);
                ctx.poly_size.0
            ],
        }
    }

    pub fn fill_with_keyswitching_key(&mut self, sk: &LWESecretKey, ctx: &mut Context) {
        assert_eq!(ctx.poly_size.0, sk.key_size().0);
        let rlwe_sk = sk.to_rlwe_sk();
        self.inner = vec![];
        for s in sk.0.as_tensor().iter() {
            // TODO what is the decomposition parameters?
            let mut rgsw_ct =
                RGSWCiphertext::allocate(ctx.poly_size, ctx.ks_base_log, ctx.ks_level_count);
            rlwe_sk.encrypt_constant_rgsw(&mut rgsw_ct, &Plaintext(*s), ctx);
            self.inner.push(rgsw_ct);
        }
    }
}

pub fn conv_lwe_to_rlwe(
    ksks: &LWEtoRLWEKeyswitchKey,
    lwe: &LWECiphertext,
    ctx: &Context,
) -> RLWECiphertext {
    let mut out = RLWECiphertext::allocate(ctx.poly_size);

    for (ksk, a) in ksks.inner.iter().zip(lwe.get_mask().as_tensor().iter()) {
        // Setup decomposition stuff
        // TODO what parameters for decomposition?
        let decomposer = SignedDecomposer::new(ctx.ks_base_log, ctx.ks_level_count);
        let closest = decomposer.closest_representable(*a);
        let decomposer_iter = decomposer.decompose(closest);

        // Get an iterator of every second row
        // we only need every second ciphertext since that is
        // a valid RLWE ciphertext in a RGSW
        let ksk_iter = ksk.0.level_matrix_iter().rev().map(|m| {
            let ct = m.row_iter().nth(1).unwrap().into_glwe();
            // TODO avoid copying
            let mut out = RLWECiphertext::allocate(ctx.poly_size);
            out.update_mask_with_add(&ct.get_mask().as_polynomial_list().get_polynomial(0));
            out.update_body_with_add(&ct.get_body().as_polynomial());
            out
        });

        for (mut ct, decomposed_a) in ksk_iter.zip(decomposer_iter) {
            mul_const(
                &mut ct
                    .get_mut_mask()
                    .as_mut_polynomial_list()
                    .get_mut_polynomial(0)
                    .as_mut_tensor(),
                decomposed_a.value(),
            );
            mul_const(
                &mut ct.get_mut_body().as_mut_polynomial().as_mut_tensor(),
                decomposed_a.value(),
            );
            out.get_mut_mask()
                .as_mut_polynomial_list()
                .get_mut_polynomial(0)
                .update_with_wrapping_sub(&ct.get_mask().as_polynomial_list().get_polynomial(0));
            out.get_mut_body()
                .as_mut_polynomial()
                .update_with_wrapping_sub(&ct.get_body().as_polynomial());
        }
    }

    let b_poly = {
        let mut v = vec![Scalar::zero(); ctx.poly_size.0];
        v[0] = lwe.get_body().0;
        Polynomial::from_container(v)
    };

    out.get_mut_body()
        .as_mut_polynomial()
        .update_with_wrapping_add(&b_poly);
    out
}

pub fn compute_noise_lwe(sk: &LWESecretKey, ct: &LWECiphertext, encoded_ptxt: Scalar) -> f64 
{
    // pt = b - a*s = Delta*m + e
    let mut pt = Plaintext(Scalar::zero());
    sk.decrypt_wrapping_lwe(&mut pt, ct);

    //println!("Delta m + e = {:?}",pt);
    //println!("Encoding: {:?}", encoded_ptxt);

    // pt = pt - Delta*m = e (encoded_ptxt is Delta*m)
    let error = pt.0.wrapping_sub(encoded_ptxt) as SignedScalar;

    //println!("Error: {:?}", error);
    ((error as SignedScalar).abs() as f64).log2()
}


//Function to bootstrap a LWE ciphertexts. It can use secret key material for debugging purpuses.
pub fn bootstrap_debug (
    lwe_in: &LWECiphertext,
    accumulator: &mut RLWECiphertext,
    accumulator_test: &mut RLWECiphertext,
    expected_rotation: usize,
    bootstrappingkeys: Vec<Vec<Vec<RGSWCiphertext>>>,
    mut ctx: &mut Context,
    global_sk: &RLWESecretKey,
    global_lwe_sk: &LWESecretKey,
) -> RLWECiphertext
{

    let (lwe_body, lwe_mask) = lwe_in.0.get_body_and_mask();

    let rotation = pbs_modulus_switch(
        lwe_body.0,
        ctx.poly_size,
        ModulusSwitchOffset(0),
        LutCountLog(0));

    accumulator.0.as_mut_polynomial_list()
        .update_with_wrapping_monic_monomial_div(rotation);

    for (index, element) in lwe_mask.mask_element_iter().enumerate() {

        let mut addition = RGSWCiphertext::allocate(ctx.poly_size, ctx.rgsw_base_log, ctx.rgsw_level_count);
    
        
        let pbs_switch = pbs_modulus_switch(*element,ctx.poly_size,ModulusSwitchOffset(0),LutCountLog(0)).0;

        let bsk = bootstrappingkeys.get(index).unwrap();
        
        for party in 1..ctx.k+1 { 

            /* 
            let mut aux = bsk2[party].clone();
            let degree = MonomialDegree(pbs_switch*party);
            bsk2[party].product_monomial(&mut aux, degree);
            let mut pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
            global_sk.decrypt_wrapping_rlwe(&mut pt, &aux.get_nth_row(1));
            poly_decode(&mut pt.as_mut_polynomial(), &ctx);
            println!("AUX: ({:?}): {:?}", index , pt);

            addition.add_ciphertext(&aux);
            addition.sub_ciphertext(&bsk2[party]);
            */
            

            let degree2 = MonomialDegree(pbs_switch*party);

            let aux2 = bsk[degree2.0%(2*ctx.poly_size.0)+1][party].clone();
            

            let mut pt2 = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());

            global_sk.decrypt_wrapping_rlwe(&mut pt2, &aux2.get_nth_row(1));
            poly_decode(&mut pt2.as_mut_polynomial(), &ctx);
            println!("AUX2: ({:?}): {:?}", index , pt2);

            //assert_eq!(pt,pt2);

            addition.add_ciphertext(&aux2);
            addition.sub_ciphertext(&bsk[0][party]);
            

        }

        let mut pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        global_sk.decrypt_wrapping_rlwe(&mut pt, &addition.get_nth_row(1));
        poly_decode(&mut pt.as_mut_polynomial(), &ctx);
        //println!("Addition: ({:?}): {:?}", index , pt); 

        let mut pt_monomial = RLWESecretKey::plaintext_index(0, 1, &mut ctx);
        let degree = MonomialDegree(pbs_modulus_switch(*element,ctx.poly_size,ModulusSwitchOffset(0),LutCountLog(0)).0);
        
        let sj = *global_lwe_sk.as_tensor().iter().nth(index).unwrap() as usize;
        poly_decode(&mut pt.as_mut_polynomial(), &ctx);
        //println!("sj * degree {:?}, Sj: {:?},{:?}",degree.0*sj, sj, index);
        
        pt_monomial.as_mut_polynomial().update_with_wrapping_monic_monomial_mul(MonomialDegree(degree.0*sj));


        poly_decode(&mut pt_monomial.as_mut_polynomial(),ctx);
        //println!("Monomial: {:?}", pt_monomial);

        let mut aux = RLWECiphertext::allocate(ctx.poly_size);

        addition.external_product(&mut aux, &accumulator);
        accumulator.update_with_add(&aux);

        let mut pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        global_sk.decrypt_wrapping_rlwe(&mut pt, &accumulator);
        poly_decode(&mut pt.as_mut_polynomial(), &ctx);
        //println!("After rotation {:?}: {:?}", index, pt); 

    }

    //println!("Local acc: {:?} \n", accumulator_test.get_body().as_tensor());

    accumulator_test.0.as_mut_polynomial_list()
        .update_with_wrapping_monic_monomial_div(MonomialDegree(expected_rotation));

    poly_decode_gate(&mut accumulator_test.get_mut_body().as_mut_polynomial());

    //println!("Expected rotation: {:?}", expected_rotation);
    //println!("Local acc after rotation ({:?}): {:?} \n", expected_rotation, accumulator_test.get_body().as_tensor());

    RLWECiphertext(accumulator.get_glwe_copy(&mut ctx))
}

//Function to bootstrap a LWE ciphertext.
pub fn bootstrap (
    lwe_in: &LWECiphertext,
    accumulator: &mut RLWECiphertext,
    bootstrappingkeys: Vec<Vec<RGSWCiphertext>>,
    mut ctx: &mut Context,
) -> RLWECiphertext
{

    let (lwe_body, lwe_mask) = lwe_in.0.get_body_and_mask();

    let rotation = pbs_modulus_switch(
        lwe_body.0,
        ctx.poly_size,
        ModulusSwitchOffset(0),
        LutCountLog(0));

    accumulator.0.as_mut_polynomial_list()
        .update_with_wrapping_monic_monomial_div(rotation);


    for (index, element) in lwe_mask.mask_element_iter().enumerate() {

        let mut addition = RGSWCiphertext::allocate(ctx.poly_size, ctx.rgsw_base_log, ctx.rgsw_level_count);
        let pbs_switch = pbs_modulus_switch(*element,ctx.poly_size,ModulusSwitchOffset(0),LutCountLog(0)).0;
        let bsk = bootstrappingkeys.get(index).unwrap();
        
        for party in 1..ctx.k+1 { 

            let mut aux = bsk[party].clone();
            let degree = MonomialDegree(pbs_switch*party);
            bsk[party].product_monomial(&mut aux, degree);
            addition.add_ciphertext(&aux);
            addition.sub_ciphertext(&bsk[party]);

        }

        let mut aux = RLWECiphertext::allocate(ctx.poly_size);
        addition.external_product(&mut aux, &accumulator);
        accumulator.update_with_add(&aux);

    }

    RLWECiphertext(accumulator.get_glwe_copy(&mut ctx))
}



//Function to bootstrap a LWE ciphertext in the Fourier domain.
pub fn bootstrap_fourier (
    lwe_in: &LWECiphertext,
    accumulator: &mut RLWECiphertext,
    bootstrappingkeys: Vec<Vec<FourierRGSWCiphertext>>,
    monomials_fourier: &Vec<FourierPolynomial<AlignedVec<Complex64>>>,
    mut ctx: &mut Context,
) -> RLWECiphertext
{
    let (lwe_body, lwe_mask) = lwe_in.0.get_body_and_mask();
    let rotation = pbs_modulus_switch(
        lwe_body.0,
        ctx.poly_size,
        ModulusSwitchOffset(0),
        LutCountLog(0));

    accumulator.0.as_mut_polynomial_list()
        .update_with_wrapping_monic_monomial_div(rotation);

    
    for (index, element) in lwe_mask.mask_element_iter().enumerate() {

        let mut addition = FourierRGSWCiphertext::allocate(ctx.poly_size, ctx.rgsw_base_log, ctx.rgsw_level_count);
        let pbs_switch = pbs_modulus_switch(*element,ctx.poly_size,ModulusSwitchOffset(0),LutCountLog(0)).0;
        let bsk = bootstrappingkeys.get(index).unwrap();
        
        for party in 1..ctx.k+1 { 

            let mut aux = bsk[party].clone();
            //let now = Instant::now();
            let degree = MonomialDegree(pbs_switch*party);
            let monomial = monomials_fourier.get(degree.0%2*ctx.poly_size.0).unwrap();
            //let now = Instant::now();
            bsk[party].product_monomial(&mut aux, monomial);
            //println!("Time of monomials fourier: {} micro seconds", now.elapsed().as_micros());
            //let now = Instant::now();
            addition.add_ciphertext(&aux);
            //println!("Time of add fourier: {} micro seconds", now.elapsed().as_micros());
            //let now = Instant::now();
            addition.sub_ciphertext(&bsk[party]);
            //println!("Time of sub fourier: {} micro seconds", now.elapsed().as_micros());
            

        }

        let mut aux = RLWECiphertext::allocate(ctx.poly_size);
        //let now = Instant::now();
        addition.external_product(&mut aux, &accumulator);
        //println!("Time of external product fourier: {} micro seconds", now.elapsed().as_micros());
        //let now = Instant::now();
        accumulator.update_with_add(&aux);
        //println!("Time of update accumulator: {} micro seconds", now.elapsed().as_micros());

    }

    RLWECiphertext(accumulator.get_glwe_copy(&mut ctx))
}

//Function to bootstrap a LWE ciphertexts in the Fourier domain. It can use secret key material for debugging purpuses.
pub fn bootstrap_fourier_debug (
    lwe_in: &LWECiphertext,
    accumulator: &mut RLWECiphertext,
    accumulator_test: &mut RLWECiphertext,
    expected_rotation: usize,
    bootstrappingkeys: Vec<Vec<Vec<FourierRGSWCiphertext>>>,
    mut ctx: &mut Context,
    global_sk: &RLWESecretKey,
    global_lwe_sk: &LWESecretKey,
) -> RLWECiphertext
{

    let (lwe_body, lwe_mask) = lwe_in.0.get_body_and_mask();

    let rotation = pbs_modulus_switch(
        lwe_body.0,
        ctx.poly_size,
        ModulusSwitchOffset(0),
        LutCountLog(0));

    accumulator.0.as_mut_polynomial_list()
        .update_with_wrapping_monic_monomial_div(rotation);

    for (index, element) in lwe_mask.mask_element_iter().enumerate() {

        let mut addition = FourierRGSWCiphertext::allocate(ctx.poly_size, ctx.rgsw_base_log, ctx.rgsw_level_count);
        
        let pbs_switch = pbs_modulus_switch(*element,ctx.poly_size,ModulusSwitchOffset(0),LutCountLog(0)).0;
        
        let bsk = bootstrappingkeys.get(index).unwrap();
        
        for party in 1..ctx.k+1 { 

            

            let mut pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());

            //global_sk.decrypt_wrapping_rlwe(&mut pt, &ComplexRLWECiphertext(&aux.0.as_glwe_list().ciphertext_iter().nth(1).unwrap()));
            poly_decode(&mut pt.as_mut_polynomial(), &ctx);
            println!("AUX (before): ({:?}): {:?}", index , pt);
            
            let degree = MonomialDegree(pbs_switch*party);

            let aux = bsk[party][degree.0].clone();

            let mut pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
            let rlwe = RLWECiphertext::allocate(ctx.poly_size);
            //let mut buffers_out = FourierBuffers::new(ctx.poly_size, GlweSize(2));

            //aux.get_nth_row(1).fill_with_backward_fourier(&mut rlwe, &mut buffers_out);

            global_sk.decrypt_wrapping_rlwe(&mut pt, &rlwe);
            poly_decode(&mut pt.as_mut_polynomial(), &ctx);
            println!("AUX: ({:?}): {:?}", index , pt);

            addition.add_ciphertext(&aux);
            addition.sub_ciphertext(&bsk[party][degree.0]);

        }

        let mut pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        let rlwe = RLWECiphertext::allocate(ctx.poly_size);
        //let mut buffers_out = FourierBuffers::new(ctx.poly_size, GlweSize(2));

        //addition.get_nth_row(1).fill_with_backward_fourier(&mut rlwe, &mut buffers_out);

        global_sk.decrypt_wrapping_rlwe(&mut pt, &rlwe);
        poly_decode(&mut pt.as_mut_polynomial(), &ctx);
        println!("Addition: ({:?}): {:?}", index , pt); 

        let mut pt_monomial = RLWESecretKey::plaintext_index(0, 1, &mut ctx);
        let degree = MonomialDegree(pbs_modulus_switch(*element,ctx.poly_size,ModulusSwitchOffset(0),LutCountLog(0)).0);
        
        let sj = *global_lwe_sk.as_tensor().iter().nth(index).unwrap() as usize;
        poly_decode(&mut pt.as_mut_polynomial(), &ctx);
        println!("sj * degree {:?}, Sj: {:?},{:?}",degree.0*sj, sj, index);
        
        pt_monomial.as_mut_polynomial().update_with_wrapping_monic_monomial_mul(MonomialDegree(degree.0*sj));
        let deg = pt_monomial.as_polynomial().coefficient_iter().position(|&x| x != 0).unwrap();
        println!("Degree of monomial: {:?}", deg);


        poly_decode(&mut pt_monomial.as_mut_polynomial(),ctx);
        println!("Monomial: {:?}", pt_monomial);

        let mut aux = RLWECiphertext::allocate(ctx.poly_size);

        addition.external_product(&mut aux, &accumulator);
        accumulator.update_with_add(&aux);

        let mut pt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        global_sk.decrypt_wrapping_rlwe(&mut pt, &accumulator);
        println!("After rotation {:?}: {:?}", index, pt); 

    }

    println!("Local acc: {:?} \n", accumulator_test.get_body().as_tensor());

    accumulator_test.0.as_mut_polynomial_list()
        .update_with_wrapping_monic_monomial_div(MonomialDegree(expected_rotation));

    poly_decode_gate(&mut accumulator_test.get_mut_body().as_mut_polynomial());

    println!("Expected rotation: {:?}", expected_rotation);
    println!("Local acc after rotation ({:?}): {:?} \n", expected_rotation, accumulator_test.get_body().as_tensor());

    RLWECiphertext(accumulator.get_glwe_copy(&mut ctx))
}

///Outputs a vector [RGSW(0), ... , RGSW(1), ..., RGSW(0)] depending on the values of c. That is, the ciphertext RGSW(1)
/// will be placed in the i-th position where i corresponds to the number of encryptions of 1 in c.
pub fn homomorphic_indicator(global_rlwe_sk: &RLWESecretKey, c: Vec<RGSWCiphertext>, mut ctx: &mut Context) -> Vec<RGSWCiphertext> {
    
    let mut l: Vec<RGSWCiphertext> = Vec::new();

    let mut ct = RGSWCiphertext::allocate(ctx.poly_size, ctx.rgsw_base_log, ctx.rgsw_level_count);
    let ptxt = RLWESecretKey::plaintext_index(0, 1, &mut ctx);
    global_rlwe_sk.encrypt_wrapping_rgsw(&mut ct, &ptxt, &mut ctx);
    l.push(ct);

    for _i in 0..ctx.k {
        let mut ct = RGSWCiphertext::allocate(ctx.poly_size, ctx.rgsw_base_log, ctx.rgsw_level_count);
        let ptxt = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
        global_rlwe_sk.encrypt_wrapping_rgsw(&mut ct, &ptxt, &mut ctx);
        l.push(ct);
    }

    for j in 0..ctx.k {
        let mut l_prime = l.clone();

        let mut ct_one = RGSWCiphertext::allocate(ctx.poly_size, ctx.rgsw_base_log, ctx.rgsw_level_count);
        let ptxt = RLWESecretKey::plaintext_index(0, 1, &mut ctx);
        global_rlwe_sk.encrypt_wrapping_rgsw(&mut ct_one, &ptxt, &mut ctx);
        //global_rlwe_sk.trivial_encrypt_wrapping_rgsw(&mut ct_one, &ptxt, &mut ctx);
        ct_one.sub_ciphertext(&c[j]);

        let mut ct = RGSWCiphertext::allocate(ctx.poly_size, ctx.rgsw_base_log, ctx.rgsw_level_count);
        ct_one.internal_product(&mut ct, &l[0], &mut ctx);

        l_prime[0] = ct;
        
        for i in 1..ctx.k+1 {
            let mut aux = l[i-1].clone();
            aux.sub_ciphertext(&l[i]); 

            let mut ct = RGSWCiphertext::allocate(ctx.poly_size, ctx.rgsw_base_log, ctx.rgsw_level_count);
            c[j].internal_product(&mut ct, &aux, &mut ctx);

            ct.add_ciphertext(&l[i]);

            l_prime[i] = ct;

        }

        l = l_prime;
        
    }

    l

}