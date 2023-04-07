use bitvec::macros::internal::funty::Integral;
use concrete_commons::numeric::CastInto;
use concrete_commons::parameters::LweDimension;
use concrete_core::backends::core::private::crypto::encoding::PlaintextList;
use concrete_core::backends::core::private::crypto::lwe::LweBody;

use concrete_core::backends::core::private::math::tensor::AsMutTensor;
use concrete_core::backends::core::private::math::tensor::AsRefTensor;

use mfhebrk::decode_gate;
use mfhebrk::decode_gate32;
use mfhebrk::encode_accumulator;
use mfhebrk::encode_accumulator32;
use mfhebrk::encode_gate;
use mfhebrk::encode_gate32;
use mfhebrk::lwe::LWECiphertext;
use mfhebrk::lwe::LWECiphertext32;
use mfhebrk::lwe::LWEKeyswitchKey;
use mfhebrk::lwe::LWEPublicKey;
use mfhebrk::lwe::LWESecretKey;
use mfhebrk::lwe::compute_noise_lwe;
use mfhebrk::lwe::homomorphic_indicator;
use mfhebrk::plaintext_index;
use mfhebrk::poly_encode;

use mfhebrk::rlwe::RLWEPublicKey;
use mfhebrk::lwe::constant_sample_extract;
use mfhebrk::Scalar;

use concrete_core::backends::core::private as ccore;
use mfhebrk::Context;
use mfhebrk::rgsw;
use mfhebrk::rlwe;
use crate::rgsw::RGSWCiphertext;
use crate::rlwe::RLWECiphertext;
use num_traits::identities::One;
use num_traits::identities::Zero;
use crate::ccore::crypto::encoding::Plaintext;
use mfhebrk::rlwe::RLWESecretKey;

use concrete_core::prelude::*;

use mfhebrk::lwe::bootstrap;

extern crate concrete_boolean;

extern crate rand;

/// We provide an example of the blind rotation set up and the computation of a NAND gate using (multiparty) TFHE bootstrapping
fn main() {

    //Initialize the context
    let mut ctx = Context::default();

    //PUBLIC, SECRET AND KEY SWITCHING KEYS SETUP

    let mask_lwe_sk = LWESecretKey::generate_binary(LweDimension(ctx.glwe_size.0), &mut ctx.secret_generator);
    let mask_rlwe_sk = RLWESecretKey::generate_binary(ctx.poly_size, &mut ctx.secret_generator);
    
    let mut global_rlwe_pk = RLWEPublicKey::allocate(ctx.poly_size, GlweSize(2), ctx.m);
    let mut global_rlwe_sk = RLWESecretKey::zero(ctx.poly_size);

    let mut global_lwe_pk = LWEPublicKey::allocate(ctx.glwe_size.0, ctx.m);
    let mut global_lwe_sk = LWESecretKey::zero(ctx.glwe_size.0 - 1);

    //We generate the global masks for the global public keys
    mask_rlwe_sk.generate_mask(&mut global_rlwe_pk, &mut ctx);
    mask_lwe_sk.generate_mask(&mut global_lwe_pk, &mut ctx);

    let mut lwe_sk_list: Vec<LWESecretKey> = Vec::new();
    let mut rlwe_sk_list: Vec<RLWESecretKey> = Vec::new();
    let mut rlwe_pk_list: Vec<RLWEPublicKey> = Vec::new();
    
    print!("Generating global pk,sk and ksk keys...");

    for _user in 0..ctx.k {
        let lwe_sk = LWESecretKey::generate_binary(LweDimension(ctx.glwe_size.0 - 1), &mut ctx.secret_generator);
        let rlwe_sk = RLWESecretKey::generate_binary(ctx.poly_size, &mut ctx.secret_generator);
        let mut rlwe_pk = RLWEPublicKey::allocate(ctx.poly_size, GlweSize(2), ctx.m);
        let mut lwe_pk = LWEPublicKey::allocate(ctx.glwe_size.0, ctx.m);

        rlwe_pk.update_with_global_mask(&global_rlwe_pk, &ctx);
        lwe_pk.update_with_global_mask(&global_lwe_pk, &ctx);
        
        global_rlwe_pk.compare_mask(&rlwe_pk, ctx.m);
        global_lwe_pk.compare_mask(&lwe_pk, ctx.m);

        rlwe_sk.fill_rlwe_pk(&mut rlwe_pk,  ctx.std_rlwe, &mut ctx.encryption_generator, ctx.m); //fill the body of the individual rlwe public key
        lwe_sk.fill_lwe_pk(&mut lwe_pk, ctx.std_lwe, &mut ctx.encryption_generator, ctx.m); //fill the body of the individual lwe public key

        global_rlwe_pk.update_body_with_pk(&rlwe_pk, &ctx);
        global_rlwe_sk.as_mut_tensor().update_with_wrapping_add(rlwe_sk.as_tensor());

        global_lwe_pk.update_body_with_pk(&lwe_pk, &ctx);
        global_lwe_sk.as_mut_tensor().update_with_wrapping_add(lwe_sk.as_tensor());

        lwe_sk_list.push(lwe_sk);
        rlwe_sk_list.push(rlwe_sk);
        rlwe_pk_list.push(rlwe_pk);

    }
    
    let mut ksk_lwe = LWEKeyswitchKey::allocate(LweDimension(ctx.poly_size.0),LweDimension(ctx.glwe_size.0 - 1), &ctx);
    let sk_extraction = LWESecretKey::from_concrete_sk(global_rlwe_sk.to_lwe_secretkey(&mut ctx));

    ksk_lwe.fill_with_keyswitching_key(&sk_extraction, &global_lwe_sk, &mut ctx);

    println!("Done");

    //GENERATION OF BOOTSTRAPPING KEYS

    print!("Generating global bootstrapping keys...");

    let mut bootstrappingkeys: Vec<Vec<RGSWCiphertext>> = Vec::new();

    for pos in 0..ctx.glwe_size.0 - 1 {
    
        let mut c: Vec<RGSWCiphertext> = Vec::new();
        let mut p: Vec<PlaintextList<Vec<Scalar>>> = Vec::new();
        
        for user in 0..ctx.k {

            let mut ct = RGSWCiphertext::allocate(ctx.poly_size, ctx.rgsw_base_log, ctx.rgsw_level_count);
            
            let sk_user = lwe_sk_list[user].clone();
            let sk_vec = sk_user.as_tensor().as_container().to_vec();
        
            if sk_vec.get(pos) == Some(&Scalar::zero()) {
                let mut ptxt = plaintext_index(0, 0, &mut ctx);
                p.push(ptxt.clone());
                global_rlwe_sk.encrypt_wrapping_rgsw(&mut ct, &ptxt, &mut ctx);
                poly_encode(&mut ptxt.as_mut_polynomial(),&ctx);
                //let noise_bski = compute_noise_rgsw(&ct, &ptxt, &global_rlwe_sk, &ctx);
            } else {
                let mut ptxt = plaintext_index(0, 1, &mut ctx);
                p.push(ptxt.clone());
                global_rlwe_sk.encrypt_wrapping_rgsw(&mut ct, &ptxt, &mut ctx);
                poly_encode(&mut ptxt.as_mut_polynomial(),&ctx);
                //let noise_bski = compute_noise_rgsw(&ct, &ptxt, &global_rlwe_sk, &ctx);
            }
            
            c.push(ct);
        }

        let l = homomorphic_indicator(&global_rlwe_sk, c, &mut ctx);

        bootstrappingkeys.push(l);

    }

    println!("Done");

    //TESTING GATE BOOTSTRAPPING (NAND gate)
    println!("Computing NAND gate...");

    let mut lwe_ct = LWECiphertext::allocate(LweSize(ctx.glwe_size.0));

    let mut lwe_ct1 = LWECiphertext::allocate(LweSize(ctx.glwe_size.0));
    let mut lwe_ct2 = LWECiphertext::allocate(LweSize(ctx.glwe_size.0));

    let mut lwe_fresh = LWECiphertext::allocate(LweSize(ctx.glwe_size.0));
    let mut lwe_out = LWECiphertext::allocate(LweSize(ctx.poly_size.0+1));


    let cleartext1 = Scalar::one();
    let cleartext2 = Scalar::one();

    let cleartext1_u32 = 1u32;
    let cleartext2_u32 = 1u32;

    //AND GATE
    //let mut result = cleartext1 * cleartext2;
    //encode_gate(&mut result, &ctx);


    //NAND GATE
    let mut result = Scalar::one() - cleartext1 * cleartext2;
    encode_gate(&mut result);

    let mut result32 = 1u32 - cleartext1_u32 * cleartext2_u32;
    encode_gate32(&mut result32);

    let mut pt1 = cleartext1.clone();
    let mut pt2 = cleartext2.clone();
    
    //AND GATE CONSTANT
    //let mut pt_constant = Scalar::zero();

    //NAND GATE CONSTANT
    let mut pt_constant = Scalar::one();
    encode_gate(&mut pt_constant);
    //pt_constant = pt_constant.wrapping_neg();

    encode_gate(&mut pt1);
    encode_gate(&mut pt2);
    //encode_gate(&mut pt_constant, &ctx);

    global_lwe_pk.encrypt_lwe(&mut lwe_ct1, &Plaintext(pt1), &mut ctx.secret_generator, ctx.m);
    global_lwe_pk.encrypt_lwe(&mut lwe_ct2, &Plaintext(pt2), &mut ctx.secret_generator, ctx.m);
    global_lwe_pk.encrypt_lwe(&mut lwe_fresh, &Plaintext(pt1), &mut ctx.secret_generator, ctx.m);
    
    //AND GATE
    //lwe_ct.update_with_add(lwe_ct1);
    //lwe_ct.update_with_add(lwe_ct2);

    //let updated_body = LweBody(lwe_ct.get_body().0.wrapping_add(pt_constant));
    //lwe_ct.get_mut_body().clone_from(&updated_body);

    //NAND GATE

    lwe_ct.update_with_sub(lwe_ct1);
    lwe_ct.update_with_sub(lwe_ct2);
    
    let updated_body = LweBody(lwe_ct.get_body().0.wrapping_add(pt_constant));
    lwe_ct.get_mut_body().clone_from(&updated_body);

    let mut accumulator =
            RLWECiphertext::allocate(ctx.poly_size);
    accumulator.get_mut_body().as_mut_tensor().iter_mut().enumerate().for_each(|(_i, a)| {
        //*a = (i as Scalar) *  Scalar::one() << ( ((Scalar::BITS as usize) - (ctx.poly_size.0 as f64).log2() as usize - 1));
        *a = Scalar::one() << ((Scalar::BITS as usize) - 3);
    });

    let output_bootstrap = bootstrap(&lwe_ct, &mut accumulator, bootstrappingkeys, &mut ctx);

    let mut pt_bootstrap = PlaintextList::allocate(Scalar::zero(), ctx.plaintext_count());
    global_rlwe_sk.decrypt_wrapping_rlwe(&mut pt_bootstrap, &output_bootstrap);

    constant_sample_extract(&mut lwe_out, &output_bootstrap);

    let sk_extraction = LWESecretKey::from_concrete_sk(global_rlwe_sk.to_lwe_secretkey(&mut ctx));

    let mut out_plaintext = Plaintext(Scalar::ZERO);

    sk_extraction.decrypt_wrapping_lwe(&mut out_plaintext, &lwe_out);

    encode_accumulator(&mut out_plaintext.0);
    decode_gate(&mut out_plaintext.0);

    let mut switched_ciphertext = LWECiphertext::allocate(LweSize(ctx.glwe_size.0));
    ksk_lwe.keyswitch_ciphertext(&mut switched_ciphertext, &lwe_out);

    let mut out_keyswitch = Plaintext(Scalar::ZERO);
    global_lwe_sk.decrypt_wrapping_lwe(&mut out_keyswitch, &switched_ciphertext);

    encode_accumulator(&mut out_keyswitch.0);
    decode_gate(&mut out_keyswitch.0);

    let output_noise = compute_noise_lwe(&global_lwe_sk, &switched_ciphertext, result);
    println!("Output noise: {:?}", output_noise);
    
    let modulus_switch = |input: Scalar| -> usize {
        let tmp: f64 = (input as f64)/4294967296.0;
        let input_hat: usize = tmp.round().cast_into();
        input_hat
    };

    let mut switched_ciphertext_mod = LWECiphertext32::allocate(LweSize(ctx.glwe_size.0));

    for (i,mask) in switched_ciphertext.get_mask().mask_element_iter().enumerate() {
        *switched_ciphertext_mod.get_mut_mask().as_mut_tensor().get_element_mut(i) = modulus_switch(*mask) as u32;
    }

    switched_ciphertext_mod.get_mut_body().0 = modulus_switch(switched_ciphertext.get_body().0) as u32;

    let mut out_keyswitch_mod = Plaintext(0u32);

    let body = switched_ciphertext_mod.get_body();
    let masks = switched_ciphertext_mod.get_mask();

    out_keyswitch_mod.0 = body.0;

    let sub = masks.as_tensor().fold_with_one(
        global_lwe_sk.as_tensor(),
        0u32,
        |ac, s_i, o_i| ac.wrapping_add(s_i.wrapping_mul((*o_i).try_into().unwrap())),
    );

    out_keyswitch_mod.0 = out_keyswitch_mod.0.wrapping_sub(sub);

    encode_accumulator32(&mut out_keyswitch_mod.0);
    decode_gate32(&mut out_keyswitch_mod.0);
    

}