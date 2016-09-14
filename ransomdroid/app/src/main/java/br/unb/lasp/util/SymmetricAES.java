package br.unb.lasp.util;

import android.util.Log;

import java.security.MessageDigest;
import java.security.SecureRandom;
import java.util.Arrays;

import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.spec.SecretKeySpec;

public class SymmetricAES {

    private static final String TAG = new Object() {
    }.getClass().getName();

    public static SecretKeySpec getRandomSecretKeySpec(String seed) {
        SecretKeySpec keySpec = null;
        try {
            KeyGenerator keygen = KeyGenerator.getInstance("AES");
            SecureRandom random = SecureRandom.getInstance("SHA1PRNG");
            random.setSeed(seed.getBytes("UTF-8"));
            keygen.init(128, random);
            keySpec = new SecretKeySpec((keygen.generateKey()).getEncoded(), "AES");
        } catch (Exception e) {
            Log.e(TAG, e.getMessage(), e);
        }

        return keySpec;
    }

    public static SecretKeySpec getSecretKeySpec(String salt, String password) {
        SecretKeySpec keySpec = null;
        try {
            byte[] key = (salt + password).getBytes("UTF-8");
            MessageDigest sha = MessageDigest.getInstance("SHA-1");
            key = sha.digest(key);
            key = Arrays.copyOf(key, 16); // use only first 128 bit
            keySpec = new SecretKeySpec(key, "AES");
        } catch (Exception e) {
            Log.e(TAG, e.getMessage(), e);
        }

        return keySpec;
    }

    public static byte[] encrypt(byte[] origin, String seed, SecretKeySpec keySpec) {

        // Encode the original data with AES
        byte[] encodedBytes = null;
        try {
            Cipher aesCipher = Cipher.getInstance("AES");
            aesCipher.init(Cipher.ENCRYPT_MODE, keySpec);
            encodedBytes = aesCipher.doFinal(origin);
        } catch (Exception e) {
            Log.e(TAG, e.getMessage(), e);
        }

        return encodedBytes;
    }

    public static byte[] encryptRandom(byte[] origin, String seed, SecretKeySpec keySpec) {

        // Set up secret key spec for 128-bit AES encryption and decryption
        keySpec = getRandomSecretKeySpec(seed);

        // Encode the original data with AES
        byte[] encodedBytes = null;
        try {
            Cipher aesCipher = Cipher.getInstance("AES");
            aesCipher.init(Cipher.ENCRYPT_MODE, keySpec);
            encodedBytes = aesCipher.doFinal(origin);
        } catch (Exception e) {
            Log.e(TAG, e.getMessage(), e);
        }

        return encodedBytes;
    }

    public static byte[] decrypt(byte[] origin, SecretKeySpec keySpec) {

        byte[] decodedBytes = null;
        try {
            Cipher aesCipher = Cipher.getInstance("AES");
            aesCipher.init(Cipher.DECRYPT_MODE, keySpec);
            decodedBytes = aesCipher.doFinal(origin);
        } catch (Exception e) {
            Log.e(TAG, e.getMessage(), e);
        }

        return decodedBytes;
    }

}