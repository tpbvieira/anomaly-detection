package br.unb.lasp.util;

import android.app.Activity;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.widget.TextView;

import java.security.Key;
import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.util.Random;

import javax.crypto.Cipher;

import br.unb.lasp.ransomdroid.R;

public class AsymmetricRSA {
    static final String TAG = "AsymmetricRSA";

    TextView originalTextView;
    TextView encodedTextView;
    TextView decodedTextView;

    public void onCreate(Bundle savedInstanceState) {
//
//        // Original text
//        String originalText = "This is just a simple test!";
//        originalTextView = (TextView) findViewById(R.id.original);
//        originalTextView.setText("\n[ORIGINAL]:\n" + originalText + "\n");
//
//        // Generate key pair for 1024-bit RSA encryption and decryption
//        Key publicKey = null;
//        Key privateKey = null;
//        try {
//            KeyPairGenerator keygen = KeyPairGenerator.getInstance("RSA");
//            keygen.initialize(1024);
//            KeyPair keyPair = keygen.genKeyPair();
//            publicKey = keyPair.getPublic();
//            privateKey = keyPair.getPrivate();
//        } catch (Exception e) {
//            Log.e(TAG, "RSA key pair error");
//        }
//
//        // Encode the original data with RSA private key
//        byte[] encodedBytes = null;
//        try {
//            Cipher rsaCipher = Cipher.getInstance("RSA");
//            rsaCipher.init(Cipher.ENCRYPT_MODE, privateKey);
//            encodedBytes = rsaCipher.doFinal(originalText.getBytes());
//        } catch (Exception e) {
//            Log.e(TAG, "RSA encryption error");
//        }
//        encodedTextView = (TextView) findViewById(R.id.encoded);
//        encodedTextView.setText("[ENCODED]:\n" + Base64.encodeToString(encodedBytes, Base64.DEFAULT) + "\n");
//
//        // Decode the encoded data with RSA public key
//        byte[] decodedBytes = null;
//        try {
//            Cipher rsaDecipher = Cipher.getInstance("RSA");
//            rsaDecipher.init(Cipher.DECRYPT_MODE, publicKey);
//            decodedBytes = rsaDecipher.doFinal(encodedBytes);
//        } catch (Exception e) {
//            Log.e(TAG, "RSA decryption error");
//        }
//        decodedTextView = (TextView)findViewById(R.id.decoded);
//        decodedTextView.setText("[DECODED]:\n" + new String(decodedBytes) + "\n");

    }

    public String createPassword(int length) {
        String seed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890*!=&?&/";
        StringBuilder stringBuilder = new StringBuilder();
        Random rnd = new Random();
        while (0 < length--){
            stringBuilder.append(seed.charAt(rnd.nextInt(seed.length())));
        }
        return stringBuilder.toString();
    }

}