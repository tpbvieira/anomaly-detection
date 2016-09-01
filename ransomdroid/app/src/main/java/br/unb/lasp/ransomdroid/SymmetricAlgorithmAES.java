package br.unb.lasp.ransomdroid;

import android.app.Activity;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.widget.TextView;

import java.security.SecureRandom;

import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.spec.SecretKeySpec;

import br.unb.ransomdroid.R;

public class SymmetricAlgorithmAES  extends Activity {

    static final String TAG = "SymmetricAlgorithmAES";
    TextView originalTextView;
    TextView encodedTextView;
    TextView decodedTextView;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_cryptest);

        // Original text
        String originalText = "This is just a simple test";
        originalTextView = (TextView)findViewById(R.id.original);
        originalTextView.setText("\n[ORIGINAL]:\n" + originalText + "\n");

        // Set up secret key spec for 128-bit AES encryption and decryption
        SecretKeySpec keySpec = null;
        try {
            SecureRandom random = SecureRandom.getInstance("SHA1PRNG");
            random.setSeed("any data used as random seed".getBytes());
            KeyGenerator keygen = KeyGenerator.getInstance("AES");
            keygen.init(128, random);
            keySpec = new SecretKeySpec((keygen.generateKey()).getEncoded(), "AES");
        } catch (Exception e) {
            Log.e(TAG, "AES secret key spec error");
        }

        // Encode the original data with AES
        byte[] encodedBytes = null;
        try {
            Cipher aesCipher = Cipher.getInstance("AES");
            aesCipher.init(Cipher.ENCRYPT_MODE, keySpec);
            encodedBytes = aesCipher.doFinal(originalText.getBytes());
        } catch (Exception e) {
            Log.e(TAG, "AES encryption error");
        }
        encodedTextView = (TextView)findViewById(R.id.encoded);
        encodedTextView.setText("[ENCODED]:\n" + Base64.encodeToString(encodedBytes, Base64.DEFAULT) + "\n");

        // Decode the encoded data with AES
        byte[] decodedBytes = null;
        try {
            Cipher aesCipher = Cipher.getInstance("AES");
            aesCipher.init(Cipher.DECRYPT_MODE, keySpec);
            decodedBytes = aesCipher.doFinal(encodedBytes);
        } catch (Exception e) {
            Log.e(TAG, "AES decryption error");
        }
        decodedTextView = (TextView)findViewById(R.id.decoded);
        decodedTextView.setText("[DECODED]:\n" + new String(decodedBytes) + "\n");

    }

}