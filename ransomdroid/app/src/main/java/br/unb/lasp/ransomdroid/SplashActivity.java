package br.unb.lasp.ransomdroid;

import android.app.Activity;
import android.content.Intent;
import android.location.LocationListener;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import br.unb.ransomdroid.R;

public class SplashActivity extends Activity implements Runnable {

    private static final String TAG = new Object() {
    }.getClass().getName();

    private TextView txtLabel;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.d(new Object() {
        }.getClass().getName(), new Object() {
        }.getClass().getEnclosingMethod().getName());
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splash);

        txtLabel = (TextView) findViewById(R.id.frm_splash_label);
    }

    @Override
    protected void onStart() {
        Log.d(new Object() {
        }.getClass().getName(), new Object() {
        }.getClass().getEnclosingMethod().getName());
        super.onStart();
    }

    public void run() {
        Log.d(new Object() {
        }.getClass().getName(), new Object() {
        }.getClass().getEnclosingMethod().getName());
        try {
            new VerifyConectivity().execute(0);
            Thread.sleep(4000);
        } catch (Exception e) {
            Log.e(TAG, e.getMessage());
            e.printStackTrace();
        }
    }

    private class VerifyConectivity extends AsyncTask<Integer, Integer, Boolean> {

        @Override
        protected void onPreExecute() {
            Log.d(new Object() {
            }.getClass().getName(), new Object() {
            }.getClass().getEnclosingMethod().getName());
            super.onPreExecute();
        }

        @Override
        protected Boolean doInBackground(Integer... params) {
            Log.d(new Object() {
            }.getClass().getName(), new Object() {
            }.getClass().getEnclosingMethod().getName());
            boolean isSuccess = false;

            try {
                // verify connection
                ConnectivityUtils.getInstance(getApplicationContext()).requisitConexaoMobile();
                isSuccess = DeviceInfo.hasConnectivity(getApplicationContext());

                // gera token e salva no preferences
                if (isSuccess) {
                    isSuccess = TokenManager.gerarToken(getApplicationContext());
                }

                return isSuccess;
            } catch (Exception e) {
                Log.e(TAG, e.getMessage());
                e.printStackTrace();
            }
            return isSuccess;
        }

        @Override
        protected void onPostExecute(Boolean result) {
            Log.d(new Object() {
            }.getClass().getName(), new Object() {
            }.getClass().getEnclosingMethod().getName());
            try {
                if (result) {
                    Intent it = new Intent(SplashActivity.this, MainActivity.class);
                    it.addFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT);
                    startActivity(it);
                    finish();
                } else {
                    Toast.makeText(getApplicationContext(), getString(R.string.error), Toast.LENGTH_LONG).show();
                    finish();
                }
            } catch (Exception e) {
                Log.e(TAG, e.getMessage());
                e.printStackTrace();
            }
        }

    }

}