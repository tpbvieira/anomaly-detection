package br.unb.lasp.ransomdroid.activities;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.AutoCompleteTextView;
import android.widget.Button;
import android.widget.TextView;

import com.google.android.gms.appindexing.AppIndex;
import com.google.android.gms.common.api.GoogleApiClient;

import br.unb.ransomdroid.R;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = MainActivity.class.getSimpleName();

    // UI references.
    private View mProgressView, mMOSTestFormView;
    private TextView mMessageResultTextView;
    private AutoCompleteTextView mFilePathView, mWindowSizeView;

    private String msg = "";
    private GoogleApiClient client;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_mostest);

        // Set up the login form.
        mFilePathView = (AutoCompleteTextView) findViewById(R.id.file_path);
        mWindowSizeView = (AutoCompleteTextView) findViewById(R.id.window_size);

        Button mEvaluateButton = (Button) findViewById(R.id.evaluate_button);
        mEvaluateButton.setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View view) {
                evaluate();
            }
        });

        mMOSTestFormView = findViewById(R.id.login_form);
        mProgressView = findViewById(R.id.login_progress);
        mMessageResultTextView = (TextView) findViewById(R.id.msgResult);
        mMessageResultTextView.setText("Result:");
        client = new GoogleApiClient.Builder(this).addApi(AppIndex.API).build();
    }

    private void evaluate() {
        // Reset errors.
        mFilePathView.setError(null);
        mWindowSizeView.setError(null);
    }

    @Override
    public void onStart() {
        super.onStart();
    }

    @Override
    public void onStop() {
        super.onStop();
    }

}