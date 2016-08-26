package co.salutary.mos;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.annotation.TargetApi;
import android.app.LoaderManager.LoaderCallbacks;
import android.content.Context;
import android.content.CursorLoader;
import android.content.Loader;
import android.database.Cursor;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.ContactsContract;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.ArrayAdapter;
import android.widget.AutoCompleteTextView;
import android.widget.Button;
import android.widget.TextView;

import com.google.android.gms.appindexing.Action;
import com.google.android.gms.appindexing.AppIndex;
import com.google.android.gms.common.api.GoogleApiClient;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import Jama.Matrix;
import br.unb.lasp.anomaly.mos.MOS;
import br.unb.lasp.anomaly.mos.Parser;
import br.unb.lasp.matrix.DateUtil;

/**
 * A login screen that offers login via email/password.
 */
public class MOSTestActivity extends AppCompatActivity implements LoaderCallbacks<Cursor> {

    private static final String TAG = MOSTestActivity.class.getSimpleName();

    /**
     * Keep track of the login task to ensure we can cancel it if requested.
     */
    private UserLoginTask mAuthTask = null;

    // UI references.
    private View mProgressView, mMOSTestFormView;
    private TextView mMessageResultTextView;
    private AutoCompleteTextView mFilePathView, mWindowSizeView;

    long parsingTime, modelingTime, eigTime, mosTime, totalTime;
    String eigMsg, mosMsg;

    private String msg = "";
    /**
     * ATTENTION: This was auto-generated to implement the App Indexing API.
     * See https://g.co/AppIndexing/AndroidStudio for more information.
     */
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
        // ATTENTION: This was auto-generated to implement the App Indexing API.
        // See https://g.co/AppIndexing/AndroidStudio for more information.
        client = new GoogleApiClient.Builder(this).addApi(AppIndex.API).build();
    }

    /**
     * Attempts to sign in or register the account specified by the login form.
     * If there are form errors (invalid email, missing fields, etc.), the
     * errors are presented and no actual login attempt is made.
     */
    private void evaluate() {

        if (mAuthTask != null) {
            return;
        }

        // Reset errors.
        mFilePathView.setError(null);
        mWindowSizeView.setError(null);

        // Store values at the time of the login attempt.
        String filePath = mFilePathView.getText().toString();
        Short windowSize = Short.parseShort(mWindowSizeView.getText().toString());

        showProgress(true);
        mAuthTask = new UserLoginTask(this.getApplicationContext(), filePath, windowSize);
        mAuthTask.execute((Void) null);

    }

    /**
     * Shows the progress UI and hides the login form.
     */
    @TargetApi(Build.VERSION_CODES.HONEYCOMB_MR2)
    private void showProgress(final boolean show) {
        // On Honeycomb MR2 we have the ViewPropertyAnimator APIs, which allow
        // for very easy animations. If available, use these APIs to fade-in
        // the progress spinner.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.HONEYCOMB_MR2) {
            int shortAnimTime = getResources().getInteger(android.R.integer.config_shortAnimTime);

            mMOSTestFormView.setVisibility(show ? View.GONE : View.VISIBLE);
            mMOSTestFormView.animate().setDuration(shortAnimTime).alpha(
                    show ? 0 : 1).setListener(new AnimatorListenerAdapter() {
                @Override
                public void onAnimationEnd(Animator animation) {
                    mMOSTestFormView.setVisibility(show ? View.GONE : View.VISIBLE);
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
            // The ViewPropertyAnimator APIs are not available, so simply show
            // and hide the relevant UI components.
            mProgressView.setVisibility(show ? View.VISIBLE : View.GONE);
            mMOSTestFormView.setVisibility(show ? View.GONE : View.VISIBLE);
        }
    }

    @Override
    public Loader<Cursor> onCreateLoader(int i, Bundle bundle) {
        return new CursorLoader(this,
                // Retrieve data rows for the device user's 'profile' contact.
                Uri.withAppendedPath(ContactsContract.Profile.CONTENT_URI,
                        ContactsContract.Contacts.Data.CONTENT_DIRECTORY), ProfileQuery.PROJECTION,

                // Select only email addresses.
                ContactsContract.Contacts.Data.MIMETYPE +
                        " = ?", new String[]{ContactsContract.CommonDataKinds.Email
                .CONTENT_ITEM_TYPE},

                // Show primary email addresses first. Note that there won't be
                // a primary email address if the user hasn't specified one.
                ContactsContract.Contacts.Data.IS_PRIMARY + " DESC");
    }

    @Override
    public void onLoadFinished(Loader<Cursor> cursorLoader, Cursor cursor) {
        List<String> emails = new ArrayList<>();
        cursor.moveToFirst();
        while (!cursor.isAfterLast()) {
            emails.add(cursor.getString(ProfileQuery.ADDRESS));
            cursor.moveToNext();
        }

        addEmailsToAutoComplete(emails);
    }

    @Override
    public void onLoaderReset(Loader<Cursor> cursorLoader) {

    }

    private void addEmailsToAutoComplete(List<String> emailAddressCollection) {
        //Create adapter to tell the AutoCompleteTextView what to show in its dropdown list.
        ArrayAdapter<String> adapter =
                new ArrayAdapter<>(MOSTestActivity.this,
                        android.R.layout.simple_dropdown_item_1line, emailAddressCollection);

        mFilePathView.setAdapter(adapter);
    }

    @Override
    public void onStart() {
        super.onStart();

        // ATTENTION: This was auto-generated to implement the App Indexing API.
        // See https://g.co/AppIndexing/AndroidStudio for more information.
        client.connect();
        Action viewAction = Action.newAction(
                Action.TYPE_VIEW, // TODO: choose an action type.
                "MOSTest Page", // TODO: Define a title for the content shown.
                // TODO: If you have web page content that matches this app activity's content,
                // make sure this auto-generated web page URL is correct.
                // Otherwise, set the URL to null.
                Uri.parse("http://host/path"),
                // TODO: Make sure this auto-generated app deep link URI is correct.
                Uri.parse("android-app://co.salutary.mos/http/host/path")
        );
        AppIndex.AppIndexApi.start(client, viewAction);
    }

    @Override
    public void onStop() {
        super.onStop();

        // ATTENTION: This was auto-generated to implement the App Indexing API.
        // See https://g.co/AppIndexing/AndroidStudio for more information.
        Action viewAction = Action.newAction(
                Action.TYPE_VIEW, // TODO: choose an action type.
                "MOSTest Page", // TODO: Define a title for the content shown.
                // TODO: If you have web page content that matches this app activity's content,
                // make sure this auto-generated web page URL is correct.
                // Otherwise, set the URL to null.
                Uri.parse("http://host/path"),
                // TODO: Make sure this auto-generated app deep link URI is correct.
                Uri.parse("android-app://co.salutary.mos/http/host/path")
        );
        AppIndex.AppIndexApi.end(client, viewAction);
        client.disconnect();
    }


    private interface ProfileQuery {
        String[] PROJECTION = {
                ContactsContract.CommonDataKinds.Email.ADDRESS,
                ContactsContract.CommonDataKinds.Email.IS_PRIMARY,
        };

        int ADDRESS = 0;
        int IS_PRIMARY = 1;
    }

    /**
     * Represents an asynchronous login/registration task used to authenticate
     * the user.
     */
    public class UserLoginTask extends AsyncTask<Void, Void, Boolean> {

        private final String mFileName;
        private final Short mWindowSize;
        private Context mContext;

        UserLoginTask(Context context, String fileName, Short windowSize) {
            mContext = context;
            mFileName = fileName;
            mWindowSize = windowSize;
            msg="";
        }

        @Override
        protected Boolean doInBackground(Void... params) {
            try{
                SimpleDateFormat sdf = new SimpleDateFormat(DateUtil.usDateTimeMS);

                // Parsing
                File file = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), mFileName);
                Log.d(TAG, "Parsing " + file.getPath());
                totalTime = parsingTime = System.currentTimeMillis();
                HashMap<String,Integer> matrixRowIndeces = new HashMap<String,Integer>();
                HashMap<Long,HashMap<String,Integer>> values = Parser.parseCSV(file.getPath(), sdf, matrixRowIndeces);
                parsingTime =  System.currentTimeMillis() - parsingTime;
                Log.d(TAG, "parsingTime=" + parsingTime );

                // Data Modeling
                Log.d(TAG, "Modeling...");
                modelingTime = System.currentTimeMillis();
                HashMap<Long,Integer> matrixColumnIndeces = new HashMap<Long,Integer>();
                Matrix[] matrices = Parser.modelIntoMatrices(mWindowSize, matrixRowIndeces, values, matrixColumnIndeces);
                modelingTime = System.currentTimeMillis() - modelingTime;
                Log.d(TAG, "modelingTime=" + modelingTime);

                values = null;
                matrixRowIndeces = null;
                matrixColumnIndeces = null;

                DescriptiveStatistics eigenStats = new DescriptiveStatistics();
                DescriptiveStatistics mosStats = new DescriptiveStatistics();

                for(int i=0; i<50;i++){
                    // EigenAnalysis
                    eigTime = System.currentTimeMillis();
                    double[][] largestEigValCov = Parser.getLargestEigValCov(matrices);
                    double[][] largestEigValCor = Parser.getLargestEigValCor(matrices);
                    eigTime = System.currentTimeMillis() - eigTime;
                    eigenStats.addValue(eigTime);

                    // MOS Analysis
                    mosTime = System.currentTimeMillis();
                    int mosCov = MOS.edcJAMA(new Matrix(largestEigValCov), mWindowSize);
                    int mosCor = MOS.edcJAMA(new Matrix(largestEigValCor), mWindowSize);
                    mosTime = System.currentTimeMillis() - mosTime;
                    mosStats.addValue(mosTime);
                }

                eigMsg = "Avg=" + eigenStats.getMean() + ", Stdv=" + eigenStats.getStandardDeviation() + ", Min=" + eigenStats.getMin() + ", Max="+ eigenStats.getMax();
                mosMsg = "Avg=" + mosStats.getMean() + ", Stdv=" + mosStats.getStandardDeviation() + ", Min=" + mosStats.getMin() + ", Max="+ mosStats.getMax();

                msg = "Success! " +
                        "\n" + file.getPath() +
                        "\nSize=" + (double)file.length()/(double)(1024*1024) + ", Window=" + mWindowSize;

                return true;

            }catch (Exception e){
                msg = e.getMessage();
                return false;
            }

        }

        @Override
        protected void onPostExecute(final Boolean success) {
            mAuthTask = null;
            showProgress(false);
            mMessageResultTextView.setText(
                    "Msg:" + msg +
                    "\n\nParsingTime: " + parsingTime +
                    "\n\nDataModellingTime: " + modelingTime +
                    "\n\nEig: " + eigMsg +
                    "\n\nMOS:" + mosMsg +
                    "\n\nTotal:" + totalTime);
        }

        @Override
        protected void onCancelled() {
            mAuthTask = null;
            showProgress(false);
        }

    }

}