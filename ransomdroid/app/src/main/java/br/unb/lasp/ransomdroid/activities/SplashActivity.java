package br.unb.lasp.ransomdroid.activities;

import android.app.Activity;
import android.content.Intent;
import android.media.MediaScannerConnection;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

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
                    Toast.makeText(getApplicationContext(), "Erro!", Toast.LENGTH_LONG).show();
                    finish();
                }
            } catch (Exception e) {
                Log.e(TAG, e.getMessage());
                e.printStackTrace();
            }
        }

    }

    /* Checks if external storage is available for read and write */
    public boolean isExternalStorageWritable() {
        String state = Environment.getExternalStorageState();
        if (Environment.MEDIA_MOUNTED.equals(state)) {
            return true;
        }
        return false;
    }

    /* Checks if external storage is available to at least read */
    public boolean isExternalStorageReadable() {
        String state = Environment.getExternalStorageState();
        if (Environment.MEDIA_MOUNTED.equals(state) ||
                Environment.MEDIA_MOUNTED_READ_ONLY.equals(state)) {
            return true;
        }
        return false;
    }

    public File[] getRoot() {
        File dir = getExternalFilesDir(null);
        return dir.listFiles();
    }

    public File[] getPictures() {
        File dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
        return dir.listFiles();
    }

    public File getDownloadsStorageDir(String albumName) {
        File file = new File(Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_DOWNLOADS), albumName);
        if (!file.mkdirs()) {
            Log.e(TAG, "Directory not created");
        }
        return file;
    }

    public File getDocumentsStorageDir(String albumName) {
        File file = new File(Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_DOCUMENTS), albumName);
        if (!file.mkdirs()) {
            Log.e(TAG, "Directory not created");
        }
        return file;
    }

    public File getDCIMStorageDir(String albumName) {
        File file = new File(Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_DCIM), albumName);
        if (!file.mkdirs()) {
            Log.e(TAG, "Directory not created");
        }
        return file;
    }

    public File getMoviesStorageDir(String albumName) {
        File file = new File(Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_MOVIES), albumName);
        if (!file.mkdirs()) {
            Log.e(TAG, "Directory not created");
        }
        return file;
    }

    public File getMusicStorageDir(String albumName) {
        File file = new File(Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_MUSIC), albumName);
        if (!file.mkdirs()) {
            Log.e(TAG, "Directory not created");
        }
        return file;
    }

    void createExternalStoragePrivateFile() {
        // Create a path where we will place our private file on external
        // storage.
        File file = new File(getExternalFilesDir(null), "DemoFile.jpg");

        try {
            // Very simple code to copy a picture from the application's
            // resource into the external file.  Note that this code does
            // no error checking, and assumes the picture is small (does not
            // try to copy it in chunks).  Note that if external storage is
            // not currently mounted this will silently fail.
            InputStream is = getResources().openRawResource(R.drawable.balloons);
            OutputStream os = new FileOutputStream(file);
            byte[] data = new byte[is.available()];
            is.read(data);
            os.write(data);
            is.close();
            os.close();
        } catch (IOException e) {
            // Unable to create file, likely because external storage is
            // not currently mounted.
            Log.w("ExternalStorage", "Error writing " + file, e);
        }
    }

    void deleteExternalStoragePrivateFile() {
        // Get path for the file on external storage.  If external
        // storage is not currently mounted this will fail.
        File file = new File(getExternalFilesDir(null), "DemoFile.jpg");
        if (file != null) {
            file.delete();
        }
    }

    boolean hasExternalStoragePrivateFile() {
        // Get path for the file on external storage.  If external
        // storage is not currently mounted this will fail.
        File file = new File(getExternalFilesDir(null), "DemoFile.jpg");
        if (file != null) {
            return file.exists();
        }
        return false;
    }

    void createExternalStoragePrivatePicture() {
        // Create a path where we will place our picture in our own private
        // pictures directory.  Note that we don't really need to place a
        // picture in DIRECTORY_PICTURES, since the media scanner will see
        // all media in these directories; this may be useful with other
        // media types such as DIRECTORY_MUSIC however to help it classify
        // your media for display to the user.
        File path = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File file = new File(path, "DemoPicture.jpg");

        try {
            // Very simple code to copy a picture from the application's
            // resource into the external file.  Note that this code does
            // no error checking, and assumes the picture is small (does not
            // try to copy it in chunks).  Note that if external storage is
            // not currently mounted this will silently fail.
            InputStream is = getResources().openRawResource(R.drawable.balloons);
            OutputStream os = new FileOutputStream(file);
            byte[] data = new byte[is.available()];
            is.read(data);
            os.write(data);
            is.close();
            os.close();

            // Tell the media scanner about the new file so that it is
            // immediately available to the user.
            MediaScannerConnection.scanFile(this,
                    new String[]{file.toString()}, null,
                    new MediaScannerConnection.OnScanCompletedListener() {
                        public void onScanCompleted(String path, Uri uri) {
                            Log.i("ExternalStorage", "Scanned " + path + ":");
                            Log.i("ExternalStorage", "-> uri=" + uri);
                        }
                    });
        } catch (IOException e) {
            // Unable to create file, likely because external storage is
            // not currently mounted.
            Log.w("ExternalStorage", "Error writing " + file, e);
        }
    }

    void deleteExternalStoragePrivatePicture() {
        // Create a path where we will place our picture in the user's
        // public pictures directory and delete the file.  If external
        // storage is not currently mounted this will fail.
        File path = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        if (path != null) {
            File file = new File(path, "DemoPicture.jpg");
            file.delete();
        }
    }

    boolean hasExternalStoragePrivatePicture() {
        // Create a path where we will place our picture in the user's
        // public pictures directory and check if the file exists.  If
        // external storage is not currently mounted this will think the
        // picture doesn't exist.
        File path = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        if (path != null) {
            File file = new File(path, "DemoPicture.jpg");
            return file.exists();
        }
        return false;
    }

}