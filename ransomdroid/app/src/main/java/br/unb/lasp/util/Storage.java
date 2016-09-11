package br.unb.lasp.util;

import android.content.Context;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.List;

public class Storage {

    private static final String TAG = new Object() {
    }.getClass().getName();


    public static boolean isExternalStorageReadable() {
        String state = Environment.getExternalStorageState();
        if (Environment.MEDIA_MOUNTED.equals(state) ||
                Environment.MEDIA_MOUNTED_READ_ONLY.equals(state)) {
            return true;
        }
        return false;
    }

    public static boolean isExternalStorageWritable() {
        String state = Environment.getExternalStorageState();
        if (Environment.MEDIA_MOUNTED.equals(state)) {
            return true;
        }
        return false;
    }

    public static File[] getRootFiles(Context context) {
        File dir = context.getExternalFilesDir(null);
        return dir.listFiles();
    }

    public static File[] getPictures() {
        File dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
        return dir.listFiles();
    }

    public static File[] getDownloads() {
        File dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
        return dir.listFiles();
    }

    public static File[] getDocuments() {
        File dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS);
        return dir.listFiles();
    }

    public static File[] getDCIMS() {
        File dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM);
        return dir.listFiles();
    }

    public static File[] getMovies() {
        File dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES);
        return dir.listFiles();
    }

    public static File[] getMusics(String albumName) {
        File dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MUSIC);
        return dir.listFiles();
    }

    public static File getRootDir(Context context) {
        File dir = context.getExternalFilesDir(null);
        return dir;
    }

    public static File getPicturesDir() {
        File dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
        return dir;
    }

    public static File getDownloadsDir() {
        File dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
        return dir;
    }

    public static File getDocumentsDir() {
        File dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS);
        return dir;
    }

    public static File getDCIMSDir() {
        File dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM);
        return dir;
    }

    public static File getMoviesDir() {
        File dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES);
        return dir;
    }

    public static File getMusicsDir(String albumName) {
        File dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MUSIC);
        return dir;
    }

    public static void saveReplaceFile(String path, String fileName, byte[] data) {

        File dir = new File(Environment.getExternalStorageDirectory() + path);
        if (!dir.exists()) {
            dir.mkdirs();
        }
        File file = new File(dir, fileName);
        if (file.exists()) {
            file.delete();
        }

        FileOutputStream os = null;
        try {
            os = new FileOutputStream(file);
            os.write(data);
            os.flush();
            os.close();
        } catch (Exception e) {
            e.printStackTrace();
        }finally {
            if (os != null) {
                try {
                    os.close();
                } catch (IOException e) {
                    Log.e(TAG, e.getMessage(), e);
                }
            }
        }
    }

    public void saveReplaceFile(File file, byte[] data) {

        if (!file.exists()) {
            File dir = new File(file.getParent());
            if (!dir.exists()) {
                dir.mkdirs();
            }
        }else{
            file.delete();
        }

        FileOutputStream os = null;
        try {
            os = new FileOutputStream(file);
            os.write(data);
            os.flush();
            os.close();
        } catch (Exception e) {
            e.printStackTrace();
        }finally {
            if (os != null) {
                try {
                    os.close();
                } catch (IOException e) {
                    Log.e(TAG, e.getMessage(), e);
                }
            }
        }
    }

    void createExternalStoragePrivatePicture(Context context, File file, byte[] data) {

        OutputStream os = null;
        boolean created = false;
        try {
            os = new FileOutputStream(file);
            os.write(data);
            created = true;
        } catch (IOException e) {
            Log.e(TAG, e.getMessage(), e);
        }finally {
            if (os != null) {
                try {
                    os.close();
                } catch (IOException e) {
                    Log.e(TAG, e.getMessage(), e);
                }
            }
        }

        if(created){
            MediaScannerConnection.scanFile(context,
                    new String[]{file.toString()}, null,
                    new MediaScannerConnection.OnScanCompletedListener() {
                        public void onScanCompleted(String path, Uri uri) {
                            Log.i(TAG, "Scanned " + path + ":");
                            Log.i(TAG, "-> uri=" + uri);
                        }
                    });
        }

    }

    public static List<File> listFilesRecursively(File path, List<File> fileList){
        File[] files = path.listFiles();
        for (File file:files) {
            if(file.isDirectory()){
                listFilesRecursively(file,fileList);
            }else{
                fileList.add(file);
            }
        }
        return fileList;
    }

}