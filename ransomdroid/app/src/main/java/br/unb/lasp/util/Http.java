package br.unb.lasp.util;

import android.util.Log;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

public class Http {

    private static final String TAG = new Object() {
    }.getClass().getName();

    static private String convertStreamToString(InputStream is) {
        BufferedReader reader = new BufferedReader(new InputStreamReader(is));
        StringBuilder sb = new StringBuilder();

        try {
            String line = null;
            while ((line = reader.readLine()) != null) {
                sb.append(line + "\n");
            }
        } catch (IOException e) {
            Log.e(TAG, e.getMessage(), e);
        } finally {
            try {
                if (reader != null) {
                    reader.close();
                }
                if (is != null) {
                    is.close();
                }
            } catch (IOException e) {
                Log.e(TAG, e.getMessage(), e);
            }
        }

        return sb.toString();
    }

    static public String loadStringFromURL(String url) {
        HttpURLConnection connection = null;
        String json = null;
        InputStream is = null;
        try {
            connection = (HttpURLConnection) new URL(url).openConnection();
            connection.setConnectTimeout(30000);
            connection.setReadTimeout(30000);

            is = new BufferedInputStream(connection.getInputStream());
            json = convertStreamToString(is);

        } catch (IOException e) {
            Log.e(TAG, e.getMessage(), e);
        } finally {
            try {
                if (is != null) {
                    is.close();
                }
            } catch (IOException e) {
                Log.e(TAG, e.getMessage(), e);
            }

            if (connection != null) {
                connection.disconnect();
            }
        }
        return json;
    }

    static public InputStream loadImageFromURL(String url) {
        InputStream is = null;
        HttpURLConnection connection = null;
        try {
            connection = (HttpURLConnection) new URL(url).openConnection();
            connection.setConnectTimeout(15000);

            is = new BufferedInputStream(connection.getInputStream());
        } catch (IOException e) {
            Log.e(TAG, e.getMessage(), e);
        }
        return is;
    }

}