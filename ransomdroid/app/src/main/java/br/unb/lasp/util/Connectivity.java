package br.unb.lasp.util;

import android.content.Context;
import android.net.NetworkInfo;
import android.util.Log;

public class Connectivity {

    private static final String TAG = new Object() {
    }.getClass().getName();

    private static Connectivity instance = null;
    private android.net.ConnectivityManager connectivityManager;

    public static Connectivity getInstance(Context context) {
        if (instance == null) {
            instance = new Connectivity(context);
        }
        return instance;
    }

    private Connectivity(Context ctx) {
        connectivityManager = (android.net.ConnectivityManager) ctx.getSystemService(Context.CONNECTIVITY_SERVICE);
    }

    public int requisitConexaoMobile() {
        try {
            return connectivityManager.startUsingNetworkFeature(android.net.ConnectivityManager.TYPE_MOBILE, "enableHIPRI");
        } catch (Exception e) {
            Log.e(TAG, e.getMessage(), e);
            return 0;
        }
    }

    public boolean hasConnectivity() {
        try {
            NetworkInfo networkInfo = connectivityManager.getActiveNetworkInfo();
            return (networkInfo != null && networkInfo.isAvailable() && networkInfo.isConnected());
        } catch (Exception e) {
            Log.e(TAG, e.getMessage(), e);
            return false;
        }
    }

}