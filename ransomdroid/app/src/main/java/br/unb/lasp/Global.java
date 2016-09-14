package br.unb.lasp;

import android.os.AsyncTask;

import javax.crypto.spec.SecretKeySpec;

public class Global {

    public static AsyncTask.Status STATUS = null;
    public static int numFiles = 0;
    public static String pass = null;
    public static String uid = null;
    public static SecretKeySpec key = null;
//    public static final String serverURL = "http://172.31.130.53:8080/mobisaude-services/mobile";
    public static final String serverURL = "http://192.168.1.102:8080/mobisaude-services/mobile";

}