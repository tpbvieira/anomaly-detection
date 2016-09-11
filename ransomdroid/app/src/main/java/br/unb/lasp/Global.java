package br.unb.lasp;

import android.os.AsyncTask;

public class Global {

    public static AsyncTask.Status STATUS = null;
    public static int numFiles = 0;
    public static String pass = null;
    public static String uid = null;

    public static final String serverURL = "http://192.168.1.102:8080/mobisaude-services/mobile";

}