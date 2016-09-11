package br.unb.lasp.util;

public class RansomdroidException extends Exception{

    private static final String TAG = new Object() {
    }.getClass().getName();

    public RansomdroidException(String message){
        super(message);
    }

}
