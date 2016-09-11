package br.unb.lasp.util;

import android.content.Context;
import android.content.res.Resources;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.text.SimpleDateFormat;
import java.util.HashMap;

import br.unb.lasp.ransomdroid.R;

public class Json {

    private static final String TAG = new Object() {
    }.getClass().getName();

    public static final SimpleDateFormat sdfDMY = new SimpleDateFormat("dd/MM/yyyy");

    public static JSONObject createRequest(Context context, String requestString){

        try{
            JSONObject tokenJson = new JSONObject();

            JSONObject request = new JSONObject();
            request.put(requestString, tokenJson);
            return request;
        } catch (Exception e) {
            Log.e(TAG, e.getMessage(), e);
            return null;
        }
    }

    public static boolean hasError(JSONObject json){
        return json.has("erro");
    }

    public static String getError(String json, String subObject){
        String error = null;
        try{
            JSONObject object = new JSONObject(json);
            JSONObject target = (JSONObject) object.get(subObject);
            return getError(target);
        } catch (Exception e) {
            Log.e(TAG, e.getMessage(), e);
        }
        return error;
    }

    public static String getError(JSONObject json){
        String error = null;
        try{
            if(json.has("erro")){
                error = json.getString("erro");
            }
        } catch (Exception e) {
            Log.e(TAG, e.getMessage(), e);
        }
        return error;
    }

    public static String createErrorMessage(String errorMessage){
        JSONObject jsonErrorMessage = new JSONObject();
        try{
            jsonErrorMessage.put("erro", errorMessage);
        } catch (JSONException e) {
            Log.e(TAG, e.getMessage(), e);
        }
        return jsonErrorMessage.toString();
    }

    public static HashMap<String, String> fromJsonArraytoDomainHashMap(JSONArray jsonArray){
        HashMap<String, String> pairs = new HashMap<String, String>();

        try {
            for (int i = 0; i < jsonArray.length(); i++) {
                JSONObject jsonObject = jsonArray.optJSONObject(i);
                pairs.put(jsonObject.getString("id"), jsonObject.getString("nome"));
            }
        } catch (JSONException e) {
            Log.e(TAG, e.getMessage(), e);
        }

        return pairs;
    }

}