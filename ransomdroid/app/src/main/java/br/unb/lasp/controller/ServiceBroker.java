package br.unb.lasp.controller;

import android.content.Context;
import android.util.Log;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.ConnectException;
import java.net.HttpURLConnection;
import java.net.URL;

import br.unb.lasp.Global;
import br.unb.lasp.ransomdroid.R;
import br.unb.lasp.util.Connectivity;
import br.unb.lasp.util.Json;

public class ServiceBroker {

    private static final String TAG = new Object() {
    }.getClass().getName();

	private static ServiceBroker instance = null;

	private Connectivity connectivityUtils;

	public static ServiceBroker getInstance(Context context) {
		if (instance == null) {
			instance = new ServiceBroker(context);
		}
		return instance;
	}

	private ServiceBroker(Context ctx) {
		connectivityUtils = Connectivity.getInstance(ctx);
	}

	public String postMessage(String json) {
		return requestJson("/postMessage", json);
	}

    private String requestJson(String service, String json) {
        String dados = null;
        try {
            if(connectivityUtils.hasConnectivity()) {
                service = Global.serverURL + service;

                URL url = new URL(service);
                HttpURLConnection servConn = (HttpURLConnection) url.openConnection();
                servConn.setRequestMethod("POST");
                servConn.setReadTimeout(15000);
                servConn.setConnectTimeout(15000);
                servConn.setRequestProperty("Content-Type", "application/json");
                servConn.setUseCaches(false);
                servConn.setAllowUserInteraction(false);
                servConn.connect();

                OutputStream os = new BufferedOutputStream(servConn.getOutputStream());
                os.write(json.getBytes("UTF-8"));
                os.flush();

                int status = servConn.getResponseCode();
                if (status == HttpURLConnection.HTTP_OK || status == HttpURLConnection.HTTP_CREATED) {
                    dados = response(servConn.getInputStream());
                }
            }
        } catch (ConnectException e) {
            Log.e(TAG, e.getMessage(), e);
            dados = Json.createErrorMessage(e.getMessage());
        } catch (Exception e) {
            Log.e(TAG, e.getMessage(), e);
            dados = Json.createErrorMessage(e.getMessage());
        }

        return dados;
    }

    private String response(InputStream in) throws IOException {
        String response = null;
        try {
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(in, "UTF-8"));
            StringBuilder stringBuilder =  new StringBuilder();
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                stringBuilder.append(line);
            }
            bufferedReader.close();
            response = stringBuilder.toString();
        } catch (Exception e) {
            Log.e(TAG,e.getMessage());
            response = Json.createErrorMessage(e.getMessage());
        }
        return response;
    }

}