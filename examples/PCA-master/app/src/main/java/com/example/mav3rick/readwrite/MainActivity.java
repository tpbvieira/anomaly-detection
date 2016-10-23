package com.example.mav3rick.readwrite;

import android.app.Activity;
import android.content.DialogInterface;
import android.support.v7.app.ActionBarActivity;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.view.Gravity;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;


public class MainActivity extends Activity implements View.OnClickListener{
    EditText fname, fnameread;
    Button build, classify, eval;
    //ImageView imageView;
    TextView filecon, textView;
    int counter = 0;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        fname = (EditText) findViewById(R.id.fname);
        final String[] filename = new String[1];

        fnameread = (EditText) findViewById(R.id.fnameread);
        build = (Button) findViewById(R.id.bbuild);
        classify = (Button) findViewById(R.id.beval);
        eval = (Button) findViewById(R.id.eval);
        filecon = (TextView) findViewById(R.id.filecon);
        filecon.setMovementMethod(new ScrollingMovementMethod());

        textView = (TextView) findViewById(R.id.textView);

        textView.setOnClickListener(this);

        build.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {

                filename[0] = fname.getText().toString();

                Tranny t1 = new Tranny();
                int flag = t1.build(filename[0]);

                if (flag == 0) {
                    Toast.makeText(getApplicationContext(),"Classifier built :)", Toast.LENGTH_SHORT).show();

                    try {
                        BufferedReader reader = new  BufferedReader(new FileReader("/sdcard/model.txt"));

                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    }
                }
                else if(flag == 1) {
                    Toast.makeText(getApplicationContext(), "Error :( File not found", Toast.LENGTH_SHORT).show();
                }
                else if(flag == 2) {
                    Toast.makeText(getApplicationContext(), "Error :( Classifier not built", Toast.LENGTH_SHORT).show();
                }

                Toast toast = Toast.makeText(getApplicationContext(), "Generating model", Toast.LENGTH_SHORT); toast.setGravity(Gravity.TOP|Gravity.CENTER, 0, 0);
                toast.show();


            }
        });

        classify.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String readfilename = fnameread.getText().toString();
                Tranny t1 = new Tranny();
                String out = t1.classify(readfilename);
                filecon.setText(out);
            }
        });

        eval.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String readfilename = fnameread.getText().toString();
                Tranny t1 = new Tranny();
                String out = t1.evaluate(readfilename);
                filecon.setText(out);
            }
        });

    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }
    @Override
    public void onClick(View v)
    {

        if(v == textView) {
            counter++;
        }
        if(counter==7)
        {
            Toast.makeText(getApplicationContext(), "Yo mama is so fat, that her weight is more than the lines of code!", Toast.LENGTH_LONG) .show();
            //imageView.setVisibility(View.VISIBLE);

        }
        else if(counter==14)
        {
            Toast.makeText(getApplicationContext(), "Says THE BATMAN!", Toast.LENGTH_LONG).show();
            counter = 0;
        }
    }
}
