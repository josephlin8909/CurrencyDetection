package com.example.currencydetectionapp;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.Drawable;
import android.media.Image;
import android.os.Build;
import android.os.Bundle;
import android.os.FileUtils;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.FormBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class ProcessSwitch  extends AppCompatActivity {
    public static final String TAG = "ProcessSwitch";
    public static final String LINE_END = "\n";

    @RequiresApi(api = Build.VERSION_CODES.O)
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.processed_view);
        ImageView myImageView = (ImageView) findViewById(R.id.image_taken);
        TextView myTextView = (TextView) findViewById(R.id.textView3);

        Button back3;

        /*
        Daniel Choi (10/2/22) - ProcessSwitch class receives pathname from CameraSwitch and if it exists, displays it on the
        processed_view page.
         */
        Intent intent = getIntent();
        Bundle b = intent.getExtras();

        String pathname = "";
        if (b!=null) {
            // Get the pathname of the image captured from the camera
            pathname = (String) b.get("path");
        }
        if (pathname != "") {
            // Display the captured image to the user
            //Drawable d = Drawable.createFromPath(pathname);
            Bitmap original = BitmapFactory.decodeFile(pathname);
            myImageView.setImageBitmap(original);
            myImageView.setRotation(90);
            Log.d("random", pathname);
        }

        /*
        Daniel Choi - Image file is turned into a base 64 string that could be sent to the server
         */
        try {
            File image = new File(pathname);

            ByteArrayOutputStream byteOutput = new ByteArrayOutputStream();
            DataOutputStream outputStream = new DataOutputStream(byteOutput);

            FileInputStream fileInputStream = new FileInputStream(image);
            outputStream = retrieveFileBytes(outputStream, fileInputStream, image);

            fileInputStream.close();
            outputStream.flush();

            // Turn outputStream into a base64 encoded byte array
            byte[] ary = byteOutput.toByteArray();
            String encoded = Base64.getEncoder().encodeToString(ary);
            //Log.d(TAG, "Base 64 encoded byte array is: " + encoded);
            //outputStream.close();

            // Daniel Choi (10/18/22) - Posting the string to the server
            String urlString = "http://128.46.69.108:5000/post"; //Put server url here

            OkHttpClient client = new OkHttpClient.Builder().connectTimeout(15, TimeUnit.MINUTES).writeTimeout(15, TimeUnit.MINUTES).readTimeout(15, TimeUnit.MINUTES).build();

            RequestBody formbody = new FormBody.Builder().add("base64Image", encoded).build();

            Request request = new Request.Builder().url(urlString).header("Connection","close").post(formbody).build();

            Log.d("random", "request built");

            myTextView.setText("Processing...");

            // Tries to connect with server
            Call call = client.newCall(request);
            call.enqueue(new Callback() {
                @Override
                public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                    // Base64 string of converted image is received
                    String receivedStr = "";
                    if (response.isSuccessful()) {
                        try {
                            Log.d("random","before");
                            receivedStr = response.body().string();
                            Log.d("random", "after");
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }

                    // Base64 string decoded and converted to bitmap
                    //byte[] imageByteArray = Base64.getDecoder().decode(receivedBase64Str);
                    //Bitmap bmp = BitmapFactory.decodeByteArray(imageByteArray, 0, imageByteArray.length);
                    Log.d("random", "server message received");
                    String[] receivedStrArray = receivedStr.split("//",0);
                    String currency_type = receivedStrArray[0];
                    String currency_denom = receivedStrArray[1];
                    String currency_conversion = receivedStrArray[2];
                    String finalStr = "Currency Type: " + currency_type + "\n" + "Currency Denomination: " + currency_denom + "\n" + "Currency to USD conversion: $" + currency_conversion + "\n";
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            // Display the bitmap/converted image to the user and change text accordingly
                            ImageView myImageView2 = (ImageView) findViewById(R.id.image_taken);
                            //myImageView2.setImageBitmap(bmp);
                            //myImageView2.setRotation(90);
                            Log.d("random", finalStr);
                            myTextView.setText(finalStr);

                            Log.d("random","image replaced");
                        }
                    });
                }

                @Override
                public void onFailure(@NonNull Call call, @NonNull IOException e) {
                    //Toast.makeText(ProcessSwitch.this, "network not found", Toast.LENGTH_LONG).show();
                    myTextView.setText("Network not found");
                    call.cancel();
                    Log.d("random", "network not found");
                    e.printStackTrace();
                }
            });

            outputStream.close();

            // Testing that the encoding worked by decoding the array (delete later)
            /*byte[] imageByteArray = Base64.getDecoder().decode(encoded);
            Bitmap bmp = BitmapFactory.decodeByteArray(imageByteArray, 0, imageByteArray.length);
            ImageView image2 = new ImageView(this);
            myImageView.setImageBitmap(bmp);*/

        } catch (IOException e) {
            e.printStackTrace();
            Log.d(TAG, "Input file not found!");
        }

        /*
        Katherine Sandys (9/30/22) - Set the Buttons up to have functionality to move to the different pages
        When a button is pressed, it will navigate to a different page
        */
        back3 = (Button)findViewById(R.id.backButton3);
        back3.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(ProcessSwitch.this, MainActivity.class);
                startActivity(intent);
            }
        });
    }

    /*
    From VIP Help
    Daniel Choi - Gets all the bytes in the image file and puts it into the DataOutputStream
     */
    public final static DataOutputStream retrieveFileBytes(DataOutputStream outputStream, FileInputStream fileInputStream, File image) throws IOException {
        int bytesRead, bytesAvailable, bufferSize;
        byte[] buffer;
        int maxBufferSize = 1024 * 1024;

        // Creates a buffer that is the maximum number of bytes that could reasonably be read
        bytesAvailable = fileInputStream.available();
        bufferSize = Math.min(bytesAvailable, maxBufferSize);
        buffer = new byte[bufferSize];

        // Image file is read and bytes are put into buffer
        bytesRead = fileInputStream.read(buffer, 0, bufferSize);

        // buffer sent to output stream, then buffer is reset in case entire file wasn't read
        while (bytesRead > 0) {
            outputStream.write(buffer, 0, bufferSize);
            bytesAvailable = fileInputStream.available();
            bufferSize = Math.min(bytesAvailable, maxBufferSize);
            bytesRead = fileInputStream.read(buffer, 0, bufferSize);
        }

        return outputStream;

    }

}
