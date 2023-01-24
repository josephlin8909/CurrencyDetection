package com.example.currencydetectionapp;

import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContract;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.io.FileOutputStream;
import java.text.SimpleDateFormat;

public class MainActivity extends AppCompatActivity {
    @Override
    protected  void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.home_view);

        /*
        Katherine Sandys (9/30/22) - Set the Buttons up to have functionality to move to the different pages
        When a button is pressed, it will navigate to a different page
        */
        Button camera_page;
        Button info_page;
        Button load_image;

        /*
        Daniel Choi (12/4/22) - Gets the filepath of the image the user selected by first getting the relative uri, then using a cursor to get the absolute uri
         */
        ActivityResultLauncher<Intent> startForResult = registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), new ActivityResultCallback<ActivityResult>() {
            @Override
            public void onActivityResult(ActivityResult result) {
                if (result != null && result.getResultCode() == RESULT_OK) {
                    if (result.getData() != null) {
                        Uri uri = result.getData().getData();
                        String[] filePathColumn = {MediaStore.Images.Media.DATA};
                        Cursor cursor = getContentResolver().query(uri, filePathColumn, null, null, null);
                        cursor.moveToFirst();
                        int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                        String pathname = cursor.getString(columnIndex);cursor.close();
                        Log.d("random", pathname);
                        Intent intent = new Intent(MainActivity.this, ProcessSwitch.class);
                        intent.putExtra("path", pathname);
                        startActivity(intent);
                    }
                }
            }
        });

        camera_page = (Button)findViewById(R.id.camera);
        camera_page.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, CameraSwitch.class);
                startActivity(intent);
            }
        });
        info_page = (Button)findViewById(R.id.info);
        info_page.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, InfoSwitch.class);
                startActivity(intent);
            }
        });
        /*
        Daniel Choi (12/3/22) - Added a button to load an image from gallery
         */
        load_image = (Button)findViewById(R.id.load_image);
        load_image.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startForResult.launch(intent);

            }
        });



    }

}

