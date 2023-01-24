package com.example.currencydetectionapp;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.media.Image;
import android.os.Bundle;
import android.os.Environment;
import android.util.Size;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class CameraSwitch extends AppCompatActivity {
    private final int REQUEST_CODE_PERMISSIONS = 1001;
    private final String[] REQUIRED_PERMISSIONS = new String[] {"android.permission.CAMERA", "android.permission.WRITE_EXTERNAL_STORAGE"};
    private final Executor cameraExecutor = Executors.newSingleThreadExecutor();

    PreviewView mPreviewView;
    Button pictureBTN;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.camera_view);

        mPreviewView = findViewById(R.id.camera_view);

        if (allPermissionsGranted()) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }

        /*
        Katherine Sandys (9/30/22) - Set the Buttons up to have functionality to move to the different pages
        When a button is pressed, it will navigate to a different page
        */
        Button back1;

        Button image_taken;

        back1 = (Button)findViewById(R.id.backButton1);
        back1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(CameraSwitch.this, MainActivity.class);
                startActivity(intent);
            }
        });
    }

    /*
    Daniel Choi (9/22/22) - Creates instance of ProcessCameraProvider and attach a listener.
    When the listener is told to run, it calls the bindPreview() function
     */
    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(new Runnable() {
            @Override
            public void run() {
                try {
                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                    bindPreview(cameraProvider);
                } catch (ExecutionException | InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }, ContextCompat.getMainExecutor(this));

    }

    /*
    Daniel Choi (9/22/22) - Builds a preview object and selects the back camera. Then attaches the
    preview object and camera selection to the cameraProvider, which allows us to see the camera
    preview on the phone.
    Daniel Choi (10/2/22) - Added capture button to save image as png to devices internal storage. If the image
    is saved properly, the processed_view activity is opened.
     */
    private void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {
        CameraSelector cameraSelector = new CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build();
        Preview preview = new Preview.Builder().build();
        //Size imagesize144p = new Size(144, 256);
        //Size imagesize240p = new Size(240, 426);
        //Size imagesize360p = new Size(360, 640);
        //Size imagesize480p = new Size(480, 854);
        Size imagesize720p = new Size(720, 1280); //Use
        //Size imagesize1080p = new Size(1080, 1920);
        //Size imagesize2160p = new Size(2160, 3840);

        ImageCapture imageCapture = new ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY).setTargetResolution(imagesize720p).build();
        //ImageCapture imageCapture = new ImageCapture.Builder().build();

        preview.setSurfaceProvider(mPreviewView.getSurfaceProvider());
        cameraProvider.bindToLifecycle((LifecycleOwner) this, cameraSelector, preview, imageCapture);

        pictureBTN = (Button) findViewById(R.id.take_image);
        pictureBTN.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String timeStamp = new SimpleDateFormat("yyyy.MM.dd.HH.mm.ss").format(new java.util.Date());
                String pathname = getCacheDir().getAbsolutePath() + "/" + timeStamp + ".jpg";
                File file = new File(pathname);
                ImageCapture.OutputFileOptions outputFileOptions = new ImageCapture.OutputFileOptions.Builder(file).build();
                Toast.makeText(CameraSwitch.this, file.getAbsolutePath(), Toast.LENGTH_SHORT).show();
                imageCapture.takePicture(outputFileOptions, Executors.newSingleThreadExecutor(), new ImageCapture.OnImageSavedCallback() {
                    @Override
                    public void onImageSaved(@NonNull ImageCapture.OutputFileResults outputFileResults) {
                        //Get errors when I try to display text using toast
                        //Toast.makeText(MainActivity.this, "Image Saved Successfully", Toast.LENGTH_SHORT).show();
                        Intent intent = new Intent(CameraSwitch.this, ProcessSwitch.class);
                        intent.putExtra("path", pathname);
                        startActivity(intent);
                    }

                    @Override
                    public void onError(@NonNull ImageCaptureException error) {
                        //Toast.makeText(MainActivity.this, "Failed", Toast.LENGTH_SHORT).show();
                        error.printStackTrace();
                    }
                });
            }
            });
    }

    /*
    Daniel Choi (9/22/22) - Checks that the camera and write external storage permissions were granted by the user
     */
    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(getApplicationContext(),permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    /*
    Daniel Choi (9/22/22) - When permissions are requested, checks to see if the user approves the permissions.
    If yes, then starts the camera preview. Otherwise, displays an error message and exits the app.
     */
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera();
            } else {
                Toast.makeText(this, "Permissions not granted by user", Toast.LENGTH_SHORT).show();
                this.finish();
            }
        }
    }

}