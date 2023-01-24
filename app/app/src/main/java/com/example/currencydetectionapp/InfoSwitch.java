package com.example.currencydetectionapp;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

public class InfoSwitch extends AppCompatActivity {
    @Override
    protected  void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.info_view);

        /*
        Katherine Sandys (9/30/22) - Set the Buttons up to have functionality to move to the different pages
        When a button is pressed, it will navigate to a different page
        */
        Button back2;

        back2 = (Button)findViewById(R.id.backButton2);
        back2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(InfoSwitch.this, MainActivity.class);
                startActivity(intent);
            }
        });
    }
}
