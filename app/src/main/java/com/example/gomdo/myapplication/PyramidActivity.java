package com.example.gomdo.myapplication;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class PyramidActivity extends AppCompatActivity {

    private final int SELECT_PHOTP=1;
    private ImageView ivImage;
    Mat src;
    static int ACTION_MODE= 0;
    static final int MODE_NONE=0,
            MODE_GAUSSIAN_PYR_UP = 1, MODE_GAUSSIAN_PYR_DOWN = 2, MODE_LAPLACIAN_PYR = 3;

    private boolean srcSelected =false;
    Button  bGaussianPyrUp, bGaussianPyrDown, bLaplacianPyr;

    private BaseLoaderCallback mOpenCvCallBack = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:

                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }

    };
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_pyramid);
//        getActionBar().setDisplayHomeAsUpEnabled(true);
        ivImage= (ImageView) findViewById(R.id.ivImage);
        bGaussianPyrUp = (Button) findViewById(R.id.bGaussianPyrUp);
        bGaussianPyrDown = (Button) findViewById(R.id.bGaussianPyrDown);
        bLaplacianPyr = (Button) findViewById(R.id.bLaplacianPyr);
        bGaussianPyrUp.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                ACTION_MODE = MODE_GAUSSIAN_PYR_UP;
                executeTask();
            }
        });
        bGaussianPyrDown.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                ACTION_MODE = MODE_GAUSSIAN_PYR_DOWN;
                executeTask();
            }
        });

        bLaplacianPyr.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                ACTION_MODE = MODE_LAPLACIAN_PYR;
                executeTask();
            }
        });
        if (!srcSelected){
            bGaussianPyrDown.setEnabled(false);
            bGaussianPyrUp.setEnabled(false);
            bLaplacianPyr.setEnabled(false);
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_pyramid, menu);
        return true;

    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();

        if (id == R.id.action_load_first_image) {
            Intent photoPickerIntent = new Intent(Intent.ACTION_PICK);
            photoPickerIntent.setType("image/*");
            startActivityForResult(photoPickerIntent, SELECT_PHOTP);
            return true;
        }


        return super.onOptionsItemSelected(item);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent imageReturnedIntent) {
        super.onActivityResult(requestCode, resultCode, imageReturnedIntent);
        switch (requestCode){
            case SELECT_PHOTP:
                if (requestCode != RESULT_OK){

                    try {
                        final Uri imageUri = imageReturnedIntent.getData();
                        final InputStream imageStream = getContentResolver().openInputStream(imageUri);
                        final Bitmap selectedImage = BitmapFactory.decodeStream(imageStream);
                        src = new Mat(selectedImage.getHeight(),selectedImage.getWidth(), CvType.CV_8UC4);
                        Utils.bitmapToMat(selectedImage,src);
                        srcSelected=true;
                        bGaussianPyrUp.setEnabled(true);
                        bGaussianPyrDown.setEnabled(true);
                        bLaplacianPyr.setEnabled(true);

                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    }
                }
                break;
        }
    }

    private void executeTask() {
        if (srcSelected){
            new AsyncTask<Void,Void,Bitmap>(){
                @Override
                protected void onPreExecute() {
                    super.onPreExecute();
                }
                @Override
                protected Bitmap doInBackground(Void... params) {
                    Mat srcRes = new Mat();
                    switch (ACTION_MODE){
                        case MODE_GAUSSIAN_PYR_UP:
                            Imgproc.pyrUp(src,srcRes);
                            break;
                        case MODE_GAUSSIAN_PYR_DOWN:
                            Imgproc.pyrDown(src,srcRes);
                            break;
                        case MODE_LAPLACIAN_PYR:
                            Imgproc.pyrDown(src,srcRes);
                            Imgproc.pyrUp(srcRes,srcRes);
                            Core.absdiff(srcRes,src,srcRes);
                            break;
                    }if (ACTION_MODE != 0){
                        Bitmap image = Bitmap.createBitmap(srcRes.cols(),srcRes.rows(),Bitmap.Config.ARGB_8888);
                        Utils.matToBitmap(srcRes,image);
                        FileOutputStream out = null;
                        try {
                            out=new FileOutputStream(Environment.getExternalStorageDirectory() +"/Users/yifanyang/Desktop/NDK/TrackingTarget/app/src/main/java/com/example/yifanyang/trackingtarget"+ACTION_MODE +".png");
                            image.compress(Bitmap.CompressFormat.PNG,100,out);

                        } catch (FileNotFoundException e) {
                            e.printStackTrace();
                        }finally {
                            if (out != null){
                                try {
                                    out.close();
                                } catch (IOException e) {
                                    e.printStackTrace();
                                }
                            }
                        }
                        Imgcodecs.imwrite(Environment.getExternalStorageDirectory() +"/Users/yifanyang/Desktop/NDK/TrackingTarget/app/src/main/java/com/example/yifanyang/trackingtarget"+ACTION_MODE +"-imwrite.png",srcRes);

                        Mat src1= new Mat();
                        Imgproc.cvtColor(srcRes,src1,Imgproc.COLOR_BGR2BGRA);
                        Imgcodecs.imwrite(Environment.getExternalStorageDirectory() + "/Users/yifanyang/Desktop/NDK/TrackingTarget/app/src/main/java/com/example/yifanyang/trackingtarget" + ACTION_MODE + "-imwriteBGRA.png", src1);

                        Imgproc.cvtColor(srcRes,src1,Imgproc.COLOR_BGR2BGRA);
                        Imgcodecs.imwrite(Environment.getExternalStorageDirectory() + "/Users/yifanyang/Desktop/NDK/TrackingTarget/app/src/main/java/com/example/yifanyang/trackingtarget" + ACTION_MODE + "-imwriteBGRA.png", src1);

                        return image;
                    }
                    return null;
                }

                @Override
                protected void onPostExecute(Bitmap bitmap) {
                    super.onPostExecute(bitmap);
                    if (bitmap != null){
                        ivImage.setImageBitmap(bitmap);
                    }
                }
            }.execute();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0,this,mOpenCvCallBack);
    }
}
