package com.example.uteqdetection;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraManager;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.util.Size;
import android.view.Surface;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.uteqdetection.ml.ModelUnquant;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.w3c.dom.Text;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Locale;

import camerax.CameraConnectionFragment;
import camerax.ImageUtils;

public class MainActivity extends AppCompatActivity implements OnSuccessListener<Text>,
        OnFailureListener, ImageReader.OnImageAvailableListener {
    public static int REQUEST_CAMERA = 111;
    Bitmap mSelectedImage;
    ImageView mImageView;
    ArrayList<String> permisosNoAprobados;
    Button btnCamara;
    TextView txtResults;
    int imageSize = 224;

    TextToSpeech t1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        txtResults = findViewById(R.id.txtresults);
        btnCamara = findViewById(R.id.btCamera);
        getPermisos();

        t1 = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int i) {
                if(i != TextToSpeech.ERROR)
                    t1.setLanguage(Locale.ENGLISH);
            }
        });

    }

    void getPermisos(){
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){
            if(checkSelfPermission(android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
                ActivityCompat.requestPermissions(MainActivity.this, new String[] {Manifest.permission.CAMERA},11);
            }
        }

    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        switch (requestCode) {
            case 11:
                if (grantResults.length > 0 && grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(getApplicationContext(), "Es necesario activar los permisos", Toast.LENGTH_SHORT).show();
                    getPermisos();
                }
                break;
        }
    }

    public void abrirCamera(View view) {
        this.setFragment();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && null != data) {
            try {
                if (requestCode == REQUEST_CAMERA)
                    mSelectedImage = (Bitmap) data.getExtras().get("data");
                else
                    mSelectedImage = MediaStore.Images.Media.getBitmap(getContentResolver(), data.getData());

                mImageView.setImageBitmap(mSelectedImage);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Override
    public void onFailure(@NonNull Exception e) {
        txtResults.setText("Error al procesar imagen");
    }

    @Override
    public void onSuccess(Text text) {
    }

    int previewHeight = 0, previewWidth = 0;
    int sensorOrientation;

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    protected void setFragment() {
        final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        String cameraId = null;
        try {
            cameraId = manager.getCameraIdList()[0];
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
        CameraConnectionFragment fragment;
        CameraConnectionFragment camera2Fragment =
                CameraConnectionFragment.newInstance(
                        new CameraConnectionFragment.ConnectionCallback() {
                            @Override
                            public void onPreviewSizeChosen(final Size size, final int rotation) {
                                previewHeight = size.getHeight();
                                previewWidth = size.getWidth();
                                sensorOrientation = rotation - getScreenOrientation();
                            }
                        },
                        this, R.layout.camerafragment, new Size(640, 480));

        camera2Fragment.setCamera(cameraId);
        fragment = camera2Fragment;
        getFragmentManager().beginTransaction().replace(R.id.container, fragment).commit();
    }

    protected int getScreenOrientation() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }

    private boolean isProcessingFrame = false;
    private byte[][] yuvBytes = new byte[3][];
    private int[] rgbBytes = null;
    private int yRowStride;
    private Runnable postInferenceCallback;
    private Runnable imageConverter;
    private Bitmap rgbFrameBitmap;

    @Override
    public void onImageAvailable(ImageReader imageReader) {
        if (previewWidth == 0 || previewHeight == 0) return;
        if (rgbBytes == null) rgbBytes = new int[previewWidth * previewHeight];
        try {
            final Image image = imageReader.acquireLatestImage();
            if (image == null) return;
            if (isProcessingFrame) {
                image.close();
                return;
            }
            isProcessingFrame = true;
            final Image.Plane[] planes = image.getPlanes();
            fillBytes(planes, yuvBytes);
            yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();
            imageConverter = new Runnable() {
                @Override
                public void run() {
                    ImageUtils.convertYUV420ToARGB8888(yuvBytes[0], yuvBytes[1], yuvBytes[2], previewWidth, previewHeight,
                            yRowStride, uvRowStride, uvPixelStride, rgbBytes);
                }
            };
            postInferenceCallback = new Runnable() {
                @Override
                public void run() {
                    image.close();
                    isProcessingFrame = false;
                }
            };
            processImage();
        } catch (final Exception e) {
        }
    }

    protected void fillBytes(final Image.Plane[] planes, final byte[][] yuvBytes) {
        for (int i = 0; i < planes.length; ++i) {
            final ByteBuffer buffer = planes[i].getBuffer();
            if (yuvBytes[i] == null) {
                yuvBytes[i] = new byte[buffer.capacity()];
            }
            buffer.get(yuvBytes[i]);
        }
    }

    private boolean isTTSPlaying = false;

    private void processImage() {

        imageConverter.run();

        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createScaledBitmap(rgbFrameBitmap, imageSize, imageSize, false);

        try {

            ModelUnquant model = ModelUnquant.newInstance(getApplicationContext());
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            rgbFrameBitmap.getPixels(intValues, 0, rgbFrameBitmap.getWidth(), 0, 0, rgbFrameBitmap.getWidth(), rgbFrameBitmap.getHeight());

            // iterate over pixels and extract R, G, and B values. Add to bytebuffer.
            int pixel = 0;
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            ModelUnquant.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputFeature0.getFloatArray();

            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidences.length; i++) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i];
                    maxPos = i;
                }
            }
            String[] classes = {"Auditorio",
                    "Biblioteca"
                    ,"Centro medico"
                    ,"Comedor 1"
                    ,"Comedor 2"
                    ,"Departamento academico"
                    ,"Departamento de investigacion"
                    ,"Departamento de archivos"
                    ,"Facultad Pedagogía"
                    ,"Facultad de ciencias empresariales"
                    ,"Facultad de ciencias sociales y económicas"
                    ,"Instituto de informática"
                    ,"Parqueadero administrativo"
                    ,"Parqueadero de autoridades"
                    ,"Parqueadero de estudiantes"
                    ,"Parqueadero institucional"
                    ,"Polideportivo"
                    ,"Rectorado"
                    ,"Rotonda"
                    ,"Facultad de salud"};

            String s = "";
            for (int i = 0; i < classes.length; i++) {
                s += String.format("%s: %.1f%%\n", classes[i], confidences[i] * 100);
            }

            txtResults.setText(classes[maxPos]);
            String text = txtResults.getText().toString();

            if (!t1.isSpeaking()) {
                t1.speak(text, TextToSpeech.QUEUE_FLUSH,null);
            }


            model.close();

        } catch (IOException e) {
            txtResults.setText("Error al procesar Modelo");
        }
        postInferenceCallback.run();
    }
}

