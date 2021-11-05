package com.example.celebclassifier;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.w3c.dom.Text;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity {
    protected Interpreter tflite ;
    private MappedByteBuffer tfliteModel ;
    private TensorImage inputImageBuffer ;
    private int imageSizeX ;
    private int imageSizeY ;
    private TensorBuffer outputProbabilityBuffer ;
    private TensorProcessor probabilityProcessor ;
    private static final float IMAGE_MEAN = 0.0f;
    private static final float IMAGE_STD = 1.0f;
    private static final float PROBABILITY_MEAN = 0.0f;
    private static final float PROBABILITY_STD = 255.0f;
    private Bitmap bitmap ;
    private List<String> labels ;
    ImageView imageView ;
    Uri imageUri ;
    Button btnClassify ;
    TextView prediction ;
   TextView toShow ;
   TextView probable ;
   TextView per ;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = (ImageView) findViewById(R.id.imageView);
        btnClassify = (Button)findViewById(R.id.classify);
        prediction = (TextView) findViewById(R.id.predictionTxt);
        toShow = (TextView) findViewById(R.id.celebHead);
        probable = (TextView)findViewById(R.id.probable) ;
        per = (TextView)findViewById(R.id.probableper);
        imageView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent() ;
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT) ;
                startActivityForResult(Intent.createChooser(intent,"Select Picture"),12);

            }
        });
        try{
            tflite = new Interpreter((loadmodelfile(MainActivity.this)));
        }catch (IOException e){
            e.printStackTrace();
        }
        btnClassify.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                int imageTensorIndex = 0 ;
                int[] imageshape = tflite.getInputTensor(imageTensorIndex).shape();
                imageSizeX = imageshape[1];
                imageSizeY =imageshape[2];
                DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();

                int probabilityTensorIndex = 0;
                int[] probabilityShape =tflite.getOutputTensor(probabilityTensorIndex).shape();
                DataType probabilityDataType = tflite.getInputTensor(probabilityTensorIndex).dataType();

                inputImageBuffer = new TensorImage(imageDataType);
                outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape,probabilityDataType);
                probabilityProcessor = new TensorProcessor.Builder().add(getPostProcessorNormalizeOP()).build();

                inputImageBuffer = loadImage(bitmap);
                tflite.run(inputImageBuffer.getBuffer(),outputProbabilityBuffer.getBuffer().rewind());
                showresults();
            }
        });
    }
    private TensorImage loadImage(final Bitmap bitmap){
        inputImageBuffer.load(bitmap);

        int cropSize = Math.min(bitmap.getWidth(),bitmap.getHeight());
        //imageprocessor
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeWithCropOrPadOp(cropSize,cropSize))
                .add(new ResizeOp(imageSizeX,imageSizeY,ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                .add(getPreProcessorNormalizeOP())
                .build();
        return imageProcessor.process(inputImageBuffer);
    }
    //loading tflite file
    private MappedByteBuffer loadmodelfile(Activity activity) throws IOException{
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd("model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startoffset = fileDescriptor.getStartOffset();
        long declareLength =fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startoffset,declareLength);
    }
    private TensorOperator getPreProcessorNormalizeOP(){
        return new NormalizeOp(IMAGE_MEAN,IMAGE_STD);
    }
    private TensorOperator getPostProcessorNormalizeOP(){
        return new NormalizeOp(PROBABILITY_MEAN,PROBABILITY_STD);
    }
private  void showresults(){
        try{
            labels = FileUtil.loadLabels(MainActivity.this,"labels.txt");

        }catch (IOException e){
            e.printStackTrace();
        }
        Map<String,Float> labelesProbability = new TensorLabel(labels,probabilityProcessor.process(outputProbabilityBuffer))
                .getMapWithFloatValue();
//        float maxValueinMap = (Collections.max(labelesProbability.values()));
//        for(Map.Entry<String,Float>entry : labelesProbability.entrySet()){
//            String[] label = labelesProbability.keySet().toArray(new String[0]);
//            Float[] label_probability = labelesProbability.values().toArray(new Float[0]);
//
//        }
//
    String resclass = "None" ;
    Float maxval = 0.0f;
    for(Map.Entry m:labelesProbability.entrySet()){
//        System.out.println(m.getKey()+" "+m.getValue());
       if((Float)m.getValue() > maxval){
           maxval =(Float) m.getValue() ;
           resclass = m.getKey().toString();
       }
    }

    prediction.setText(resclass);


   Float nmax = maxval * 100 ;
   String toDisp = nmax.toString() + "%" ;
   per.setText(toDisp);
}
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode == 12 && resultCode == RESULT_OK && data != null){
            imageUri = data.getData();
            try{
                bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(),imageUri);
                imageView.setImageBitmap(bitmap);

            }catch (IOException e){
                e.printStackTrace();
            }

        }
    }
}