package com.kislay.keras.javaimport;

import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;


public class ImportFromKeras {
    public static Logger logger = LoggerFactory.getLogger(ImportFromKeras.class);

    public static void main(String args[]) throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {
        MultiLayerNetwork model = org.deeplearning4j.nn.modelimport.keras.KerasModelImport.importKerasSequentialModelAndWeights("/home/kislay/deep-learning/data/model/ann_kislay.json", "/home/kislay/deep-learning/data/model/ann_kislay");


        BufferedReader read = new BufferedReader(new FileReader("/home/kislay/deep-learning/data/test.csv"));
        String line = "";
        String data[][] = new String[2000][10];
        int correctLabel[][] = new int[2000][2];
        int counter = 0;
        int count_one = 0;
        while ((line = read.readLine()) != null) {
            String cols[] = line.split(",");
            for (int i = 3; i < cols.length - 1; i++) {
                data[counter][i - 3] = cols[i];

            }
            correctLabel[counter][1] = Integer.parseInt(cols[cols.length - 1]);
            correctLabel[counter][0] = Integer.parseInt(cols[1]);

            counter++;
            if (cols[cols.length - 1].equals("1")) {
                count_one++;
            }
        }

        for (int i = 0; i < 5; i++)
            System.out.println(Arrays.toString(data[i]));

        for (int i = 0; i < 2000; i++) {

            switch (data[i][1]) {
                case "Germany":
                    data[i][1] = "1";
                    break;
                case "France":
                    data[i][1] = "0";
                    break;
                case "Spain":
                    data[i][1] = "2";
                    break;
            }

            switch (data[i][2]) {
                case "Male":
                    data[i][2] = "1";
                    break;
                case "Female":
                    data[i][2] = "0";

            }

        }

        double dataDouble[][] = new double[2000][10];
        for (int i = 0; i < 2000; i++) {
            for (int j = 0; j < 10; j++) {
                dataDouble[i][j] = Float.parseFloat(data[i][j]);
            }
        }

        System.out.println("Converted double array looks like");
        for (int i = 0; i < 5; i++)
            System.out.println(Arrays.toString(data[i]));

        //Perform one hot encoder
        double dataFloatOneHot[][] = new double[2000][11];
        for (int i = 0; i < 2000; i++) {
            for (int j = 3; j < 11; j++) {
                dataFloatOneHot[i][j] = dataDouble[i][j - 1];
            }
            dataFloatOneHot[i][2] = dataDouble[i][0]; //credit score
            if (dataDouble[i][1] == 1)
                dataFloatOneHot[i][0] = 1; // Germany
            else if (dataDouble[i][1] == 2) {
                dataFloatOneHot[i][1] = 1; // Spain

            }
        }

        System.out.println("onehotencoded float Result");
        for (int i = 0; i < 5; i++)
            System.out.println(Arrays.toString(dataFloatOneHot[i]));


        INDArray array = Nd4j.create(dataFloatOneHot);
        DataSet sd = new DataSet(array, null);

        //normalize approach
        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fit(sd);
        normalizer.transform(sd);


        System.out.println(">>" + array);

        INDArray predict = model.output(array);

        //System.out.println(       model.f1Score(sd.get(1)));
        double[] toDoubleMatrix = predict.toDoubleVector();

        int calculatedLabel[] = new int[2000];
        for (int i = 0; i < toDoubleMatrix.length; i++) {
            if (toDoubleMatrix[i] > 0.5) {
                calculatedLabel[i] = 1;
                System.out.println((i + 1) + "--> " + toDoubleMatrix[i]);

            }
        }

        int per = 0;

        for (int i = 0; i < 2000; i++) {
            if (correctLabel[i][1] == calculatedLabel[i]) {
                if(calculatedLabel[i]==1)
                System.out.println(correctLabel[i][0]+" is leaving");
                per++;
            }
        }
        System.out.println("Percetnage Match -->"+(((float)per)/2000)*100+"%" );



    }

}
