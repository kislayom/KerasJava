package com.kislay.keras.javaimport;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class ImportFromKerasCSV {


    public static Logger logger = LoggerFactory.getLogger(ImportFromKeras.class);

    public static void main(String args[]) throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException, InterruptedException {
        MultiLayerNetwork model = org.deeplearning4j.nn.modelimport.keras.KerasModelImport.importKerasSequentialModelAndWeights("/home/kislay/deep-learning/data/model/ann_kislay.json", "/home/kislay/deep-learning/data/model/ann_kislay");

        RecordReader reader= new CSVRecordReader(0,",");
        reader.initialize(new FileSplit(new File("/home/kislay/deep-learning/data/Churn_Modelling_test.csv")));






    }
}