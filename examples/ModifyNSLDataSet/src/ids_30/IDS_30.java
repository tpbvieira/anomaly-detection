/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ids_30;

import java.io.File;
import java.io.IOException;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;


/**
 *
 * @author User
 */
public class IDS_30 {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
        // TODO code application logic here
//        FileIO testIDS = new FileIO();
//        testIDS.readFromFile("C:\\Users\\User\\Downloads\\projectII\\NSL_KDD-master\\NSL_KDD-master\\KDDTrain+.txt");

        DataSetBuilder builder = new DataSetBuilder();
        builder.readFromFile("C:/Users/User/Downloads/projectII/NSL-Processed/Normalized/KDDTrain+_normalized.txt");
        
        
        /* ------------ Classification--------------*/
        /* Load a data set */
//        Dataset train1 = FileHandler.loadDataset(new File("C:\\Users\\User\\Downloads\\project\\NewBuildDataset\\supervised\\Labeled_balanced_100K.txt"), 41, ",");
//        Dataset test1 = FileHandler.loadDataset(new File("C:\\Users\\User\\Downloads\\project\\NewBuildDataset\\attacksLabeled\\Attacklabeled_Imbalanced_100K.txt"), 41, ",");
//        SupervisedLearn.knnClassification(train1, test1);
//
//        Dataset train2 = FileHandler.loadDataset(new File("C:\\Users\\User\\Downloads\\project\\NewBuildDataset\\supervised\\Labeled_balanced_100K.txt"), 41, ",");
//        Dataset test2 = FileHandler.loadDataset(new File("C:\\Users\\User\\Downloads\\project\\NewBuildDataset\\attacksLabeled\\Attacklabeled_Imbalanced_400K.txt"), 41, ",");
//        SupervisedLearn.knnClassification(train2, test2);

        /*SVM classifier*/
        /*
         * Contruct a LibSVM classifier with default settings.
         */



//        SMO smo = new SMO();
//        /* Wrap Weka classifier in bridge */
//        Classifier svm = new WekaClassifier(smo);
//        svm.buildClassifier(data);
//        
//            int correct = 0, wrong = 0;
//            /* Classify all instances and check with the correct class values */
//            for (Instance inst : dataForClassification) {
//                Object predictedClassValue = svm.classify(inst);
//                Object realClassValue = inst.classValue();
//                if (predictedClassValue.equals(realClassValue)) {
//                    correct++;
//                } else {
//                    wrong++;
//                }
//            }
//            double accuracy = ((double) correct / (correct + wrong)) * 100.0;
//            System.out.println("Accuracy : " + accuracy);        
//        Dataset train = FileHandler.loadDataset(new File("C:\\Users\\User\\Downloads\\project\\iris.txt"), 4, ",");
//        Dataset test = FileHandler.loadDataset(new File("C:\\Users\\User\\Downloads\\project\\iris.txt"), 4, ",");
//
//        /* Construct new cross validation instance with the KNN classifier */
//        Classifier knn = new KNearestNeighbors(5);
//        knn.buildClassifier(train);
//
//        CrossValidation cv = new CrossValidation(knn);
//
//        Map<Object, PerformanceMeasure> p = EvaluateDataset.testDataset(knn,test);
//        for (Object o : p.keySet()) {
//            System.out.println(o + ": " + p.get(o).getTPRate());
//            System.out.println(o + ": " + p.get(o).getFPRate());
//            System.out.println("-------------------------");
//        }
//        System.out.println("________________________________");
//        /* Perform 5-fold cross-validation on the data set */
//        Map<Object, PerformanceMeasure> pm = cv.crossValidation(test);
//        for (Object o : pm.keySet()) {
//            System.out.println(o + ": " + pm.get(o).getTPRate());
//            System.out.println(o + ": " + pm.get(o).getFPRate());
//            System.out.println("-------------------------");
//        }

        /* Counters for correct and wrong predictions. */
        /* We create a clustering algorithm, in this case the k-means
         * algorithm with 2 clusters. */
//        Clusterer km = new KMeans(2);
//        /* We cluster the data */
//        Dataset[] clusters = km.cluster(data);
//        /* Create a measure for the cluster quality */
//        ClusterEvaluation sse = new SumOfSquaredErrors();
//        /* Measure the quality of the clustering */
//        double score = sse.score(clusters);
//        System.out.println("SSE :" + score);
    }
}