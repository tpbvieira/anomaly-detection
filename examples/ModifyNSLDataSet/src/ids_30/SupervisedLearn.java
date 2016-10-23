/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ids_30;

/**
 *
 * @author User
 */
public class SupervisedLearn {
/*
    public static void knnClassification(Dataset train, Dataset test) {

        for (int n = 1; n < 33; n = n + 2) {

            Classifier knn = new KNearestNeighbors(n);
            knn.buildClassifier(train);

            Map<Object, PerformanceMeasure> pm = EvaluateDataset.testDataset(knn, test);
            System.out.println("____________" + n + "______________");
            for (Object o : pm.keySet()) {
                System.out.println(o + ": " + pm.get(o).getTPRate());
                System.out.println(o + ": " + pm.get(o).getFPRate());
                System.out.println("-------------------------");
            }

        }

    }

    public static void svmClassification(Dataset train, Dataset test) {

        Classifier svm = new LibSVM().setParameters(null);

        svm.buildClassifier(train);

        Map<Object, PerformanceMeasure> pm = EvaluateDataset.testDataset(svm, test);
        System.out.println("____________" + n + "______________");
        for (Object o : pm.keySet()) {
            System.out.println(o + ": " + pm.get(o).getTPRate());
            System.out.println(o + ": " + pm.get(o).getFPRate());
            System.out.println("-------------------------");
        }

    }
*/
}