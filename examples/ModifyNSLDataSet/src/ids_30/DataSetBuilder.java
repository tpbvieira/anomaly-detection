/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ids_30;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 *
 * @author User
 */
public class DataSetBuilder {

    static int totalAttacks = 0;
    static int totalNormal = 0;

    public static void readFromFile(String path) {
        // TODO code application logic here

        BufferedReader br = null;

        try {

            String sCurrentLine;
            int count = 0; // currently reading line in the buffer
            int total = 0;
            int normal = 0;
            int attacks = 0;

            //           List<String> AttackCollection = new ArrayList<>(Arrays.asList("0.05677300", "-0.43357941", "-0.58681454", "-0.59447629", "2.01818264", "-0.59702024", "-0.55616751", "-0.59192238", "0.71057621", "-0.59702959", "-0.27012861", "-0.59702897", "-0.59687059", "-0.59702772", "-0.59575325", "-0.59639173", "-0.59671097", "-0.59695040", "-0.59702522", "-0.57659886", "-0.59699031", "-0.51530481", "-0.59701026"));

            // Map<String, ArrayList<String>> features = new HashMap<>();
            //       for (int i = 0; i < AttackCollection.size(); i++) {
            br = new BufferedReader(new FileReader(path));
            while ((sCurrentLine = br.readLine()) != null && count
                    <125973 ) {

                List<String> AttributeList = new ArrayList<>(Arrays.asList(sCurrentLine.split(",")));

                //for labeled datasets
                FileIO.writeToFile("C:/Users/User/Downloads/projectII/NSL-Processed/Final/KDDTrain.txt", setLabel(AttributeList, "0.851833458727"));

                //for balanced datasets
//                     String label = AttributeList.get(AttributeList.size() - 1);
//
//                     if (label.equals("1")) {
//                     attacks++;
//                     } else {
//                     normal++;
//                     }
//
//                     if (total <= 87526) {
//
//                     if (label.equals("1") && attacks <= 43763) {
//                     FileIO.writeToFile("C:/Users/User/Downloads/project/NewBuildDataset/labeledUnbalanced.txt", sCurrentLine);
//                     total++;
//                     }
//                     if (label.equals("0") && normal <= 43763) {
//                     FileIO.writeToFile("C:/Users/User/Downloads/project/NewBuildDataset/labeledUnbalanced.txt", sCurrentLine);
//                     total++;
//                     }
//                    
//                     } else {
//                     break;
//                     }



//                     count++;  
//
//                    List<String> AttributeList = new ArrayList<>(Arrays.asList(sCurrentLine.split(",")));
//                    //for balanced datasets   
//                    String label = AttributeList.get(AttributeList.size() - 1);

                //     if (AttackCollection.get(i).equals(label)) {
                //  FileIO.writeToFile("C:/Users/User/Downloads/project/NewBuildDataset/latest/neptunetest_28000.txt", sCurrentLine);

                //     }

//                    if (count > 494020) {
//                        count = 0;
//                    }
//                }
                count++;
            }

        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (br != null) {
                    br.close();
                    System.out.println("TotalAttacks: " + totalAttacks);
                    System.out.println("TotalNormal: " + totalNormal);
                    System.out.println("DataSetBuilder Done...");
                }
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }

    private static String setLabel(List<String> AttributeList, String normalval) {

        String label = AttributeList.get(AttributeList.size() - 1);
        if (label.equals(normalval)) {
            AttributeList.set(AttributeList.size() - 1, "0");  //if normal then 0
            totalNormal++;
        } else {
            AttributeList.set(AttributeList.size() - 1, "1");
            //AttributeList.remove(AttributeList.size() - 1);
            totalAttacks++;
        }

        StringBuilder commaSepValueBuilder = new StringBuilder();
        for (int i = 0; i < AttributeList.size(); i++) {

            //append the value into the builder
            commaSepValueBuilder.append(AttributeList.get(i));

            //if the value is not the last element of the list
            //then append the comma(,) as well
            if (i != AttributeList.size() - 1) {
                commaSepValueBuilder.append(",");
            }

        }


        return commaSepValueBuilder.toString();
    }

}