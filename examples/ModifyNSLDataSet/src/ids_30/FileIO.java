/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ids_30;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

/**
 *
 * @author User
 */
public class FileIO {

    //write preprocessed data to a file
    public static void writeToFile(String path, String content) {

        try {

            File file = new File(path);

            // if file doesnt exists, then create it
            if (!file.exists()) {
                file.createNewFile();
            }

            FileWriter fw = new FileWriter(file.getAbsoluteFile(), true);
            BufferedWriter bw = new BufferedWriter(fw);
            bw.write(content.concat("\n"));
            bw.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //read data set
    public static void readFromFile(String path) {
        // TODO code application logic here
        List<String> AttackCollection = new ArrayList<>(Arrays.asList("back", "buffer_overflow", "ftp_write", "guess_passwd", "imap", "ipsweep", "land", "loadmodule", "multihop", "neptune", "nmap", "perl", "phf", "pod", "portsweep", "rootkit", "satan", "smurf", "spy", "teardrop", "warezclient", "warezmaster", "normal", "unknown"));
        BufferedReader br = null;

        try {

            String sCurrentLine;
            int count = 0;

            br = new BufferedReader(new FileReader(path));
            Map<String, ArrayList<String>> features = new HashMap<>();

            while ((sCurrentLine = br.readLine()) != null && count < 125973) {

                List<String> AttributeList = new ArrayList<>(Arrays.asList(sCurrentLine.split(",")));
                AttributeList.remove(AttributeList.size() - 1); //for NSL-KDD only

                for (int i = 0; i < AttributeList.size(); i++) {

                    ArrayList<String> list;
                    if (features.containsKey(String.valueOf(i))) {
                        list = features.get(String.valueOf(i));

                    } else {
                        list = new ArrayList<String>();

                    }
                    list.add(AttributeList.get(i));
                    features.put(String.valueOf(i), list);
                }

                count++;

            }

            features.put(String.valueOf(1), getNumericValues(features.get(String.valueOf(1)), 1));
            features.put(String.valueOf(2), getNumericValues(features.get(String.valueOf(2)), 1));
            features.put(String.valueOf(3), getNumericValues(features.get(String.valueOf(3)), 1));
            features.put(String.valueOf(features.size() - 1), getNumericValues(features.get(String.valueOf(features.size() - 1)), features.size() - 1));

            //Normalization
            for (int i = 0; i < features.size(); i++) {

                features.put(String.valueOf(i), Normalize(features.get(String.valueOf(i)), i));
            }

            writeString(features);

        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (br != null) {
                    br.close();
                    System.out.println("Normalized Done...");
                }
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }

    public static ArrayList<String> Normalize(ArrayList<String> strList, int featureID) {

        List<Double> numericList = new ArrayList<>();
        ArrayList<String> normalizedList = new ArrayList<>();

        for (int i = 0; i < strList.size(); i++) {
            numericList.add(Double.valueOf(strList.get(i)));
        }

//        double max = Collections.max(numericList);
//        double min = Collections.min(numericList);
        double mean = getMean(numericList);
        double stdDeviation = getStdDev(numericList, mean);

        for (int i = 0; i < numericList.size(); i++) {

//            double nor = (numericList.get(i) - min) / (max - min);
            double nor = 0.0;
            if (stdDeviation != 0) {
                nor = (numericList.get(i) - mean) / stdDeviation;
            }
            normalizedList.add(String.format("%.12f", nor));
        }

        //find correct label
        if (featureID == 41) {

            ArrayList<String> compare = new ArrayList<String>(new LinkedHashSet<String>(strList));//to compare after & before norm
            ArrayList<String> ncompare = new ArrayList<String>(new LinkedHashSet<String>(normalizedList));
            HashMap<String, String> cmap = new HashMap<>();

            for (int i = 0; i < compare.size(); i++) {
                cmap.put(compare.get(i), ncompare.get(i));
            }
            System.out.println("__________________Normalized_____________________");
            for (String key : cmap.keySet()) {
                System.out.println(key + " : " + cmap.get(key));
            }
        }
        return normalizedList;

    }

    public static ArrayList<String> getNumericValues(ArrayList<String> list, int featureID) {
        /* Remove duplicates*/
        ArrayList<String> listNew = new ArrayList<String>(new LinkedHashSet<String>(list));

        /* Convert nominal values to binary values */
        String binaryVal = null;
        String[] binaries = new String[listNew.size()];
        if (listNew.size() != 0) {
            binaryVal = "0";
        }
        for (int i = 0; i < listNew.size() - 1; i++) {
            binaryVal += "0";
        }
        for (int i = 0; i < listNew.size(); i++) {
            binaries[i] = binaryVal;
        }
        for (int i = 0; i < listNew.size(); i++) {
            StringBuilder binary = new StringBuilder(binaries[i]);
            binary.setCharAt(i, '1');
            binaries[i] = binary.toString();
        }

        /* Convert binary values to numeric values */
        String[] numeric = new String[binaries.length];
        for (int i = 0; i < binaries.length; i++) {
//            BigInteger bigInt = new BigInteger(binaries[i]);
            numeric[i] = String.valueOf(binaryToInteger(binaries[i]));
//            numeric[i] = String.valueOf(bigDec);
        }

        /* Fill hash map with nominal and numeric values, nominal values as keys */
        HashMap<String, String> map = new HashMap<>();

        for (int i = 0; i < numeric.length; i++) {
            map.put(listNew.get(i), numeric[i]);
        }
        if (featureID == 41) {

            for (String key : map.keySet()) {
                System.out.println(key + " : " + map.get(key));
            }

        }

//        /* Convert list to array */
        ArrayList<String> attributeArr = new ArrayList<>();
//        attributeArr = list.toArray(attributeArr);

        /* Replcae nominal with numeric values */
        for (int i = 0; i < list.size(); i++) {
            for (String key : map.keySet()) {
                if (key.equals(list.get(i))) {
                    attributeArr.add(map.get(key));
                }
            }
        }

        return attributeArr;
    }

    public static void writeString(Map<String, ArrayList<String>> features) {

        for (int i = 0; i < 125973; i++) {
            StringBuilder commaSepValueBuilder = new StringBuilder();
            for (int j = 0; j < features.size(); j++) {

                //normalization
                //features.put(String.valueOf(i), Normalize(features.get(String.valueOf(i))));
                //append the value into the builder
                commaSepValueBuilder.append(features.get(String.valueOf(j)).get(i));

                //if the value is not the last element of the list
                //then append the comma(,) as well
                if (j != features.size() - 1) {
                    commaSepValueBuilder.append(",");
                }
            }
            writeToFile("C:/Users/User/Downloads/projectII/NSL-Processed/Normalized/KDDTrain+_normalized.txt", commaSepValueBuilder.toString());

        }

    }

    public static double getMean(List<Double> data) {

        double sum = 0.0;
        for (double a : data) {
            sum += a;
        }
        return sum / data.size();
    }

    public static double getStdDev(List<Double> data, double mean) {

        double temp = 0;
        double varience = 0;
        for (double a : data) {
            temp += (a - mean) * (a - mean);
        }
        varience = temp / data.size();

        return Math.sqrt(varience);
    }

    public static BigInteger binaryToInteger(String binary) {
        char[] numbers = binary.toCharArray();
        BigInteger result = new BigInteger("0");
        int count = 0;
        BigInteger base = new BigInteger("2");
        for (int i = numbers.length - 1; i >= 0; i--) {
            if (numbers[i] == '1') {
                result = result.add(base.pow(count));
            }
            count ++;
        }
        return result;
    }

}