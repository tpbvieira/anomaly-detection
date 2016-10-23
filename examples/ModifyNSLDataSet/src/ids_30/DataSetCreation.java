package ids_30;

import org.apache.commons.lang3.StringUtils;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by anuradha on 8/12/16.
 */
public class DataSetCreation {
    public void removeColumn(int[] colId, String fileNameRead, String fileNameWrite) {
        try {
        File newFile = new File(fileNameWrite);
            newFile.createNewFile();
            FileReader fileReader = new FileReader(fileNameRead);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            String str = "",currentLine;
            while((currentLine = bufferedReader.readLine()) !=null) {
                int cnt =0;
                ArrayList<String> list = null;
                String [] line = currentLine.split(",");
                list = new ArrayList<String>(Arrays.asList(line));

                for(int i:colId) {
                    list.remove(i-cnt);
                    cnt++;
                }
                str += StringUtils.join(list, ',')+"\n";
            }

            bufferedReader.close();
            FileWriter fileWriter = new FileWriter(newFile);
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
            bufferedWriter.write(str);
            bufferedWriter.flush();
            bufferedWriter.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void selectNormalDataSet(String fileNameRead, String fileNameWrite) {
        try {
            File newFile = new File(fileNameWrite);
            newFile.createNewFile();

            FileReader fileReader = new FileReader(fileNameRead);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            int i=0;
            String writeStr = "";
            while(bufferedReader.readLine() != null) {
                String str = bufferedReader.readLine();
               if(str !=null) {
                   if(str.charAt(str.length()-1) == '0') {
                        writeStr += str+"\n";
                   }
               }

            }
            bufferedReader.close();

            FileWriter fileWriter = new FileWriter(newFile);
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
            bufferedWriter.write(writeStr);
            bufferedWriter.flush();
            bufferedWriter.close();
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    public static void main(String[] args) {
        DataSetCreation dsc = new DataSetCreation();
        int [] arr = {3,11,22,24,25,26,27,28,29,31,32,33,34,37,38,39,40};
        //KDDTrain
        //KDDTest
        dsc.removeColumn(arr,"/home/anuradha/Project/processed_nsl_kdd/KDDTrain","/home/anuradha/Project/processed_nsl_kdd/KDDTrain_above_5000.txt");
        //dsc.selectNormalDataSet("/home/anuradha/Project/NSL_processed/KDDTrain.txt","/home/anuradha/Project/NSL_processed/KDDTrainNormal.txt");
    }
}
