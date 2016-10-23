package labels;

import java.io.*;

/**
 * Created by anuradha on 8/1/16.
 */
public class LabelFile {

    public static void main(String[] args) {
        try {
            String line,a="",c="";
            File file = new File("/home/anuradha/Project/NSL_KDD_master/Modified/label.txt");
            file.createNewFile();
            FileReader fr = new FileReader("/home/anuradha/Project/NSL_KDD_master/Modified/KDDTest.csv");
            BufferedReader br = new BufferedReader(fr);

            FileWriter fw = new FileWriter(file);
            BufferedWriter bw = new BufferedWriter(fw);

            while((line = br.readLine())!=null) {
                String [] arr = line.split(",");
                    a += (arr[arr.length-1]+" ");
            }
            bw.write(a);
            br.close();
            bw.flush();
            bw.flush();
        } catch(IOException e) {
            e.printStackTrace();
        }
    }


}
