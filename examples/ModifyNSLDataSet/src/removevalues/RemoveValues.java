package removevalues;

import org.apache.commons.lang.ArrayUtils;

import java.io.*;
import java.util.Arrays;

/**
 * Created by anuradha on 8/1/16.
 */
public class RemoveValues {

    public static void main(String[] args) {
        try {
            String line,b="",c="";
            File file = new File("/home/anuradha/Project/NewDatasets/Testknown/testKnownAttacks1.csv");
           file.createNewFile();
            FileReader fReader = new FileReader("/home/anuradha/Project/NewDatasets/Testknown/testKnownAttacks.csv");
            BufferedReader buffReader = new BufferedReader(fReader);

            FileWriter fw = new FileWriter(file);
            BufferedWriter bw  = new BufferedWriter(fw);

            while((line = buffReader.readLine())!=null) {
               String [] a = line.split(",");
                if(a[a.length-1].equals("Yes")) {
                    a[a.length-1] = "1";
                } else {
                    a[a.length-1] = "0";
                }
                b = Arrays.toString(a);
                c += b.substring(1,b.length()-1)+"\n";
            }
           bw.write(c);
            buffReader.close();

           bw.flush();
            bw.close();

        }catch(IOException e) {
            e.printStackTrace();
        }

    }
}
