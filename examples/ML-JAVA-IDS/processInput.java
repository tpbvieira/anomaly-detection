import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.Scanner;

public class processInput {
	public static void main(String[] args) throws FileNotFoundException {
		FileInputStream fs = new FileInputStream(new File("nsl-kdd-train.arff"));
		PrintWriter fo = new PrintWriter("data.csv");
		//PrintWriter fr = new PrintWriter("result.csv");
		Scanner sc = new Scanner(fs);
		sc.next();
		int c=0;
		while(sc.hasNext()){
			c++;
			String[] tmp = sc.next().split(",");
			//if(tmp[41].equals("normal"))fr.println("0");
			//else fr.println("1");
			//fr.flush();
			for(int i=0;i<40;i++)fo.print(tmp[i]+",");
			fo.print(tmp[40]+"\n");
			fo.flush();
		}
		System.out.println(c);
	}
}
