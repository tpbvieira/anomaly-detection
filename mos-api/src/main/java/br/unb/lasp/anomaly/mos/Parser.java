package br.unb.lasp.anomaly.mos;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import java.util.StringTokenizer;

import org.apache.commons.lang.time.DateUtils;

import com.univocity.parsers.csv.CsvParser;
import com.univocity.parsers.csv.CsvParserSettings;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;
import br.unb.lasp.matrix.DateUtil;
import br.unb.lasp.matrix.MatrixUtil;

public class Parser {

	public static void main(String[] args) { 
		long initTime, startTime, endTime;
		short windowSize = 60;// minutes		
		//		String folderPath = "/home/thiago/Dropbox/doutorado/tanya/storgrid.log/";
		String filePath = "/home/thiago/Dropbox/doutorado/tanya/storgrid.log/4.1.log";
		SimpleDateFormat sdf = new SimpleDateFormat(DateUtil.usDateTimeMS);

		// Parsing
		initTime = startTime = System.currentTimeMillis();
		HashMap<String,Integer> matrixRowIndeces = new HashMap<String,Integer>();
		HashMap<Long,HashMap<String,Integer>> values = Parser.parseCSV(filePath, sdf, matrixRowIndeces);
		System.out.println(values.size());
		endTime = System.currentTimeMillis();
		System.out.println("ParsingTime: " + (endTime - startTime));

		// Data Modeling
		startTime = System.currentTimeMillis();
		HashMap<Long,Integer> matrixColumnIndeces = new HashMap<Long,Integer>();		
		Matrix[] matrices = Parser.modelIntoMatrices(windowSize, matrixRowIndeces, values, matrixColumnIndeces);
		endTime = System.currentTimeMillis();
		System.out.println("DataModelingTime: " + (endTime - startTime));

		// EigenAnalysis
		startTime = System.currentTimeMillis();
		double[][] largestEigValCov = getLargestEigValCov(matrices);
		endTime = System.currentTimeMillis();
		System.out.println("EigenCovAnalysisTime: " + (endTime - startTime));
		startTime = System.currentTimeMillis();
		double[][] largestEigValCor = getLargestEigValCor(matrices);
		endTime = System.currentTimeMillis();
		System.out.println("EigenCorAnalysisTime: " + (endTime - startTime));

		// MOS Analysis
		startTime = System.currentTimeMillis();
		int mosCov = MOS.edcJAMA(new Matrix(largestEigValCov), windowSize);
		int mosCor = MOS.edcJAMA(new Matrix(largestEigValCor), windowSize);
		endTime = System.currentTimeMillis();
		System.out.println("MOSTime: " + (endTime - startTime));

		System.out.println("MOSCor: " + mosCor);
		System.out.println("MOSCov: " + mosCov);
		System.out.println("TotalTime: " + (endTime - initTime));
	}

	public static double[][] getLargestEigValCor(Matrix[] matrices) {
		double[][] largestEigValCor = new double[1][matrices.length];
		for(int i = 0; i < matrices.length; i++){
			EigenvalueDecomposition eig = EigenAnalysis.eigencorrelation(matrices[i]);
			largestEigValCor[0][i] = MatrixUtil.max(eig.getD());
		}
		return largestEigValCor;
	}

	public static double[][] getLargestEigValCov(Matrix[] matrices) {
		double[][] largestEigValCov = new double[1][matrices.length];
		for(int i = 0; i < matrices.length; i++){
			EigenvalueDecomposition eig = EigenAnalysis.eigencovariance(matrices[i]);
			largestEigValCov[0][i] = MatrixUtil.max(eig.getD());
		}
		return largestEigValCov;
	}

	public static Matrix[] modelIntoMatrices(short windowSize, HashMap<String, Integer> matrixRowIndeces,
			HashMap<Long, HashMap<String, Integer>> values, HashMap<Long, Integer> matrixColumnIndeces) {
		Set<Long> times = values.keySet();
		long firstTime = Collections.min(times);
		long lastTime = Collections.max(times);
		
		int matrixColumnIndex = 0; 
		long currTime = firstTime;
		
		for(;currTime <= lastTime;){
			matrixColumnIndeces.put(currTime,matrixColumnIndex);				
			matrixColumnIndex++;
			currTime = currTime + (1000 * 60);//one more minute
		}

		double[][] arrMatrix = new double[matrixRowIndeces.size()][matrixColumnIndeces.size()];			
		times = values.keySet();
		for(final long time: times){
			HashMap<String,Integer> events = values.get(time);

			int columnNum = matrixColumnIndeces.get(time);
			Set<String> eventsCode = events.keySet();
			for(final String eventCode: eventsCode){
				int rowNum = matrixRowIndeces.get(eventCode);
				arrMatrix[rowNum][columnNum] = events.get(eventCode);
			}

		}

		Matrix matrix = new Matrix(arrMatrix);
		Matrix[] matrices = new Matrix[matrixColumnIndeces.size()/windowSize];
		int startColumn = 0;
		int endColumn = windowSize - 1;
		for(int i = 0; endColumn < matrixColumnIndeces.size(); i++){				
			matrices[i] = matrix.getMatrix(0, matrixRowIndeces.size()-1, startColumn, endColumn);
			startColumn = startColumn + windowSize;
			endColumn = endColumn + windowSize;
		}
		return matrices;
	}

	public static HashMap<Long,HashMap<String,Integer>> parseCSV(String path, SimpleDateFormat sdf, HashMap<String,Integer> matrixRowIndeces){
		CsvParserSettings settings = new CsvParserSettings();
		settings.getFormat().setDelimiter(';');
		settings.getFormat().setLineSeparator("\n");
		CsvParser parser = new CsvParser(settings);

		HashMap<Long,HashMap<String,Integer>> values = new HashMap<Long,HashMap<String,Integer>>();

		try {
			FileReader csvFile = new FileReader(new File(path));
			//	    File [] files = new File(folderPath).listFiles(new FileFilter() {
			//	        @Override
			//	        public boolean accept(File path) {
			//	            if(path.isFile()) {
			//	                //Do something with a single file
			//	                //or just return true to put it in the list
			//	                return true;
			//	            }
			//	            return false;
			//	        }
			//	    });
			
			List<String[]> fileRows = parser.parseAll(csvFile);
			csvFile.close();

			int matrixRowIndex = 0;			
			for(final String[] row: fileRows){
				Date dateTime = sdf.parse(row[0]);				
				StringTokenizer strt = new StringTokenizer(row[5], "'");
				String eventStr =  strt.nextToken();
				Date truncDateTime = DateUtils.truncate(dateTime, Calendar.MINUTE);
				long time = truncDateTime.getTime();
				if(values.containsKey(time)){
					HashMap<String,Integer> events = values.get(time);
					if(events.containsKey(eventStr)){
						int eventCount = events.get(eventStr);
						eventCount++;
						events.put(eventStr, eventCount);
					}else{
						events.put(eventStr, 1);
						if(!matrixRowIndeces.containsKey(eventStr)){
							matrixRowIndeces.put(eventStr, matrixRowIndex);
							matrixRowIndex++;
						}
					}
					values.put(time, events);
				}else{
					HashMap<String,Integer> events = new HashMap<String,Integer>();
					events.put(eventStr, 1);
					matrixRowIndeces.put(eventStr, matrixRowIndex);
					values.put(time, events);
				}
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (ParseException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return values;		
	}

}