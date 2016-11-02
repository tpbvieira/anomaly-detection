package br.unb.lasp.anomaly.mos.cases;

import java.text.SimpleDateFormat;
import java.util.HashMap;

import Jama.Matrix;
import br.unb.lasp.anomaly.mos.MOS;
import br.unb.lasp.matrix.DateUtil;
import br.unb.lasp.matrix.MatrixUtil;
import br.unb.lasp.parser.Parser;

public class Storegrid {

	public static void main(String[] args) { 
		long initTime, startTime, endTime;
		short windowSize = 60;// minutes
		
		//		String folderPath = "/home/thiago/Dropbox/doutorado/tanya/storgrid.log/";
		String filePath = "/home/thiago/Dropbox/doutorado/tanya/storgrid.log/4.1.log";
		SimpleDateFormat sdf = new SimpleDateFormat(DateUtil.usDateTimeMS);

		// Parsing
		initTime = startTime = System.currentTimeMillis();
		HashMap<String,Integer> matrixRowIndeces = new HashMap<String,Integer>();
		HashMap<Long,HashMap<String,Integer>> values = Parser.csvToPortTimeCountMap(filePath, sdf, matrixRowIndeces);
		System.out.println(values.size());
		endTime = System.currentTimeMillis();
		System.out.println("ParsingTime: " + (endTime - startTime));

		// Data Modeling
		startTime = System.currentTimeMillis();
		HashMap<Long,Integer> matrixColumnIndeces = new HashMap<Long,Integer>();		
		Matrix[] matrices = MatrixUtil.modelIntoMatrices(windowSize, matrixRowIndeces, values, matrixColumnIndeces);
		endTime = System.currentTimeMillis();
		System.out.println("DataModelingTime: " + (endTime - startTime));

		// EigenAnalysis
		startTime = System.currentTimeMillis();
		double[][] largestEigValCov = MatrixUtil.getLargestEigValCov(matrices);
		endTime = System.currentTimeMillis();
		System.out.println("EigenCovAnalysisTime: " + (endTime - startTime));
		startTime = System.currentTimeMillis();
		double[][] largestEigValCor = MatrixUtil.getLargestEigValCor(matrices);
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

}