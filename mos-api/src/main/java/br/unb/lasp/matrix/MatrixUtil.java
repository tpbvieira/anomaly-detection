package br.unb.lasp.matrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.moment.GeometricMean;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.ojalgo.array.Array2D;
import org.ojalgo.matrix.PrimitiveMatrix;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;
import br.unb.lasp.anomaly.mos.EigenAnalysis;
import br.unb.lasp.parser.Parser;

/**
 * @author thiago
 *
 */
public class MatrixUtil{
	
	/**
	 * Prints into br.unb.lasp.matrix format
	 * 
	 * @param br.unb.lasp.matrix
	 * @param msg
	 */	
	public static void printMatrix(Matrix matrix, String msg){
		System.out.println(msg);
		matrix.print(matrix.getColumnDimension(), 4);    	
	}


	/**
	 * Calculates the geometric mean of a sample. For vectors, it is the geometric mean of the elements into the vector. 
	 * For matrices, it is a row vector containing the geometric means of each column.
	 * 
	 * @param br.unb.lasp.matrix
	 * @return
	 */
	public static Matrix geomean(Matrix matrix){
		int rowSize = matrix.getRowDimension();
		int columnSize = matrix.getColumnDimension();
		RealMatrix realMatrix = MatrixUtils.createRealMatrix(matrix.getArray());
		GeometricMean geomean = new GeometricMean();
		double[] geomeanArray = null;

		if (columnSize > 1 && rowSize > 1){// Matrix
			geomeanArray = new double[matrix.getColumnDimension()];
			for(int i = 0; i < columnSize; i++){
				geomeanArray[i] = geomean.evaluate(realMatrix.getColumn(i));
			}    		
		}else if (columnSize == 1 && rowSize > 1){ // Column vector
			geomeanArray = new double[matrix.getColumnDimension()];
			geomeanArray[0] = geomean.evaluate(realMatrix.getColumn(0));
		}else if (columnSize > 1 && rowSize == 1){ // Row vector
			geomeanArray = new double[matrix.getRowDimension()];
			geomeanArray[0] = geomean.evaluate(realMatrix.getRow(0));
		}else{
			geomeanArray = new double[1];
			geomeanArray[0] = geomean.evaluate(realMatrix.getRow(0));
		}    	

		double[][] geomeanMatrix = {geomeanArray};
		Matrix result = new Matrix(geomeanMatrix);

		return result;
	}

	/**
	 * Calculates the mean of a sample. For vectors, it is the mean of the elements into the vector. 
	 * For matrices, it is a row vector containing the means of each column.
	 * 
	 * @param br.unb.lasp.matrix
	 * @return
	 */
	public static Matrix mean(Matrix matrix){
		int rowSize = matrix.getRowDimension();
		int columnSize = matrix.getColumnDimension();
		RealMatrix realMatrix = MatrixUtils.createRealMatrix(matrix.getArray());
		Mean mean = new Mean();
		double[] meanArray = null;

		if (columnSize > 1 && rowSize > 1){// Matrix
			meanArray = new double[matrix.getColumnDimension()];
			for(int i = 0; i < columnSize; i++){
				meanArray[i] = mean.evaluate(realMatrix.getColumn(i));
			}    		
		}else if (columnSize == 1 && rowSize > 1){ // Column vector
			meanArray = new double[matrix.getColumnDimension()];
			meanArray[0] = mean.evaluate(realMatrix.getColumn(0));
		}else if (columnSize > 1 && rowSize == 1){ // Row vector
			meanArray = new double[matrix.getRowDimension()];
			meanArray[0] = mean.evaluate(realMatrix.getRow(0));
		}else{
			meanArray = new double[1];
			meanArray[0] = mean.evaluate(realMatrix.getRow(0));
		}    	

		double[][] meanMatrix = {meanArray};
		Matrix result = new Matrix(meanMatrix);

		return result;
	}

	/**
	 * Read file, with br.unb.lasp.matrix separated by spaces, into a br.unb.lasp.matrix Matrix
	 * 
	 * @param filename
	 * @return
	 * @throws IOException
	 */
	public static Matrix readMatrixFromFile(String filename) throws IOException {
		// example
		BufferedReader buffer = new BufferedReader(new FileReader(filename));
		double[][] matrix;
		String line;
		List<double[]> rows = new ArrayList<double[]>();
		while ((line = buffer.readLine()) != null) {
			String[] rowValues = line.trim().split("\\s+");
			double[] row = new double[rowValues.length];
			for (int col = 0; col < rowValues.length; col++) {
				row[col] = Double.parseDouble(rowValues[col]);
			}
			rows.add(row);
		}
		buffer.close();

		matrix = new double[rows.size()][rows.get(0).length];
		int i = 0;
		for(double[] row : rows){
			matrix[i] = row;
			i++;
		}
		return new Matrix(matrix);
	}


	public static Matrix primitiveMatrixToJamaMatrix(PrimitiveMatrix pMatrix) {
		Matrix eigCorMatrix;
		Array2D.Factory<Double> array2DFactory = Array2D.PRIMITIVE;
		Array2D<Double> tmpArray2D = array2DFactory.copy((org.ojalgo.matrix.MatrixUtils.wrapPrimitiveAccess2D(pMatrix)));
		eigCorMatrix = new Matrix(tmpArray2D.toRawCopy());
		return eigCorMatrix;
	}

	
	public static double max(PrimitiveMatrix pMatrix) {
		return Collections.max(pMatrix.toListOfElements());
	}

	
	public static double max(Matrix values) {
		double[] vals = values.getColumnPackedCopy();
		double max = Double.MIN_VALUE;
		for(int i = 0; i < vals.length; i++){
			if (vals[i] > max)
				max = vals[i];
		}
		return max;
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

	public static Matrix[] toPortTimeMatrices(Map<String, Map<Integer, Integer>> timeCounters, short frameSize, 
			Map<Integer, Integer> portIndices, Set<Long> timeIndeces) throws ParseException {
				

		// gets start and end time, considering the time frame size and an aggregation by minute
		Set<String> timesSet = timeCounters.keySet();
		String[] timesStr = timesSet.toArray(new String[timesSet.size()]);
		Arrays.sort(timesStr);		
		Date startTime = Parser.sdf.parse(timesStr[0]);
		Date endTime = Parser.sdf.parse(timesStr[timesStr.length - 1]);
		
		int numFrames = 0;
		int numMinutes = 0;
		long diffTime = endTime.getTime() - startTime.getTime();
		int diffMinutes = (int) (long) (diffTime/(1000 * 60));
		float rest = diffMinutes%frameSize;
		System.out.println("Rest="+rest);
		if(rest > 0){
			numFrames = (diffMinutes/frameSize) + 1;
			numMinutes = numFrames * frameSize;
			endTime.setTime(startTime.getTime() + TimeUnit.MINUTES.toMillis(numMinutes));
		}
		System.out.println("Start="+Parser.sdf.format(startTime));
		System.out.println("End="+Parser.sdf.format(endTime));
		
		
		// creates timeIndeces according to possible times between start and end time
		timeIndeces.clear();
		long currTime = startTime.getTime();
		for(;currTime <= endTime.getTime();){
			timeIndeces.add(currTime);
			currTime = currTime + (1000 * 60);//one more minute
		}
		
		// fill the entire matrix with collected values. for each column (sorted times), fill each row(sorted ports) 
		double[][] arrayMatrix = new double[portIndices.size()][numMinutes];
		Long[] arrayTimeIndeces = timeIndeces.toArray(new Long[timeIndeces.size()]);
		Arrays.sort(arrayTimeIndeces);
		int timeIndex = 0;
		for(final long time: arrayTimeIndeces){
			String timeStr = Parser.sdf.format(time);
			Map<Integer,Integer> portCounters = timeCounters.get(timeStr);
			if(portCounters != null){
				Set<Integer> ports = portCounters.keySet();
				if(ports != null){
					for(final Integer port: ports){				
						arrayMatrix[portIndices.get(port)][timeIndex] = portCounters.get(port);				
					}	
				}
			}
			System.out.println(timeStr + ":" + timeIndex);
			timeIndex++;
		}

		// split the entire matrix into time frames
		Matrix matrix = new Matrix(arrayMatrix);
		Matrix[] matrices = new Matrix[numFrames];
		int startColumn = 0;
		int endColumn = frameSize - 1;
		for(int i = 0; endColumn < numMinutes; i++){				
			matrices[i] = matrix.getMatrix(0, portIndices.size() - 1, startColumn, endColumn);
			startColumn = startColumn + frameSize;
			endColumn = endColumn + frameSize;
		}
		
		return matrices;
	}
		
}