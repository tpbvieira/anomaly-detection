package anomaly_detection;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import Jama.Matrix;
import matrix.MatrixUtil;

/**
 * @author thiago
 *
 */
public class MOS{
	
	public static int edcJAMA(Matrix matrix, short N){
//		System.out.println();
		
		int maxSize = Math.max(matrix.getRowDimension(), matrix.getColumnDimension());
		
//		MatrixUtil.printMatrix(matrix, "Matrix: ", 4);
		
//		System.out.println("PeriodSize: " + N);
//		System.out.println("RowSize: " + matrix.getRowDimension());
//		System.out.println("ColumnSize: " + matrix.getColumnDimension());
//		System.out.println("MaxSize: " + maxSize);
		
		matrix = matrix.transpose();
//		MatrixUtil.printMatrix(matrix, "Matrix_transposed: ", 4);
		

		// sorts into ascending order, by columns and individually
		double[][] matrixValues = matrix.getArray();		
		RealMatrix realMatrix = MatrixUtils.createRealMatrix(matrixValues);		
		for (int j = 0; j < matrixValues[0].length; j++) {
			double[] column = realMatrix.getColumn(j);
			Arrays.sort(column);
			realMatrix.setColumn(j,column);
        }
		matrix = new Matrix(realMatrix.getData());
//		MatrixUtil.printMatrix(matrix, "Matrix_serted: ", 4);
		
		// ??
		double c_n = Math.sqrt(N * Math.log(Math.log(N)));
//		System.out.println("c_N: " + c_n + ". Success: " + (Double.compare(c_n, 4.684418214388952D) == 0));
		
		// ??
		double[] edcValues = new double[maxSize];
		for(int i = 0; i < maxSize; i++){
			int windowSize = maxSize - i;
//			System.out.println("WindowSize: " + windowSize);
			double[][] window = Arrays.copyOfRange(matrix.getArray(), 0, windowSize);			
//			MatrixUtil.printMatrix(new Matrix(window), "Matrix_window: ", 4);
			Matrix windowGeomean = MatrixUtil.geomean(new Matrix(window));			
//			MatrixUtil.printMatrix(windowGeomean, "Window_geomean: ", 4);			
			Matrix windowMean = MatrixUtil.mean(new Matrix(window));			
//			MatrixUtil.printMatrix(windowMean, "Window_mean: ", 4);
			
			// mm must vary from 0 to M - 1. ii starts with 1 and ends in M, wich means M-1 until 0
			int mm = maxSize - windowSize;
//			System.out.println("mm: " + mm);
			edcValues[i] = -2*N * ( maxSize - mm ) * Math.log(windowGeomean.getArray()[0][0]/windowMean.getArray()[0][0]) + mm * (2*maxSize - mm) * c_n;
//			System.out.println("edc[" + i + "]: " + edcValues[i]);
		}
		
		Double[] edcValuesArray = ArrayUtils.toObject(edcValues);
		List<Double> edcValuesList = Arrays.asList(edcValuesArray);
		Double minValue = Collections.min(edcValuesList);
		int minIndex = edcValuesList.indexOf(minValue);
		
//		System.out.println("EdcValues: " + Arrays.toString(edcValues));
//		System.out.println("EdcMinValue: " + minValue);
//		System.out.println("EdcMinIndex: " + minIndex);
		
        return minIndex;
	}
	
    public static void main( String[] args ){
    	
    	long startTime = System.currentTimeMillis();
        double[][] covLargEig = {{1887545D, 2341327D, 3213867D, 133238294D, 92384021611D, 708335D}};
        Matrix covLargEigMat = new Matrix(covLargEig);
        edcJAMA(covLargEigMat,(short)20);
        long stopTime = System.currentTimeMillis();
        long elapsedTime = stopTime - startTime;
//        System.out.println("Time: " + elapsedTime);

        startTime = System.currentTimeMillis();
        double[][] corLargEig = {{2.0734D, 2.1451D, 10.0718D, 2.1620D, 2.4253D, 1.7948D}};
        Matrix corLargEigMat = new Matrix(corLargEig);
        edcJAMA(corLargEigMat,(short)20);
        stopTime = System.currentTimeMillis();
        elapsedTime = stopTime - startTime;
//        System.out.println("Time: " + elapsedTime);
        
    }
}