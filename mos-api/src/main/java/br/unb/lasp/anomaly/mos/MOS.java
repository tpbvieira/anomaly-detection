package br.unb.lasp.anomaly.mos;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import Jama.Matrix;
import br.unb.lasp.matrix.MatrixUtil;

/**
 * @author thiago
 *
 */
public class MOS{
	
	public static int edcJAMA(Matrix matrix, short N){
//		System.out.println();
		
		int maxSize = Math.max(matrix.getRowDimension(), matrix.getColumnDimension());
		
//		MatrixUtil.printMatrix(matrix, "Matrix: ");
		
//		System.out.println("PeriodSize: " + N);
//		System.out.println("RowSize: " + matrix.getRowDimension());
//		System.out.println("ColumnSize: " + matrix.getColumnDimension());
//		System.out.println("MaxSize: " + maxSize);
		
		matrix = matrix.transpose();
//		MatrixUtil.printMatrix(matrix, "Matrix_transposed: ");
		

		// sorts into ascending order, by columns and individually
		double[][] matrixValues = matrix.getArray();		
		RealMatrix realMatrix = MatrixUtils.createRealMatrix(matrixValues);		
		for (int j = 0; j < matrixValues[0].length; j++) {
			double[] column = realMatrix.getColumn(j);
			Arrays.sort(column);
			realMatrix.setColumn(j,column);
        }
		matrix = new Matrix(realMatrix.getData());
//		MatrixUtil.printMatrix(matrix, "Matrix_serted: ");
		
		// ??
		double c_n = Math.sqrt(N * Math.log(Math.log(N)));
//		System.out.println("c_N: " + c_n + ". Success: " + (Double.compare(c_n, 4.684418214388952D) == 0));
		
		// ??
		double[] edcValues = new double[maxSize];
		for(int i = 0; i < maxSize; i++){
			int windowSize = maxSize - i;
//			System.out.println("WindowSize: " + windowSize);
			double[][] window = Arrays.copyOfRange(matrix.getArray(), 0, windowSize);			
//			MatrixUtil.printMatrix(new Matrix(window), "Matrix_window: ");
			Matrix windowGeomean = MatrixUtil.geomean(new Matrix(window));			
//			MatrixUtil.printMatrix(windowGeomean, "Window_geomean: ");			
			Matrix windowMean = MatrixUtil.mean(new Matrix(window));			
//			MatrixUtil.printMatrix(windowMean, "Window_mean: ");
			
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
    
}