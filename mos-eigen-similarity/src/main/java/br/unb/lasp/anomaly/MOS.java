package br.unb.lasp.anomaly;

import Jama.Matrix;
import br.unb.lasp.matrix.MatrixUtil;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * @author thiago
 *
 */
public class MOS{
	
	public static int edcJAMA(Matrix matrix, short N){
		
		int maxSize = Math.max(matrix.getRowDimension(), matrix.getColumnDimension());
		matrix = matrix.transpose();

		// sorts into ascending order, by columns and individually
		double[][] matrixValues = matrix.getArray();		
		RealMatrix realMatrix = MatrixUtils.createRealMatrix(matrixValues);		
		for (int j = 0; j < matrixValues[0].length; j++) {
			double[] column = realMatrix.getColumn(j);
			Arrays.sort(column);
			realMatrix.setColumn(j,column);
        }
		matrix = new Matrix(realMatrix.getData());
		
		// ??
		double c_n = Math.sqrt(N * Math.log(Math.log(N)));
		
		// ??
		double[] edcValues = new double[maxSize];
		for(int i = 0; i < maxSize; i++){
			int windowSize = maxSize - i;
			double[][] window = Arrays.copyOfRange(matrix.getArray(), 0, windowSize);			
			Matrix windowGeomean = MatrixUtil.geomean(new Matrix(window));			
			Matrix windowMean = MatrixUtil.mean(new Matrix(window));			
			
			// mm must vary from 0 to M - 1. ii starts with 1 and ends in M, wich means M-1 until 0
			int mm = maxSize - windowSize;
			edcValues[i] = -2*N * ( maxSize - mm ) * Math.log(windowGeomean.getArray()[0][0]/windowMean.getArray()[0][0]) + mm * (2*maxSize - mm) * c_n;
		}
		
		Double[] edcValuesArray = ArrayUtils.toObject(edcValues);
		List<Double> edcValuesList = Arrays.asList(edcValuesArray);
		Double minValue = Collections.min(edcValuesList);

        return edcValuesList.indexOf(minValue);
	}
    
}