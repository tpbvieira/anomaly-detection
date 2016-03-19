package anomaly_detection;

import java.io.IOException;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.ojalgo.matrix.BasicMatrix.Factory;
import org.ojalgo.matrix.PrimitiveMatrix;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;
import matrix.MatrixUtil;

/**
 * @author thiago
 *
 */
public class EigenAnalysis{

	public static EigenvalueDecomposition eigencorrelation(Matrix matrix){

		RealMatrix realMatrix = MatrixUtils.createRealMatrix(matrix.getArray());
		int rowSize = matrix.getRowDimension();
		int columnSize = matrix.getColumnDimension();

		double[][] eigCor = new double[rowSize][columnSize];

		StandardDeviation std = new StandardDeviation(false);// false for population standard deviation
		Mean mean = new Mean();

		for(int line = 0; line < rowSize; line++){
			double[] row = realMatrix.getRow(line);			
			double rowStd = std.evaluate(row);			
			double rowMean;

			if(rowStd > 0){
				rowMean = mean.evaluate(row);
				for(int column = 0; column < columnSize; column++){
					eigCor[line][column] = (row[column] - rowMean)/rowStd;
				}
			}
		}

		// EstimatedCorrelationMatrix
		Matrix eigCorMatrix = new Matrix(eigCor);
		eigCorMatrix = eigCorMatrix.times(1D/Double.valueOf(columnSize));
		final Factory<PrimitiveMatrix> tmpFactory = PrimitiveMatrix.FACTORY;
		PrimitiveMatrix pMatrix = tmpFactory.rows(eigCor).conjugate();
		Matrix conjTranspose = MatrixUtil.primitiveMatrixToJamaMatrix(pMatrix);
		eigCorMatrix = eigCorMatrix.times(conjTranspose);		

		return eigCorMatrix.eig();
	}

	public static EigenvalueDecomposition eigencovariance(Matrix matrix){

		RealMatrix realMatrix = MatrixUtils.createRealMatrix(matrix.getArray());
		int rowSize = matrix.getRowDimension();
		int columnSize = matrix.getColumnDimension();

		double[][] eigCor = new double[rowSize][columnSize];

		Mean mean = new Mean();

		for(int line = 0; line < rowSize; line++){
			double[] row = realMatrix.getRow(line);			
			double rowMean = mean.evaluate(row);
			for(int column = 0; column < columnSize; column++){
				eigCor[line][column] = row[column] - rowMean;
			}			
		}

		// EstimatedCorrelationMatrix
		Matrix eigCorMatrix = new Matrix(eigCor);
		eigCorMatrix = eigCorMatrix.times(1D/Double.valueOf(columnSize));
		final Factory<PrimitiveMatrix> tmpFactory = PrimitiveMatrix.FACTORY;
		PrimitiveMatrix pMatrix = tmpFactory.rows(eigCor).conjugate();
		Matrix conjTranspose = MatrixUtil.primitiveMatrixToJamaMatrix(pMatrix);
		eigCorMatrix = eigCorMatrix.times(conjTranspose);		

		return eigCorMatrix.eig();
	}

	public static void main( String[] args ){    	
		try {
			System.out.println("###EigenAnalysisTest###");
			Matrix matrix = MatrixUtil.readMatrixFromFile("//media//thiago//shared//backup//doutorado//data//all//traffic//1.txt");
			EigenvalueDecomposition eig = eigencorrelation(matrix);
			Matrix eigVal = eig.getD();
			Matrix eigVec = eig.getV();
			System.out.println("eigencorrelation.LargestEigenvalue=" + MatrixUtil.max(eigVal));		
			MatrixUtil.printMatrix(eigVal, "Eigenvalues: ", 4);
			MatrixUtil.printMatrix(eigVec, "Eigenvectors: ", 4);

			eig = eigencovariance(matrix);
			eigVal = eig.getD();
			eigVec = eig.getV();
			System.out.println("eigencovariance.LargestEigenvalue=" + MatrixUtil.max(eigVal));		
			MatrixUtil.printMatrix(eigVal, "Eigenvalues: ", 4);
			MatrixUtil.printMatrix(eigVec, "Eigenvectors: ", 4);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}