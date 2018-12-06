package br.unb.lasp.anomaly;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;
import br.unb.lasp.matrix.MatrixUtil;
import org.junit.Test;

import java.io.IOException;

/**
 * @author thiago
 *
 */
public class EigenAnalysisTest{

	@Test
	public void testEigencorrelation(){
		try {
			System.out.println("### testEigencorrelation");
			Matrix matrix = MatrixUtil.readMatrixFromFile("//media//thiago//shared//backup//doutorado//data//all//traffic//1.txt");
			EigenvalueDecomposition eig = EigenAnalysis.eigencorrelation(matrix);
			Matrix eigVal = eig.getD();
			Matrix eigVec = eig.getV();
			System.out.println("eigencorrelation.LargestEigenvalue=" + MatrixUtil.max(eigVal));		
			MatrixUtil.printMatrix(eigVal, "Eigenvalues: ");
			MatrixUtil.printMatrix(eigVec, "Eigenvectors: ");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Test
	public void testEigencovariance(){
		try {
			System.out.println("### testEigencovariance");
			Matrix matrix = MatrixUtil.readMatrixFromFile("//media//thiago//shared//backup//doutorado//data//all//traffic//1.txt");
			EigenvalueDecomposition eig = EigenAnalysis.eigencovariance(matrix);
			Matrix eigVal = eig.getD();
			Matrix eigVec = eig.getV();
			System.out.println("eigencovariance.LargestEigenvalue=" + MatrixUtil.max(eigVal));		
			MatrixUtil.printMatrix(eigVal, "Eigenvalues: ");
			MatrixUtil.printMatrix(eigVec, "Eigenvectors: ");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}