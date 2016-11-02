package br.unb.lasp.matrix;
import Jama.Matrix;
import Jama.SingularValueDecomposition; 

public class JAMATest { 

	static public void printMatrix(Matrix m){
		double[][] d = m.getArray();

		for(int row = 0; row < d.length; row++){
			for(int col = 0; col < d[row].length; col++){
				System.out.printf("%6.4f\t", m.get(row, col));
			}
			System.out.println();
		}
		System.out.println();
	}

	public static void main(String[] args) { 
		double[][] vals = { 
				{1., 0., 0., 0., 2.}, 
				{0., 0., 3., 0., 0.}, 
				{0., 0., 0., 0., 0.}, 
				{0., 4., 0., 0., 0.} 
		};
		
		Matrix A = new Matrix(vals);         
		SingularValueDecomposition svd = new SingularValueDecomposition(A); 

		System.out.println("A = ");
		printMatrix(A);

		System.out.println("U = ");
		printMatrix(svd.getU());

		System.out.println("Sigma = ");
		printMatrix(svd.getS());

		System.out.println("V = ");
		printMatrix(svd.getV());
	} 
}