package matrix;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.descriptive.moment.GeometricMean;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.ojalgo.array.Array2D;
import org.ojalgo.matrix.PrimitiveMatrix;

import Jama.Matrix;

/**
 * @author thiago
 *
 */
public class MatrixUtil{
	
	/**
	 * Prints into matrix format
	 * 
	 * @param matrix
	 * @param msg
	 * @param decimals
	 */	
	public static void printMatrix(Matrix matrix, String msg, int decimals){
    	System.out.println(msg);
		matrix.print(matrix.getColumnDimension(), 4);    	
    }
    
    
    /**
     * Calculates the geometric mean of a sample. For vectors, it is the geometric mean of the elements into the vector. 
     * For matrices, it is a row vector containing the geometric means of each column.
     * 
     * @param matrix
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
     * @param matrix
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
     * Read file, with matrix separated by spaces, into a matrix Matrix
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
    
}