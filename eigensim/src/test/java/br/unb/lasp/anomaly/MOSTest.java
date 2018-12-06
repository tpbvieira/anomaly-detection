package br.unb.lasp.anomaly;

import Jama.Matrix;
import org.junit.Test;

import static org.junit.Assert.fail;

public class MOSTest {

	@Test
	public void testEdcJAMA() {
		
		double[][] covLargEig = {{1887545D, 2341327D, 3213867D, 133238294D, 92384021611D, 708335D}};
        Matrix covLargEigMat = new Matrix(covLargEig);
        int mosCov = MOS.edcJAMA(covLargEigMat,(short)20);
        if(mosCov != 2){
			fail("MosCovError: Expected 2 intead of " + mosCov);
		}
        
        double[][] corLargEig = {{2.0734D, 2.1451D, 10.0718D, 2.1620D, 2.4253D, 1.7948D}};
        Matrix corLargEigMat = new Matrix(corLargEig);
        int mosCor = MOS.edcJAMA(corLargEigMat,(short)20);
        if(mosCor != 1){
			fail("MosCorError: Expected 1 intead of " + mosCor);
		}
		
	}

}