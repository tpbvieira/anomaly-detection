package anomaly_detection.mos;

import static org.junit.Assert.fail;

import org.junit.Test;

import Jama.Matrix;
import br.unb.lasp.anomaly.mos.MOS;

public class EdcJAMATest {

	@Test
	public void test() {
		
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