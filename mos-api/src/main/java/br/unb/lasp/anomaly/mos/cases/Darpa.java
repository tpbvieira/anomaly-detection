package br.unb.lasp.anomaly.mos.cases;

import java.text.ParseException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import Jama.Matrix;
import br.unb.lasp.anomaly.mos.MOS;
import br.unb.lasp.matrix.MatrixUtil;
import br.unb.lasp.parser.Parser;

public class Darpa {

	public static void main(String[] args) {
		
		// Init
		String targetIp = "192.168.1.30";
		Set<Integer> targetPorts = new HashSet<>(Arrays.asList(20,21,22,23,25,53,79,80,109,110,111,113,512,513,514,515,1021));		
		final String filePath = "/media/thiago/ubuntu/datasets/darpa/1998/01_sample_data/sample_data01.tcpdump";
		Map<String, Map<Integer, Integer>> timeCounters = new HashMap<String, Map<Integer, Integer>>();
		short frameSize =20;
		Set<Long> timeIndeces = new HashSet<Long>();
		Matrix[] matrices = null;
		
		// Parsing		
		Parser.pcapToPortTimeCountMap(filePath, targetIp, targetPorts, timeCounters);
		
		// Data Modeling
		Integer[] portIndices = targetPorts.toArray(new Integer[targetPorts.size()]);
		Arrays.sort(portIndices);
		int i = 0;
		Map<Integer,Integer> portIndicesMap = new HashMap<Integer,Integer>();
		for(Integer port: portIndices){
			portIndicesMap.put(port, i);
			i++;			
		}
		try {
			matrices = MatrixUtil.toPortTimeMatrices(timeCounters, frameSize, portIndicesMap, timeIndeces);			
			for(Matrix matrix: matrices){
				MatrixUtil.printMatrix(matrix, "Resultado=");
			}
		} catch (NumberFormatException e) {
			e.printStackTrace();
		} catch (ParseException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}

		// EigenAnalysis
		double[][] largestEigValCov = MatrixUtil.getLargestEigValCov(matrices);
		double[][] largestEigValCor = MatrixUtil.getLargestEigValCor(matrices);
	
		// MOS Analysis
		int mosCov = MOS.edcJAMA(new Matrix(largestEigValCov), frameSize);
		int mosCor = MOS.edcJAMA(new Matrix(largestEigValCor), frameSize);
		
		System.out.println("MOSCor: " + mosCor);
		System.out.println("MOSCov: " + mosCov);
		
	}

}