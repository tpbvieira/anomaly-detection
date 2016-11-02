package br.unb.lasp.anomaly.mos.cases;

import java.text.ParseException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import Jama.Matrix;
import br.unb.lasp.anomaly.mos.MOS;
import br.unb.lasp.matrix.MatrixUtil;
import br.unb.lasp.parser.Parser;

public class Darpa {

	public static void main(String[] args) {
		
		// Init
		String targetIp = "172.16.112.50";//Pascal
//		String targetIp = "172.16.113.50";//Zeno
//		String targetIp = "172.16.114.50";//Marx
		Set<Integer> targetPorts = new HashSet<>(Arrays.asList(20,21,22,23,25,79,80,88,107,109,110,113,115,143,161,389,443));
//		final String filePath = "/media/thiago/ubuntu/datasets/darpa/1998/03_Training Data/week01/04_wednesday/outside.tcpdump";	//22
//		final String filePath = "/media/thiago/ubuntu/datasets/darpa/1998/03_Training Data/week03/05_thursday/outside.tcpdump";		//16
//		final String filePath = "/media/thiago/ubuntu/datasets/darpa/1998/03_Training Data/week03/04_wednesday/outside.tcpdump";		//
//		final String filePath = "/media/thiago/ubuntu/datasets/darpa/1998/03_Training Data/week04/03_tuesday/outside.tcpdump";		//22
//		final String filePath = "/media/thiago/ubuntu/datasets/darpa/1998/03_Training Data/week04/04_wednesday/outside.tcpdump";		//
//		final String filePath = "/media/thiago/ubuntu/datasets/darpa/1998/03_Training Data/week04/03_tuesday/outside.tcpdump";		//
//		final String filePath = "/media/thiago/ubuntu/datasets/darpa/1998/03_Training Data/week05/05_thursday/outside.tcpdump";		//
//		final String filePath = "/media/thiago/ubuntu/datasets/darpa/1998/03_Training Data/week05/05_thursday/outside.tcpdump";		//5
//		final String filePath = "/media/thiago/ubuntu/datasets/darpa/1998/03_Training Data/week05/06_friday/outside.tcpdump";		//16
//		final String filePath = "/media/thiago/ubuntu/datasets/darpa/1998/03_Training Data/week06/02_monday/outside.tcpdump";		//16
//		final String filePath = "/media/thiago/ubuntu/datasets/darpa/1998/03_Training Data/week06/03_tuesday/outside.tcpdump";		//
//		final String filePath = "/media/thiago/ubuntu/datasets/darpa/1998/03_Training Data/week06/04_wednesday/outside.tcpdump";	//8
//		final String filePath = "/media/thiago/ubuntu/datasets/darpa/1998/03_Training Data/week06/05_thursday/outside.tcpdump";		//11
//		final String filePath = "/media/thiago/ubuntu/datasets/darpa/1998/03_Training Data/week06/06_friday/outside.tcpdump";		//
//		final String filePath = "/media/thiago/ubuntu/datasets/darpa/1998/03_Training Data/week07/06_friday/outside.tcpdump";		//
//		final String filePath = "/media/thiago/ubuntu/datasets/darpa/1998/03_Training Data/week07/03_tuesday/outside.tcpdump";		//
//		final String filePath = "/media/thiago/ubuntu/datasets/darpa/1998/03_Training Data/week07/03_tuesday/outside.tcpdump";		//
		
		final List<String> filesPath = Arrays.asList(
				"/media/thiago/ubuntu/datasets/darpa/1998/03_Training Data/week04/06_friday/outside.tcpdump",
				"/media/thiago/ubuntu/datasets/darpa/1998/03_Training Data/week03/06_friday/outside.tcpdump",
				"/media/thiago/ubuntu/datasets/darpa/1998/03_Training Data/week05/03_tuesday/outside.tcpdump",
				"/media/thiago/ubuntu/datasets/darpa/1998/03_Training Data/week04/03_tuesday/outside.tcpdump",
				"/media/thiago/ubuntu/datasets/darpa/1998/03_Training Data/week05/05_thursday/outside.tcpdump");
		final List<Short> frameSizes = Arrays.asList((short)10,(short)20,(short)60,(short)120);
		final int numIterations = 2;
		
		for(String filePath: filesPath){
			System.out.println("FilePath:"+filePath);
			Map<String, Map<Integer, Integer>> timeCounters = new HashMap<String, Map<Integer, Integer>>();
			
			Set<Long> timeIndeces = new HashSet<Long>();
			Matrix[] matrices = null;
			
			// Parsing		
			Parser.pcapToPortTimeCountMapDst(filePath, targetIp, targetPorts, timeCounters);		
			
			// Data Modeling
			for(short frameSize: frameSizes){
				System.out.println("FrameSize:"+frameSize);
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
				} catch (NumberFormatException e) {
					e.printStackTrace();
				} catch (ParseException e) {
					e.printStackTrace();
				} catch (Exception e) {
					e.printStackTrace();
				}

	
				for(int j = 0; j< numIterations; j++){
					// EigenAnalysis
					long iniTime = System.currentTimeMillis();		
					double[][] largestEigValCov = MatrixUtil.getLargestEigValCov(matrices);
					System.out.println("EigValCov:"+ (System.currentTimeMillis()-iniTime));
					iniTime = System.currentTimeMillis();		
					double[][] largestEigValCor = MatrixUtil.getLargestEigValCor(matrices);
					System.out.println("EigValCor:"+ (System.currentTimeMillis()-iniTime));
				
					// MOS Analysis
					iniTime = System.currentTimeMillis();
					int mosCov = MOS.edcJAMA(new Matrix(largestEigValCov), frameSize);
					System.out.println("EDCCov:"+ (System.currentTimeMillis()-iniTime));
					iniTime = System.currentTimeMillis();
					int mosCor = MOS.edcJAMA(new Matrix(largestEigValCor), frameSize);
					System.out.println("EDCCor:"+ (System.currentTimeMillis()-iniTime));		
				}
							
				System.out.println("MOSCor:\t" + mosCor);
				System.out.println(Arrays.deepToString(largestEigValCor));
				System.out.println("MOSCov:\t" + mosCov);
				System.out.println(Arrays.deepToString(largestEigValCov));				
			}
			
		}
		

	}

}