package br.unb.lasp.anomaly.examples;

import Jama.Matrix;
import br.unb.lasp.anomaly.MOS;
import br.unb.lasp.matrix.MatrixUtil;
import br.unb.lasp.parser.Parser;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import java.text.ParseException;
import java.util.*;

public class Darpa {

	public static void main(String[] args) {

		// Init
		String targetIp = "172.16.112.50";//Pascal
		//		String targetIp = "172.16.113.50";//Zeno
		//		String targetIp = "172.16.114.50";//Marx
		Set<Integer> targetPorts17 = new HashSet<>(Arrays.asList(20,21,22,23,25,79,80,88,107,109,110,113,115,143,161,389,443));
		Set<Integer> targetPorts34 = new HashSet<>(Arrays.asList(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,20,21,22,23,25,79,80,88,107,109,110,113,115,143,161,389,443));
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
		final int numIterations = 1;

		long start = System.currentTimeMillis();

		for(String filePath: filesPath){
			Map<String, Map<Integer, Integer>> timeCounters = new HashMap<String, Map<Integer, Integer>>();

			Set<Long> timeIndeces = new HashSet<Long>();
			Matrix[] matrices = null;

			// Parsing		
			Parser.pcapToPortTimeCountMapDst(filePath, targetIp, targetPorts17, timeCounters);		

			// Data Modeling
			for(short frameSize: frameSizes){
				System.out.print("FilePath:"+filePath);
				System.out.print("\t FrameSize:"+frameSize);
				System.out.print("\t PortSize:"+targetPorts17.size());
				Integer[] portIndices = targetPorts17.toArray(new Integer[targetPorts17.size()]);
				Arrays.sort(portIndices);
				int i = 0;
				Map<Integer,Integer> portIndicesMap = new HashMap<Integer,Integer>();
				for(Integer port: portIndices){
					portIndicesMap.put(port, i);
					i++;
				}
				try {
					matrices = MatrixUtil.toPortTimeMatrices(timeCounters, frameSize, portIndicesMap, timeIndeces);
				} catch (Exception e) {
					e.printStackTrace();
				}

				DescriptiveStatistics eigValCov = new DescriptiveStatistics();
				DescriptiveStatistics eigValCor = new DescriptiveStatistics();
				DescriptiveStatistics eDCCov = new DescriptiveStatistics();
				DescriptiveStatistics eDCCor = new DescriptiveStatistics();

				for(int j = 0; j< numIterations; j++){
					// EigenAnalysis
					long iniTime = System.currentTimeMillis();		
					double[][] largestEigValCov = MatrixUtil.getLargestEigValCov(matrices);
					eigValCov.addValue(System.currentTimeMillis()-iniTime);
					iniTime = System.currentTimeMillis();		
					double[][] largestEigValCor = MatrixUtil.getLargestEigValCor(matrices);
					eigValCor.addValue(System.currentTimeMillis()-iniTime);

					// MOS Analysis
					iniTime = System.currentTimeMillis();
					int mosCov = MOS.edcJAMA(new Matrix(largestEigValCov), frameSize);
					eDCCov.addValue(System.currentTimeMillis()-iniTime);

					iniTime = System.currentTimeMillis();
					int mosCor = MOS.edcJAMA(new Matrix(largestEigValCor), frameSize);
					eDCCor.addValue(System.currentTimeMillis()-iniTime);
				}
				System.out.print("\t EigValCov:"+ eigValCov.getMean());
				System.out.print("\t EigValCor:"+ eigValCor.getMean());
				System.out.print("\t EDCCov:"+ eDCCov.getMean());
				System.out.println("\t EDCCor:"+ eDCCor.getMean());
			}


			// Parsing		
			Parser.pcapToPortTimeCountMapDst(filePath, targetIp, targetPorts34, timeCounters);		

			// Data Modeling
			for(short frameSize: frameSizes){
				if(frameSize > 34){
					System.out.print("FilePath:"+filePath);
					System.out.print("\t FrameSize:"+frameSize);
					System.out.print("\t PortSize:"+targetPorts34.size());
					Integer[] portIndices = targetPorts34.toArray(new Integer[targetPorts34.size()]);
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

					DescriptiveStatistics eigValCov = new DescriptiveStatistics();
					DescriptiveStatistics eigValCor = new DescriptiveStatistics();
					DescriptiveStatistics eDCCov = new DescriptiveStatistics();
					DescriptiveStatistics eDCCor = new DescriptiveStatistics();

					for(int j = 0; j< numIterations; j++){
						// EigenAnalysis
						long iniTime = System.currentTimeMillis();		
						double[][] largestEigValCov = MatrixUtil.getLargestEigValCov(matrices);
						eigValCov.addValue(System.currentTimeMillis()-iniTime);
						iniTime = System.currentTimeMillis();		
						double[][] largestEigValCor = MatrixUtil.getLargestEigValCor(matrices);
						eigValCor.addValue(System.currentTimeMillis()-iniTime);

						// MOS Analysis
						iniTime = System.currentTimeMillis();
						int mosCov = MOS.edcJAMA(new Matrix(largestEigValCov), frameSize);
						eDCCov.addValue(System.currentTimeMillis()-iniTime);

						iniTime = System.currentTimeMillis();
						int mosCor = MOS.edcJAMA(new Matrix(largestEigValCor), frameSize);
						eDCCor.addValue(System.currentTimeMillis()-iniTime);
					}
					System.out.print("\t EigValCov:"+ eigValCov.getMean());
					System.out.print("\t EigValCor:"+ eigValCor.getMean());
					System.out.print("\t EDCCov:"+ eDCCov.getMean());
					System.out.println("\t EDCCor:"+ eDCCor.getMean());
				}
			}

		}

		System.out.println("Total:" + (System.currentTimeMillis() - start));		
	}

}