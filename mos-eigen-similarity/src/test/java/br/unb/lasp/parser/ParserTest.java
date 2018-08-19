package br.unb.lasp.parser;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.junit.Test;

import java.util.*;

public class ParserTest {

	private static final Log logger = LogFactory.getLog(ParserTest.class);
	
	@Test
	public void testPcapToPortTimeCountMap() {
		
		String targetIp = "172.16.112.50";
		Set<Integer> targetPorts = new HashSet<>(Arrays.asList(20,21,22,23,25,53,79,80,81,113,123,984,1024,1025,1026,1027,1028));		
		final String filePath = "/media/thiago/ubuntu/datasets/network/darpa-kdd-network-attacks/1998/03_Training_Data/week01/04_wednesday/outside.tcpdump";
		Map<String, Map<Integer, Integer>> timeCounters = new HashMap<String, Map<Integer, Integer>>();
		long t0,t1;
		t0 = System.currentTimeMillis();
		Parser.numPackets = 0;
		Parser.countedPackets = 0;
		Parser.pcapToPortTimeCountMap(filePath, targetIp, targetPorts, timeCounters);
		t1 = System.currentTimeMillis();
		
		System.out.println("ProcessingTime=" + ((t1-t0)/1000f) + "s");
		System.out.println("CollectedTime=" + timeCounters.keySet().size());
		System.out.println("NumPackets=" + Parser.numPackets);
		System.out.println("CountedPackets=" + Parser.countedPackets);
		System.out.println("CountedTimes=" + timeCounters.size());
		
	}

	@Test
	public void testPrintPcapStatistics(){		
		Parser.printPcapStatistics("/media/thiago/ubuntu/datasets/network/darpa-kdd-network-attacks/1998/03_Training_Data/week01/04_wednesday/outside.tcpdump", "172.16.112.50", true);	
	}

}