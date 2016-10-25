package br.unb.lasp.parser;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;

import org.apache.commons.lang.time.DateUtils;
import org.jnetpcap.Pcap;
import org.jnetpcap.packet.JPacket;
import org.jnetpcap.packet.JPacketHandler;
import org.jnetpcap.packet.format.FormatUtils;
import org.jnetpcap.protocol.network.Ip4;
import org.jnetpcap.protocol.tcpip.Tcp;
import org.jnetpcap.protocol.tcpip.Udp;

import com.univocity.parsers.csv.CsvParser;
import com.univocity.parsers.csv.CsvParserSettings;

public class Parser {

	public static SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd HH:mm");
	
	public static int numPackets = 0;
	public static int countedPackets = 0;
	
	public static void main(String[] args) { 
		printPcapStatistics("/media/thiago/ubuntu/datasets/darpa/1998/03_Training Data/week01/04_wednesday/outside.tcpdump");	
	}

	public static void pcapToPortTimeCountMap(final String filePath, final String targetIp, final Set<Integer> targetPorts, 
			final Map<String, Map<Integer, Integer>> timeCounters) {

		final StringBuilder errbuf = new StringBuilder();
		final Pcap pcap = Pcap.openOffline(filePath, errbuf);

		pcap.loop(Pcap.LOOP_INFINITE, new JPacketHandler<StringBuilder>() {					

			Ip4 ip = new Ip4();
			Tcp tcp = new Tcp();
			Udp udp = new Udp();

			public void nextPacket(JPacket packet, StringBuilder errbuf) {
				numPackets++;
				if(packet.hasHeader(Ip4.ID)){

					// Source
					String srcIp = FormatUtils.ip(packet.getHeader(ip).source());					
					String dstIp = FormatUtils.ip(packet.getHeader(ip).destination());

					if(srcIp.equals(targetIp) || dstIp.equals(targetIp)){

						Integer port = null;

						if(packet.hasHeader(Tcp.ID)){
							// Is a target port?
							Integer tmpPort = packet.getHeader(tcp).source();
							if(targetPorts.contains(tmpPort)){
								port = packet.getHeader(tcp).source();
							}else {
								tmpPort = packet.getHeader(tcp).destination();
								if(targetPorts.contains(tmpPort)){
									port = packet.getHeader(tcp).destination();
								}								
							}
						}else if(packet.hasHeader(Udp.ID)){
							// Is a target port?
							Integer tmpPort = packet.getHeader(udp).source();
							if(targetPorts.contains(tmpPort)){
								port = packet.getHeader(udp).source();
							}else {
								tmpPort = packet.getHeader(udp).destination();
								if(targetPorts.contains(tmpPort)){
									port = packet.getHeader(udp).destination();
								}								
							}
						}

						if(port != null){							
							String time = sdf.format(new Date(packet.getCaptureHeader().timestampInMillis()));
							
							Map<Integer, Integer> portCounters;							
							if(timeCounters.containsKey(time)){
								portCounters = timeCounters.get(time);	
							}else{
								portCounters = new HashMap<>();
							}
							
							Integer count = 1;
							if(portCounters.containsKey(port)){
								count = portCounters.get(port) + 1;	
							}
							
							portCounters.put(port, count);
							timeCounters.put(time, portCounters);
							
							countedPackets++;
						}

					}

				}

			}

		}, errbuf);
		
		pcap.close();
	}

	public static void printPcapStatistics(final String filePath) {

		final StringBuilder errbuf = new StringBuilder();
		final Pcap pcap = Pcap.openOffline(filePath, errbuf);

		final Map<Integer,Integer> tcpPortCount = new HashMap<>();
		final Map<Integer,Integer> udpPortCount = new HashMap<>();
		final Map<String,Integer> ipCount = new HashMap<>();

		pcap.loop(Pcap.LOOP_INFINITE, new JPacketHandler<StringBuilder>() {					

			Ip4 ip = new Ip4();
			Tcp tcp = new Tcp();
			Udp udp = new Udp();

			public void nextPacket(JPacket packet, StringBuilder errbuf) {

				if(packet.hasHeader(Ip4.ID)){

					// Source
					String srcIp = FormatUtils.ip(packet.getHeader(ip).source());
					Integer count = 1;
					if(ipCount.containsKey(srcIp)){
						count = ipCount.get(srcIp) + 1;
					}
					ipCount.put(srcIp, count);

					//Destination
					String dstIp = FormatUtils.ip(packet.getHeader(ip).destination());
					count = 1;
					if(ipCount.containsKey(dstIp)){
						count = ipCount.get(dstIp) + 1;
					}
					ipCount.put(dstIp, count);

					if(packet.hasHeader(Tcp.ID)){
						Integer srcPort = packet.getHeader(tcp).source();
						count = 1;
						if(tcpPortCount.containsKey(srcPort)){
							count = tcpPortCount.get(srcPort) + 1;
						}
						tcpPortCount.put(srcPort, count);

						Integer dstPort = packet.getHeader(tcp).destination();
						count = 1;
						if(tcpPortCount.containsKey(dstPort)){
							count = tcpPortCount.get(dstPort) + 1;
						}
						tcpPortCount.put(dstPort, count);					
					}else 
						if(packet.hasHeader(Udp.ID)){
							Integer srcPort = packet.getHeader(udp).source();
							count = 1;
							if(udpPortCount.containsKey(srcPort)){
								count = udpPortCount.get(srcPort) + 1;
							}
							udpPortCount.put(srcPort, count);

							Integer dstPort = packet.getHeader(udp).destination();
							count = 1;
							if(udpPortCount.containsKey(dstPort)){
								count = udpPortCount.get(dstPort) + 1;
							}
							udpPortCount.put(dstPort, count);					
						}					

				}

			}

		}, errbuf);

		pcap.close();

		for (Map.Entry<String,Integer> entry : ipCount.entrySet()) {
			String key = entry.getKey();
			Integer value = entry.getValue();
			System.out.println("IP: " + key + "=" + value);
		}

		for (Map.Entry<Integer,Integer> entry : tcpPortCount.entrySet()) {
			Integer key = entry.getKey();
			Integer value = entry.getValue();
			System.out.println("TCP: " + key + "=" + value);
		}

		for (Map.Entry<Integer,Integer> entry : udpPortCount.entrySet()) {
			Integer key = entry.getKey();
			Integer value = entry.getValue();
			System.out.println("UDP: " + key + "=" + value);
		}

	}

	public static HashMap<Long,HashMap<String,Integer>> csvToPortTimeCountMap(String path, SimpleDateFormat sdf, HashMap<String,Integer> matrixRowIndeces){
		CsvParserSettings settings = new CsvParserSettings();
		settings.getFormat().setDelimiter(';');
		settings.getFormat().setLineSeparator("\n");
		CsvParser parser = new CsvParser(settings);

		HashMap<Long,HashMap<String,Integer>> values = new HashMap<Long,HashMap<String,Integer>>();

		try {
			FileReader csvFile = new FileReader(new File(path));
			List<String[]> fileRows = parser.parseAll(csvFile);
			csvFile.close();

			int matrixRowIndex = 0;			
			for(final String[] row: fileRows){
				Date dateTime = sdf.parse(row[0]);				
				StringTokenizer strt = new StringTokenizer(row[5], "'");
				String eventStr =  strt.nextToken();
				Date truncDateTime = DateUtils.truncate(dateTime, Calendar.MINUTE);
				long time = truncDateTime.getTime();
				if(values.containsKey(time)){
					HashMap<String,Integer> events = values.get(time);
					if(events.containsKey(eventStr)){
						int eventCount = events.get(eventStr);
						eventCount++;
						events.put(eventStr, eventCount);
					}else{
						events.put(eventStr, 1);
						if(!matrixRowIndeces.containsKey(eventStr)){
							matrixRowIndeces.put(eventStr, matrixRowIndex);
							matrixRowIndex++;
						}
					}
					values.put(time, events);
				}else{
					HashMap<String,Integer> events = new HashMap<String,Integer>();
					events.put(eventStr, 1);
					matrixRowIndeces.put(eventStr, matrixRowIndex);
					values.put(time, events);
				}
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (ParseException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}

		return values;		
	}

}