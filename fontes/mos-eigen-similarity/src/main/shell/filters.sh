echo ">Removing previous data"
rm -rf data/all
rm -rf data/signal
rm -rf data/noise
rm -rf data/synflood
rm -rf data/fraggle
rm -rf data/portscan

echo ">Creating traffic directories"
mkdir data/all
mkdir data/signal
mkdir data/noise
mkdir data/synflood
mkdir data/fraggle
mkdir data/portscan
mkdir data/all/traffic
mkdir data/signal/traffic
mkdir data/noise/traffic
mkdir data/synflood/traffic
mkdir data/fraggle/traffic
mkdir data/portscan/traffic

echo ">ALL = excludes ARP, ICMP and 224.0.0.251"
cat data/traffic | grep -vP 'ARP|ICMP|224.0.0.251' > data/all/traffic/all.txt

echo ">NOISE = excludes flags(service traffic) and known ports"
cat data/all/traffic/all.txt | grep -vP 'Flags|\.53: |\.53 >|\.445: |\.445 >|\.69: |\.69 >|\.123: |\.123 >|\.19: |\.19 >' > data/noise/traffic/noise.txt

echo ">SIGNAL = http, https and dns"
cat data/all/traffic/all.txt | grep -P '\.80: |\.80 >|\.53: |\.53 >|\.443: |\.443 >' > data/signal/traffic/signal.txt

echo ">SYNFLOOD = In/Out port 600 traffic"
cat data/all/traffic/all.txt | grep -P '\.600: |\.600 >' > data/synflood/traffic/synflood.txt	

echo ">FRAGGLE = In/Out port 19 traffic"
cat data/all/traffic/all.txt | grep '\.19: ' > data/fraggle/traffic/fraggle.txt

echo ">PORTSCAN = Ports without service running"
cat data/all/traffic/all.txt | grep -vP '\.600: |\.600 >' | grep -P '\.21: |\.21 >|\.22: |\.22 >|\.23: |\.23 >|\.25: |\.25 >|\.110: |\.110 >|\.143: |\.143 >|\.161: |\.161 >|\.445: |\.69: |\.123: ' > data/portscan/traffic/portscan.txt

for port in 80 443 53 21 22 23 25 110 143 161 69 123 445 600 19 67 68
do
	# selects traffic per port
	cat data/all/traffic/all.txt | grep -P "\."$port": |\."$port" >" > data/all/traffic/all_$port.txt
	cat data/noise/traffic/noise.txt | grep -P "\."$port": |\."$port" >" > data/noise/traffic/noise_$port.txt
	cat data/signal/traffic/signal.txt | grep -P "\."$port": |\."$port" >" > data/signal/traffic/signal_$port.txt
	cat data/synflood/traffic/synflood.txt | grep -P "\."$port": |\."$port" >" > data/synflood/traffic/synflood_$port.txt
	cat data/fraggle/traffic/fraggle.txt | grep -P "\."$port": |\."$port" >" > data/fraggle/traffic/fraggle_$port.txt
	cat data/portscan/traffic/portscan.txt | grep -P "\."$port": |\."$port" >" > data/portscan/traffic/portscan_$port.txt	

	# counts packets per minute
	cat data/all/traffic/all_$port.txt | awk -F: '{print $1":"$2}' | awk '{a[$1]+=1}END{for(i in a) print i,a[i]}' | sort > data/all/traffic/minutes_$port.txt
	cat data/noise/traffic/noise_$port.txt | awk -F: '{print $1":"$2}' | awk '{a[$1]+=1}END{for(i in a) print i,a[i]}' | sort > data/noise/traffic/minutes_$port.txt
	cat data/signal/traffic/signal_$port.txt | awk -F: '{print $1":"$2}' | awk '{a[$1]+=1}END{for(i in a) print i,a[i]}' | sort > data/signal/traffic/minutes_$port.txt
	cat data/synflood/traffic/synflood_$port.txt | awk -F: '{print $1":"$2}' | awk '{a[$1]+=1}END{for(i in a) print i,a[i]}' | sort > data/synflood/traffic/minutes_$port.txt
	cat data/fraggle/traffic/fraggle_$port.txt | awk -F: '{print $1":"$2}' | awk '{a[$1]+=1}END{for(i in a) print i,a[i]}' | sort > data/fraggle/traffic/minutes_$port.txt
	cat data/portscan/traffic/portscan_$port.txt | awk -F: '{print $1":"$2}' | awk '{a[$1]+=1}END{for(i in a) print i,a[i]}' | sort > data/portscan/traffic/minutes_$port.txt

	# counts packets per second
	cat data/all/traffic/all_$port.txt | awk -F. '{print $1}' | awk '{a[$1]+=1}END{for(i in a) print i,a[i]}' | sort > data/all/traffic/seconds_$port.txt
	cat data/noise/traffic/noise_$port.txt | awk -F. '{print $1}' | awk '{a[$1]+=1}END{for(i in a) print i,a[i]}' | sort > data/noise/traffic/seconds_$port.txt
	cat data/signal/traffic/signal_$port.txt | awk -F. '{print $1}' | awk '{a[$1]+=1}END{for(i in a) print i,a[i]}' | sort > data/signal/traffic/seconds_$port.txt
	cat data/synflood/traffic/synflood_$port.txt | awk -F. '{print $1}' | awk '{a[$1]+=1}END{for(i in a) print i,a[i]}' | sort > data/synflood/traffic/seconds_$port.txt
	cat data/fraggle/traffic/fraggle_$port.txt | awk -F. '{print $1}' | awk '{a[$1]+=1}END{for(i in a) print i,a[i]}' | sort > data/fraggle/traffic/seconds_$port.txt
	cat data/portscan/traffic/portscan_$port.txt | awk -F. '{print $1}' | awk '{a[$1]+=1}END{for(i in a) print i,a[i]}' | sort > data/portscan/traffic/seconds_$port.txt
done
echo ">Done!"
