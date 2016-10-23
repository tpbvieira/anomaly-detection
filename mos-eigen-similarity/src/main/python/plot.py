import matplotlib.pyplot as plt
import numpy as np

x = []
y = np.cos(y1)


1887545	2341327	3213867	133238294	6367983.40451472	708335
21831	212783	44266	531843	4596480	66547
5306	192	647	634322	32547	148
0	0.442400000000000	1.39360000000000	491	3139	0
0	0	0	0.322600000000000	0.0832000000000000	0

plt.plot(x,'g--^', label='regional')
plt.plot(y,'b-o', label='local')
plt.legend( loc='upper left', numpoints = 1 )
plt.show()
plt.savefig('test.eps')
