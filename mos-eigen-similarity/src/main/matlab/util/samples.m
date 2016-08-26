j=0;
w=0;
z=1;
k=1;
[a,b]=size(x);
for k = 1:120
    t=1;
    y=0;
    if j == 60 
       j=0;
       while z <= a && x(z) == j 
       y(t) = x(z);
       z=z+1;
       t=t+1;
       end
       j=j+1;
       w(k)=t-1;
       k=k+1;
    else    
       while z <= a && x(z) == j 
       y(t) = x(z);
       z=z+1;
       t=t+1;
       end
       j=j+1;
       w(k)=t-1;
       k=k+1;
end
w
end