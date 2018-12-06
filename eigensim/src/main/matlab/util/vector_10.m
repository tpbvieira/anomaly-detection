b=1;
y=0;
j=1;
m(b)=0;
while j~=121;
    for i=j:(j+9) 
        y = w(i)+ y; 
    end
    m(b+1)=y;
    b=b+1;
    j=j+10;
    y=0;
end