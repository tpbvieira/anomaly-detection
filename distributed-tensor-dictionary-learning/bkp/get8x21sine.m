function D = get8x21sine(betamin, makePlot)
% get8x21sine     Get the 8x21 dictionary (frame) with sine elements
%
% examples:
% D = get8x21sine(betamin, makePlot);  
% % betamin is minimal angle between vectors, should be <= 51.8 (degrees)
% D = get8x21sine(36);    % include the DCT synthesis vectors and more LP
% D = get8x21sine(47.5);  % include the DCT synthesis vectors
% D = get8x21sine(51.8, 1);

if (nargin < 1); 
    betamin = 51.8;
end
if (nargin < 2); 
    makePlot = false; 
    sc = 0;
else
    sc = 0.75;
end

N = 8; 
K = 21;
D = zeros(N,K);
fp = zeros(2,K);   % fs, phase

if (betamin >= 50)
    k = 1;
    D(:,k) = ones(N,1)*sqrt(1/N);
    clim = cos(betamin*pi/180);
    % try many
    L = 250;
    for th = (1:(L-1))*(pi/L)
        for phi = (1:(L-1))*(pi/L);
            dsine = d(N,th,phi);
            c = D(:,1:k)'*dsine;  % cos of angles
            if (max(abs(c)) < clim)
                k = k+1;
                D(:,k) = dsine;
                fp(:,k) = [th, phi]';
            end
            if (k>K); break; end;
        end
        if (k>=K); break; end;
    end
else
    % for betamin < 50 different alternatives may be tried
    % now use DCT and add sine vectors
    D(:,1:N) = idct(eye(N));
    % also frequency and phase for these
    fp(:,1:N) = [(0:(N-1))*pi/N; pi/2 + (0:(N-1))*pi/(2*N)];
    k = N;
    clim = cos(betamin*pi/180);
    % try many
    L = 350;
    for th = (1:(L-1))*(pi/L)
        for phi = (1:(L-1))*(pi/L);
            dsine = d(N,th,phi);
            c = D(:,1:k)'*dsine;  % cos of angles
            if (max(abs(c)) < clim)
                k = k+1;
                D(:,k) = dsine;
                fp(:,k) = [th, phi]';
            end
            if (k>K); break; end;
        end
        if (k>=K); break; end;
    end
end

% sort the vectors according to frequency
[temp,I] = sort(fp(1,:));
D = D(:,I);
fp = fp(:,I);
for k=1:K
    if (D(1,k) < 0)
        D(:,k) = -D(:,k);
    end
end

if makePlot
    k = 1; 
    h=figure(1);clf;hold on;
    set(h,'Color',0.95*[1,1,1]);
    for y0 = 3:(-1):1
        for x0 = 1:7;
            % plot dictionary element k
            x = x0 + (2:(N+1))/12; 
            h = text(x0+1/12,y0,int2str(k));
            set(h,'HorizontalAlignment','right');
            plot([x(1),x(end)], [y0, y0], 'b-');
            for n=1:N
                plot([x(n), x(n)], [y0, y0+sc*D(n,k)], 'b-');
                plot(x(n), y0+sc*D(n,k), 'b.');
            end
            % plot sine curve
            try
                if k == 1
                    x = x0 + (2:(N+1))/12; 
                    s = D(:,k);
                else
                    s = sin(fp(2,k)+(0:(N-1))*fp(1,k));
                    ab = [s(:),ones(N,1)]\D(:,k);
                    a = ab(1); b = ab(2);
                    s = a*sin(fp(2,k)+(0:0.1:(N-1))'*fp(1,k))+b;
                    x = x0 + (2:0.1:(N+1))/12;
                end
                plot(x(:),y0+sc*s(:),'g-');
            catch E1
                disp(E1.message);
            end
            k = k+1;
        end
    end
    title('The dictionary vectors for a 8x21 sine dictionary.');
    axis off;
end

return;


% sample the sine function with 8 points, start at phase phi
% and go forward theta at each sample.
% Make the result orthogonal to DC-atom (scaled ones) or return DC
function dsine = d(N,theta,phi)
dsine = sin(phi+theta*(0:(N-1))');
dd = dsine'*dsine;
if dd>(1/(N*N))   
    dc = ones(N,1)*sqrt(1/N);
    dsine = dsine/sqrt(dd);
    dsine = dsine-(dc'*dsine)*dc;
    dd = dsine'*dsine;
    dsine = dsine/sqrt(dd);
else
    dsine = ones(N,1)/sqrt(N);
end
return
