clc;
clear all;
close all;
 
N = 10;
 
u = rand(N,1);
y = zeros(N+1,1);
 
y_act = zeros(N+1,1);
y_act(1) = 1;
y_act(2) = -0.5*y_act(1) + 0.5*u(1);
for j = 2:length(u)
    y_act(j+1) = -0.5*y_act(j) - y_act(j-1) + 0.5*u(j);
end
 
eta = 0.01;
 
 
w1 = rand;
w2 = rand;
w3 = rand;
 
p_w1 = zeros(N+1,1);
p_w2 = zeros(N+1,1);
p_w3 = zeros(N+1,1);
 
no_of_eps = 5000;
 
y(1) = 1;
 
cost = zeros(no_of_eps,1);
 
for i = 1:no_of_eps
    for j = 1:length(u)
        
        if j == 1
            y(j+1) = w1*y(j) + w3*u(j);
            p_w1(j+1) = y(j) + w1*p_w1(j);
            p_w2(j+1) = w1*p_w2(j);
            p_w3(j+1) = u(j) + w1*p_w3(j);
            
        else
            y(j+1) = w1*y(j) + w2*y(j-1) + w3*u(j);
            p_w1(j+1) = y(j) + w1*p_w1(j) + w2*p_w1(j-1);
            p_w2(j+1) = y(j-1) + w1*p_w2(j) + w2*p_w2(j-1);
            p_w3(j+1) = u(j) + w1*p_w3(j) + w2*p_w3(j-1);
        
        end
        
        dw1 = -(y_act(j+1) - y(j+1))*p_w1(j+1);
        dw2 = -(y_act(j+1) - y(j+1))*p_w2(j+1);
        dw3 = -(y_act(j+1) - y(j+1))*p_w3(j+1);
        
        w1 = w1 - eta*dw1;
        w2 = w2 - eta*dw2;
        w3 = w3 - eta*dw3;
    end
    
    for k = 1:length(u)
        if k==1
            y(k+1) = w1*y(k) + w3*u(k);
        else
            y(k+1) = w1*y(k) + w2*y(k-1) + w3*u(k);
        end
    end
            
    cost(i) = (1/(N+1))*sum(0.5*((y - y_act).^2));
    if cost(i) == 0
        break;
    end
end
fprintf('Training Set Error:');
disp(cost(i));
fprintf('Optimal value of w1:');
disp(w1);
fprintf('Optimal value of w2:');
disp(w2);
fprintf('Optimal value of w3:');
disp(w3);
 
figure(1);
epoch_no = 1:no_of_eps;
err = cost';
semilogy(epoch_no,err,'color',[0 0.6 0.3],'linewidth',1.5);
xlabel('\bf Number of epochs');
ylabel('\bf Cost (MSE)');
title('\bf Learning Curve');
 
figure(2);
K = 1:length(u)+1;
plot(K,y_act,'ro','MarkerFaceColor','r');
xlabel('\bf {\it k}');
hold on;
plot(K,y,'bo','MarkerFaceColor','b');
legend('Predicted Output','Target Output');
title('\bf Training Set: Predicted v/s Actual Output');
