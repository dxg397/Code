%hopfield net for TSP
clear all
load 'intercity_distances.dat'; %read in the intercity distances
temp = size(intercity_distances);
Ncities=temp(1);
%%%%%%%%%%%%%

% the following parameters are defined by Hopfield, pg 147 of
% "Neural Computation of Decisions in Optimization Problems"
%values listed are those suggested.  Experiment with alternative values
A=500.0;
B=500.0;
C=200.0;
D=1000.0;%=0 TEST looking for viable solns only--no penalty for trip length

u0 = -1.75; %find initial value for U such that sum of V's starts out ~= Ncities
Jbias = C*Ncities;
Jbias_factor=1.0; %1.5;
LAMBDA=5000.0; %// addl factor introduced by wsn for
                          % influence of integral-error term that
                         %enforces eventual convergence to a legal soln
%NN_CONV_RETRYS=1000;
 DT=0.0001; %0.00001;  % // time step for one-step Euler integration of
                         % differential eqns; need this small enough so 
                          % simulation does not blow up

%initializations...
%Tabc is set of synapses, excluding the cost-of-trip penalty
%W is the synapses, including cost-of-trip penalty
[W,Tabc] = assign_weights(A,B,C,D,Ncities,intercity_distances);
J = ones(Ncities,Ncities)*Jbias*Jbias_factor;  %bias currents; per Hopfield, may need to adjust this term
int_Eabc=zeros(Ncities,Ncities); %optional for integral-error feedback
%matrix of neurons: row index==> city, column index==> day
%initialize U:
%force city 1 on day 1:
U=ones(Ncities,Ncities)*u0; %nominal value, then add noise
dU=(rand(10,10)-0.5)*2; %noise term, range +/-1;
%figure(3)
%bar3(dU)
U=U+dU*u0*0.01; %add  noise as fraction of  u0
U(1,:)=-10.0; %suppress change for city-1, all days
U(:,1)=-10.0; %suppress change for day1, all cities
U(1,1)=10; %coerce initialization for city 1 on day 1
%U(2,2)=10; %debug...
Udot=zeros(10,10);
V=logsig(U);  %corresponding Vs; use 0 to 1 sigmoid squashing fnc
temp = sum(V);
sumV=sum(temp)

%bar3(V); %to visualize neural outputs
niter = 1;
while niter>0
    for i=1:niter
        [Udot,int_Eabc] = compute_udot(V,W,U,J ,DT,LAMBDA,int_Eabc,Tabc,Jbias);%Jbias); %function [Udot] = compute_udot(V,T,DT,LAMBDA,int_Eabc)
        %[Udot,int_Eabc] = compute_udot2(V,W,U,J ,DT,LAMBDA,int_Eabc,Tabc,Jbias,Tab);%Jbias); %function [Udot] = compute_udot(V,T,DT,LAMBDA,int_Eabc)

        U = U+Udot*DT;
        V=logsig(U); 
        figure(1)
        bar3(V);
        title('sigmas')
        figure(2)
        bar3(Udot)
        title('udot')
        figure(3)
        bar3(int_Eabc)
        title('int Eabc')
    end
%    dist=eval_abstract_cost(V,intercity_distances)
%    valid = eval_validity(V,0.9) %give net outputs and threshhold requirement for defining a trip
   figure(1)
    bar3(V);
    temp = sum(V);
    sumV=sum(temp)
    tripcost = compute_trip_cost(V,intercity_distances)
   %  input  number of iterations-- quit for negative
   niter = input('enter number of iterations (<=0 to quit)')
end
hist(tripcost, niter)
title('Histogram of TSP Trip Costs');
xlabel('Trip Cost')
ylabel('Number of Itineraries')

% print out statistics about trip cost history
fprintf('Average iteration count per solution: %f\n', mean(niter));
fprintf('Mean trip cost: %f\n', mean(tripcost));
fprintf('Std trip cost: %f\n', std(tripcost));
fprintf('Min trip cost: %f\n', min(tripcost));
fprintf('Max trip cost: %f\n', max(tripcost));



