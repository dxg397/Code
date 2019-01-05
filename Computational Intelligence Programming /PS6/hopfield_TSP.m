% hopfield net for TSP

clear all

% read in the intercity distances
load 'intercity_distances.dat'; 
temp=size(intercity_distances);
Ncities=temp(1);

% the following parameters are defined by Hopfield, pg 147 of
% "Neural Computation of Decisions in Optimization Problems"
% values listed are those suggested.  Experiment with alternative values
A=50.0;
B=50.0;
C=20.0;

% TEST looking for viable solns only--no penalty for trip length
D=1000.0; %=0 

Jbias=C*Ncities;
Jbias_factor=1.0; %1.5;

% additional factor introduced by wsn for
% influence of integral-error term that
% enforces eventual convergence to a legal soln
LAMBDA=500.0; 

% time step for one-step Euler integration of
% differential eqns; need this small enough so 
% simulation does not blow up
DT=0.01; %0.00001;

% initializations...
% Tabc is set of synapses, excluding the cost-of-trip penalty
% W is the synapses, including cost-of-trip penalty
[W,Tabc]=assign_weights(A,B,C,D,Ncities,intercity_distances);

% bias currents; per Hopfield, may need to adjust this term
J=ones(Ncities,Ncities)*Jbias*Jbias_factor;  

% find initial value for U such that sum of V's starts out ~= Ncities
% initial U of negative log function on number of cities minus one
u0=-log(Ncities-1);

% scale noise on input
noise_scale = 0.1; 

% iterate through n-cities problem this number of times
MAX_ITERS=100;
curr_iter=1;

% store trip costs for statistics
trip_costs=zeros(MAX_ITERS,1);

% store num of iters until convergence for statistics
num_iters_total=zeros(MAX_ITERS,1);

while curr_iter<=MAX_ITERS
    % noise term, range +/-1;
    dU=(rand(10,10)-0.5)*2;
    
    % matrix of neurons: row index==> city, column index==> day
    % initialize U:
    % force city 1 on day 1:
    % nominal value, then add noise
    U=ones(Ncities,Ncities)*u0; 

    % add noise to U
    U=U+dU*noise_scale; 

    % suppress change for city-1, all days
    U(1,:)=-10.0; 

    % suppress change for day1, all cities
    U(:,1)=-10.0; 

    % coerce initialization for city 1 on day 1
    U(1,1)=10; 

    % corresponding Vs; use 0 to 1 sigmoid squashing fnc
    V=logsig(U);

    fprintf('Computing trip %d for TSP\n', curr_iter);

    Udot=zeros(10,10);
    ndistinct=0;
    ndist_prev=0;
    num_iters_curr=0;
    
    % integral-error feedback matrix
    int_Eabc=zeros(Ncities,Ncities);
    
    % run n-cities Hopfield evaluation
    while ndistinct < Ncities
        num_iters_curr=num_iters_curr+1;
        
        [Udot,int_Eabc]=compute_udot(V,W,U,J,DT,LAMBDA,int_Eabc,Tabc,Jbias);

        U=U+Udot*DT;
        V=logsig(U);
        [tripcost, ndistinct]=compute_trip_cost(V,intercity_distances);
        
%         % only display when more valid cities are found
%         if ndistinct > ndist_prev
%             ndist_prev=ndistinct;
%             fprintf('Number of distinct trips is: %d\n', ndistinct);
%             fprintf('Current trip cost is: %f\n', tripcost);
%             figure(1)
%             bar3(V);
%             title('sigmas')
%             figure(2)
%             bar3(Udot)
%             title('udot')
%             figure(3)
%             bar3(int_Eabc)
%             title('int Eabc')
%             pause(1)
%         end
    end
    num_iters_total(curr_iter)=num_iters_curr;

    % figure(1)
    % bar3(V);
    
    % find the total output of all neurons in net
    % sumV=sum(sum(V));
    tripcost=compute_trip_cost(V,intercity_distances);
    trip_costs(curr_iter)=tripcost;
    curr_iter=curr_iter+1;
end

% create a histogram of trip cost data
hist(trip_costs, MAX_ITERS)
title('Histogram of TSP Trip Costs');
xlabel('Trip Cost')
ylabel('Number of Itineraries')

% print out statistics about trip cost history
fprintf('Average iteration count per solution: %f\n', mean(num_iters_total));
fprintf('Mean trip cost: %f\n', mean(trip_costs));
fprintf('Std trip cost: %f\n', std(trip_costs));
fprintf('Min trip cost: %f\n', min(trip_costs));
fprintf('Max trip cost: %f\n', max(trip_costs));