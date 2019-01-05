% for a given trip, compute the cost
function [tripcost, ndistinct] = compute_trip_cost(V,costs)
% first, threshhold V's to get a definitive itinerary:
temp = size(V);
Ncities = temp(1);
Ndays = Ncities;
Vbinary = 0*V;
for X=1:Ncities
    for i=1:Ndays
        if V(X,i)>0.5
            Vbinary(X,i)=1;
        end
    end
end
%convert to a trip:
trip = ones(1,Ncities+1);

for i=2:Ndays
    for X=2:Ncities
        if Vbinary(X,i)==1
            %assign this city
            trip(i)=X; 
        end
    end
end
tripcost=eval_soln(trip,costs)
%check if this trip is valid:
ndistinct=num_cities_visited(trip) 
if ndistinct ~=Ncities %valid tour
    tripcost = -tripcost;
end

        