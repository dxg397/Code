%assign synapses per Hopfield
function [T,Tabc] = assign_weights(A,B,C,D,N_CITIES,costs)	
   N_DAYS=N_CITIES;
   T=zeros(N_CITIES,N_DAYS,N_CITIES,N_DAYS);
   Tabc=T;  %Tabc is the weight matrix for ONLY the constraint terms, excluding trip cost
   Td = T; %weight matrix corresponding to trip-cost penalty
	% fill in T matrix: all 10,000 terms
	for X=1:N_CITIES
		for Y=1:N_CITIES
			for ix=1:N_DAYS
				for jy=1:N_DAYS
					Tabc(X,ix,Y,jy)= -C;%uniform inhibition
					if( (X==Y) && (ix ~= jy)) % penalize if same city on two different days
                        %FIX THIS!!
					  Tabc(X,ix,Y,jy) =Tabc(X,ix,Y,jy) -A ; 
                    end
					% penalize two cities on the same day
					% DO CORRECT ASSIGNMENT OF SYNAPSES FOR THIS
                    if (ix==jy)&&(X~=Y)
                        Tabc(X,ix,Y,jy)=Tabc(X,ix,Y,jy)-B;
                    end
					% distance data terms--no changes necessary
					% connect predecessor and successor city distances
					if(jy==ix-1) 
					 Td(X,ix,Y,jy) =Td(X,ix,Y,jy) - D*costs(X,Y);
                    end
                    
					if(jy==ix+1) 
					  Td(X,ix,Y,jy) =Td(X,ix,Y,jy)- D*costs(X,Y);
                    end
					%special case: ix1 = N_DAYS; wrap-around
					if((ix==N_DAYS)&&(jy==1))
						Td(X,ix,Y,jy) =Td(X,ix,Y,jy)- D*costs(X,Y);
                    end
					% another special case: ix=1, wrap-around ix-1
					if((ix==1)&&(jy==N_DAYS))
						Td(X,ix,Y,jy) =Td(X,ix,Y,jy)- D*costs(X,Y);
                    end
                end
            end
        end
    end
T=Tabc+Td; %combined effects of constraints and trip cost
