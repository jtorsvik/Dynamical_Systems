#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 28 November 2021 v1.3
# Made by Professor Luc Berthouze, University of Sussex

def gillespie_ABA(N,B0,beta,gamma,Tmax):
    
    from random import expovariate # Generate variates from exponential distribution
    import numpy as np
    
    A = [N - B0] # We cannot predict how many elements there will be unfortunately
    B = [B0]
    T = [0] 
    state = np.random.permutation([0] * (N-B0) + [1] * B0) # Randomly allocate B0 individuals to have state B (state=1), A (state=0) otherwise 
    B_contacts = np.where(state == 1)[0] # Index of individuals in state B (state=1).
    rate_vector = B0 * beta * np.ones((N, 1)) / N # Set rates to be B0*beta/N (rate for individuals in state A) to all individuals (initialisation). 
    rate_vector[B_contacts] = gamma # Update rate of B_contacts to be gamma (the rate for individuals in state B)
    
    time = 0
    while time <= Tmax + 0.5: # some (arbitrary) buffer after Tmax
        rate = np.sum(rate_vector) # Total rate (refer to Gillespie algorithm for details)
        cumrate = np.cumsum(rate_vector) # Cumulated sum of rates
        if rate > 0.000001: # if rate is sufficiently large
            tstep = expovariate(rate) # Pick an exponentially distributed time. Beware of difference with exprnd in Matlab where it is 1/rate
            T.append(T[-1] + tstep) # Time of next event
            event = np.where(cumrate > np.random.rand()*rate)[0][0] # Find which individual will see its state change 
            if state[event] == 0: # individual is in state A 
                A.append(A[-1] - 1) # this state A individual becomes state B so number of state A individuals is decreased
                B.append(B[-1] + 1) # obviously, number of state B individuals is increased 
                state[event] = 1 # Update state vector
                rate_vector[event] = gamma # Change rate of individual to B->A rate, namely gamma
                A_contacts = np.where(state == 0)[0] # List of state A individuals after change
                rate_vector[A_contacts] += beta / N # Update rate of state A individuals to account for the extra state B individual
            else: # individual is in state B
                B.append(B[-1] - 1) # this state B individual becomes state A so number of state B individuals is decreased
                A.append(A[-1] + 1) # obviously, number of state A individuals is increased
                state[event] = 0 # Update state vector
                A_contacts = np.where(state == 0)[0] # List of state A individuals after changes                                
                rate_vector[A_contacts] = beta * len(np.where(state == 1)[0]) / N # Update rate of state A individuals based on number of B individuals  
        else: # Nothing will happen from now on so we can accelerate the process
            time = T[-1] # current time
            while time <= Tmax + 0.5:
                A.append(A[-1]) # Just keep things as they are
                B.append(B[-1])
                T.append(T[-1] + 0.5) # arbitrarily add 0.5 to clock
                time = T[-1]
        # Update time and proceed with loop 
        time = T[-1]         

    return T,A,B

