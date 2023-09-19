# EXPERIMENT 04:VALUE ITERATION ALGORITHM

## AIM:
To perform value iteration algorithm for the given MDP.
## PROBLEM STATEMENT:
A 4*4 frozen lake environment is taken into consideration where there are 4 holes and 1 goal which is considered as terminal state.It is stochastic environment and the transition probability varies according to the actions taken.
### State Space:
{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
### Action space:
{Left[0],Right[1],Up[2],Down[3]}
### Transition probability:
The transition probabilities for the problem statement is:  
* 33.33% - The agent moves in intended direction.  
* 66.66% - The agent moves in orthogonal direction.  
### Reward:
To reach state 15 (Goal) : +1 otherwise : 0
## VALUE ITERATION ALGORITHM:
1. Initialize the value function `V` with zeros for each state.
2. Repeat the following until the change in `V` for all states is smaller than a threshold `theta`:  
   a. Initialize a Q-value function `Q` with zeros for each state-action pair.  
   b. For each state `s` and action `a`, compute the Q-value using the Bellman equation:   
      Q[s][a] = Σ[prob * (reward + gamma * V[next_state])] for all possible transitions (prob, next_state, reward, done).  
   c. Update `V` by taking the maximum Q-value for each state: V[s] = max(Q[s]).  
3. Define a policy `pi` that selects actions by maximizing the Q-values: pi(s) = argmax(Q[s]).
4. Return the final value function `V` and the corresponding policy `pi`.

## VALUE ITERATION FUNCTION:
``` python
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
      Q=np.zeros((len(P),len(P[0])),dtype=np.float64)
      for s in range(len(P)):
        for a in range(len(P[s])):
          for prob,next_state,reward,done in P[s][a]:
            Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
      if(np.max(np.abs(V-np.max(Q,axis=1))))<theta:
        break
      V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s , a in enumerate(np.argmax(Q,axis=1))}[s]
    return V, pi
```

## OUTPUT:
### OPTIMAL POLICY:  
![image](https://github.com/Rithigasri/Reinforcement-Learning/assets/93427256/5b1d2fd1-d159-4baf-8ede-573a9eaeaf2d)
### OPTIMAL VALUE FUNCTION:  
![image](https://github.com/Rithigasri/Reinforcement-Learning/assets/93427256/83d7f7a8-ea01-46b9-a524-855e4ca56f37)
### SUCCESS RATE FOR THE OPTIMAL POLICY:  
![image](https://github.com/Rithigasri/Reinforcement-Learning/assets/93427256/690d4519-458b-4391-8af6-1901a558e966)

## RESULT:
Thus, a program is developed to perform value iteration for given MDP.
