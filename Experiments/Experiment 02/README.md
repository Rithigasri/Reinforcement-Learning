# EXPERIMENT 02:POLICY EVALUATION
## AIM:
To develop a program to evauate the performance of a policy.

## PROBLEM STATEMENT:
The problem statement is a Five stage slippery walk where there are five stages excluding goal and hole.The problem is stochastic thus doesnt allow transition probability of 1 for each action it takes.It changes according to the state and policy.
### State Space:
The states include two terminal states: 0-Hole[H] and 6-Goal[G].  
It has five non terminal states includin starting state.
### Action Space:
* Left:0
* Right:1
### Transition probability:
The transition probabilities for the problem statement is:
* 50% - The agent moves in intended direction.
* 33.33% - The agent stays in the same state.
* 16.66% - The agent moves in orthogonal direction.
### Reward:
To reach state 7 (Goal) : +1
otherwise : 0

## POLICY EVALUATION FUNCTION
``` python
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    # Write your code here to evaluate the given policy
    while True:
      V=np.zeros(len(P))
      for s in range(len(P)):
        for prob,next_state,reward, done in P[s][pi(s)]:
          V[s]+=prob*(reward+gamma*prev_V[next_state]*(not done))
      if np.max(np.abs(prev_V-V))<theta:
        break
      prev_V=V.copy()

    return V
```

## OUTPUT:
### Policy 1:
* Policy:<br/>  
![image](https://github.com/Rithigasri/Reinforcement-Learning/assets/93427256/c849e4ea-05ac-48e7-a67a-222630b4615d)<br/>
* Probability of success:<br/>
![image](https://github.com/Rithigasri/Reinforcement-Learning/assets/93427256/77799f3d-692d-41fd-beb0-49c746f53e74)<br/>
* State-value function:<br/>
![image](https://github.com/Rithigasri/Reinforcement-Learning/assets/93427256/3b620791-e707-4088-957f-058be901bf1a)<br/>
* Array:<br/>
![image](https://github.com/Rithigasri/Reinforcement-Learning/assets/93427256/ec16334c-db3e-4cb1-b523-d07af92adace)<br/>


### Policy 2:
* Policy:<br/>
![image](https://github.com/Rithigasri/Reinforcement-Learning/assets/93427256/6e9ee303-5a06-4af3-8db8-a93fe3c44c79)<br/>
* Probability of success:<br/>
![image](https://github.com/Rithigasri/Reinforcement-Learning/assets/93427256/3195870a-1d3c-4e4f-a890-26f2fb53755e)<br/>
* State-value function:<br/>
![image](https://github.com/Rithigasri/Reinforcement-Learning/assets/93427256/6f748a76-3bb1-43b5-96b7-b4a5919a3b75)<br/>
* Array:<br/>
![image](https://github.com/Rithigasri/Reinforcement-Learning/assets/93427256/5e4ab2be-ee94-4e95-94f3-55dac11ccd4c)<br/>

### Comparison:
![image](https://github.com/Rithigasri/Reinforcement-Learning/assets/93427256/c8f99521-db2b-427e-9971-e80b709be6ac)<br/>
![image](https://github.com/Rithigasri/Reinforcement-Learning/assets/93427256/94cc84e6-7f20-4844-8d4b-e78c2bf5e2e4)<br/>

### Interpretation:
* The value function of two policies is calculated. The policy with higher state value function has more probability to reach the goal.
* Thus, by comparing two value functions the occurence of True is less for V1 than V2.
* Therby, policy 2 is better in terms of reaching the goal based on the state value function.
## RESULT:
Thus, a program is executed to evaluate the performance of a policy.
