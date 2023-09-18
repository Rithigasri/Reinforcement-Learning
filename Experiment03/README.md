# EXPERIMENT 03:POLICY ITERATION ALGORITHM
## AIM:
To implement a policy iteration algorithm for the given MDP.
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

## POLICY ITERATION ALGORITHM


## POLICY IMPROVEMENT FUNCTION
```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    # Write your code here to implement policy improvement algorithm
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob,next_state,reward,done in P[s][a]:
          Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
    new_pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return new_pi
```

## POLICY ITERATION FUNCTION
```python
def policy_iteration(P, gamma=1.0, theta=1e-10):
    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    # Write your code here to implement the policy iteration algorithm
    pi=lambda s:{s:a for s,a in enumerate(random_actions)}[s]
    while True:
      old_pi={s:pi(s) for s in range(len(P))}
      V=policy_evaluation(pi,P,gamma,theta)
      pi=policy_improvement(V,P,gamma)
      if old_pi=={s:pi(s) for s in range(len(P))}:
        break

    return V, pi
```

## OUTPUT:
Mention the optimal policy, optimal value function , success rate for the optimal policy.

## RESULT:
Thus, a program is developed to perform policy iteration for the given MDP.
