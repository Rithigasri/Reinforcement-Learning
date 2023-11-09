# EXPERIMENT 07: Q LEARNING ALGORITHM


## AIM
To implement Q learning algorithm for given environment of the problem statement.

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

## Q LEARNING ALGORITHM:
1. Initialize Q-values for all state-action pairs to zero.
2. Set learning rate (alpha) and exploration rate (epsilon) decay schedules.
3. For each episode:
   a. Reset the environment and choose an initial action using epsilon-greedy strategy.
   b. Repeat until the episode is done:
      i. Take the chosen action, observe the next state and reward.
      ii. Update the Q-value for the current state-action pair using the Q-learning update rule.
      iii. Move to the next state.
4. Track the evolution of Q-values, policy, and other metrics over episodes.
5. Compute the optimal value function (V) and policy (pi) based on the final Q-values.
6. Return the learned Q-values, optimal value function, optimal policy, and tracking information.

## Q LEARNING FUNCTION:
``` python
def q_learning(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action=lambda state,Q,epsilon: np.argmax(Q[state]) if np.random.random()>epsilon else np.random.randint(len(Q[state]))
    alphas=decay_schedule(
        init_alpha,min_alpha,
        alpha_decay_ratio,
        n_episodes)
    epsilons=decay_schedule(
        init_epsilon,min_epsilon,
        epsilon_decay_ratio,
        n_episodes)
    for e in tqdm(range(n_episodes),leave=False):
      state,done=env.reset(),False
      action=select_action(state,Q,epsilons[e])
      while not done:
        action=select_action(state,Q,epsilons[e])
        next_state,reward,done,_=env.step(action)
        td_target=reward+gamma*Q[next_state].max()*(not done)
        td_error=td_target-Q[state][action]
        Q[state][action]=Q[state][action]+alphas[e]*td_error
        state=next_state
      Q_track[e]=Q
      pi_track.append(np.argmax(Q,axis=1))
    V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
```
### Optimal Policy:  
![image](https://github.com/Rithigasri/Reinforcement-Learning/assets/93427256/fee60f33-1cca-45d2-9c4c-59bbfa4f8adc)
### First Visit Monte Carlo Method:
![image](https://github.com/Rithigasri/Reinforcement-Learning/assets/93427256/7ece11b1-a421-4d43-b019-f90bd6f63dd8)
### Q Learning Algorithm:
![image](https://github.com/Rithigasri/Reinforcement-Learning/assets/93427256/aeeb6566-059f-4d52-a83f-ae6b81fa3026)

### Plot for State Value Function -Monte Carlo VS Q Learning:
* First Visit Monte Carlo:
  ![image](https://github.com/Rithigasri/Reinforcement-Learning/assets/93427256/f02df975-7e22-4a2e-9452-b51509c1156e)
* Q Learning:  
![image](https://github.com/Rithigasri/Reinforcement-Learning/assets/93427256/af0d9d57-54b7-4e6e-b047-44b4e2baad50)

## RESULT:
Thus, the implementation of Q learning algorithm was implemented successfully.
