**Description of the problem:**

The  objective  is  to  develop  a  deep  double  Q  learning  algorithm  with  neuralnetworks as the Q function approximators in the Acrobot – v1 environment.

**1.1  Acrobot environment:**
It  consists  of  two  links  and  two  joints  with  only  the  joint  between  two  linksare actuated.  Initially the two links are hanging downwards, and the goal is toswing the end of the bottom arm up to a given height.  The state consists ofsin (), cos () and the joint angular velocities of the two rotational angles.  Theaction is either applying +1, 0 or -1 torque on the joint between the two links

**1.2 Double Q Learning:**
The purpose of the learning algorithms in reinforcement learning is to estimatethe best action at a given state.  This is the purpose of Q function.  Optimalaction value function is defined as the maximum expected return given a states and and action a.  Therefore, the agent performs an action with the maximumvalue of Q function, given by Bellman equation as,
Q(s,a) =r+γ∗max(Q(s′,a′))

if we are in state s and perform action a to reach at next state s’.  Neuralnets are used to estimateQby gradient descent update of the parameters.We choose an action and it moves towards the next state to receive awardwhich is all added to as the transition in the replay memory.  A transition refersto state, action, reward, next state and done.  As the replay memory is beingbuilt, a random mini batch is drawn from the replay memory.  When a life islost or objective is reached, done is True in the replay memory.The main network estimates the Q for a particular value of current state.The main network finds the best action for the next state and another neuralnetwork  which  has  the  same  architecture  of  the  main  network  estimates  thetarget Q values which is added with the reward to get new Q value.  The main
network is updated using the mean squared difference between predicted Q andthe new Q values.The Q learning gives high Q values as the maximum value in the Bellmanequation chooses small positive values.  In Q learning we estimate the Q valuesin  next  state  using  the  target  network.   Instead  we  estimate  which  action  isbest  using  the  main  network  and  then  estimate  the  Q  value  for  that  actionusing target network.  This updated Q value is multiplied with gamma and thenadded to reward for action a.

**1.3  Description of the neural networks:**
All  the  layers  are  fully  connected  layers  with  64  neurons  each  except  for  theoutput layer which has 3 outputs corresponding to the three different actions.The  DQN  uses  Relu  activation  function  in  the  hidden  layers,  normal  kernelinitializer and Adam optimizer.  These hyper parameters gave us good resultswithout much effort in hyper parameter search.The  same  neural  network  architecture  is  used  for  estimating  both  actionvalue function and target action value function.  After every 20 steps, we updatethe target network using the main network.

**1.4  Exploration - Exploitation:**
Using  the  best  action  that  produces  the  maximum  value  of  Q  function  maynot produce good results.  This is because by choosing the best action leads toexploitation and not exploration.  The agent might go on sticking to the actioneven though it might produce a small reward.  Therefore we use greedy policy which will help in exploration as well as exploitation.  In this process, we choosea random action with a probability of and with the remaining probability wechoose best action.  There is a need to choose an epsilon decay rate so that asthe number of episodes are increased, the value of epsilon become very small sothat exploitation is maximum towards the end.

**3  Possible Improvement:**

Duelling DQN is one possible improvement strategy for this model.  Its possibleto decompose Q(s, a) as the sum of sum of value function V(s) and Advantage oftaking that action at that state A(s, a).  Therefore the neural network estimatorseparately  estimates  the  V(s)  and  A(s,a).   This  helps  to  learn  if  the  state  isimportant without evakuating the effect of each action at that state.  That is ifthe value function at a state is bad, there is no need to find all actions possiblein that state.The second possible improvement is through Prioritized Experience Replay.But  randomly  sampling  has  the  chances  that  some  of  the  experiences  whichoccur  rarely  has  no  chance  of  selecting  even  though  it  might  be  important.Therefore a new probability distribution is defined that define the priority ofeach transition tuple.  Priority is defined as the difference between the predictionand target.  Priority value is put in the transition of each replay memory.  A
stochastic optimization is introduced to generate the probability of being chosenfor the replay.

**4. References:**

1.  “Improvements in Deep Q Learning:  Dueling Double DQN, PrioritizedExperience Replay, and Fixed...” FreeCodeCamp.org, FreeCodeCamp.org,16 Apr. 2018, https://www.freecodecamp.org/news/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682/.
2.  “fg91/Deep-Q-Learning.”  GitHub,  https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb
3.  “rae0924/AcrobotDQN.” GitHub, https://github.com/rae0924/AcrobotDQN/blob/master/agent.py.

