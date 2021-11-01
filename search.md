## Search
#draft/aitp_thoughts
- - - -
All of the AI for theorem proving projects that I can think of have a search component.  It is not enough to just select next steps or tactics in a greedy fashion, but one must backtrack to find more and deeper proofs.

In many ways, this is not unique to theorem proving.  In robots, board games, and other combinatorial search problems (like the Rubik’s cube), search is an important part of the equation.  In many areas of AI, search goes by the name _planning_ (the idea being that a robot would plan through its available actions before executing on them in the real world).  

### Neural-guided search
Neural networks (and other ML methods) are generally used for two aspects of search.  A _policy_ is used to select the available next action.  Usually, this is done by specifying a probability distribution over all actions.  (Therefore each action has a small probability of occurring.)  Unlike chess, go, or the Rubik’s cube, in theorem proving (regardless of the formal system), there are usually way too many possible next actions, so we must select a handful to try out and rank them.   

The second thing they provide is a _value_.  Every state of the search is given a score.  It might be the number of steps left in the proof, or the probability that the proof would succeed.  (However, a good value function often has to be learned through RL, while a decent policy function can be learned through supervised learning—although RL will strength it.)

These policy and value replace the hand-crafted heuristics that are often found in other search algorithms like alpha-beta (mini-max) search, A* search, best-first search, beam search, and so on.  My impression so far is that:
* Neural-guided search works pretty well.
* Many search algorithms, like MCTS, beam search, best first search all work about the same more of less.
* Reinforcement learning can significantly strengthen the search heuristics by “trading compute for training single”.  In other words, by training the network with the result of a search, it can be strengthened to speed up that search again when it reaches another similar situation.
* Search still fails to find deep examples, and we are only scratching the surface of what is possible.

### Why proof search fails
I think a lot of deep introspection is needed into current methods and why the search fails, but I can offer some guesses and possible improvements.

Proof search is very different from board games for many reasons.  Board games are adversarial, so one can’t easily plan far ahead, because the opponent could cut off any plan.  This balances out the game and puts a lot of importance on any one move.  In proofs, there is no obvious adversary.  One can make a proof long or short.  One can even make some mistakes as long as they are reversible (like apply symmetry, only to have to apply it again later).

Games are made efficient.  There aren’t a lot of redundant or equivalent moves in go or chess.  Most every move is distinct and can lead to different gameplay.  However, in proofs, there are many possible actions, with a lot of redundancy built in.  This depends a lot on the proof system, but most tactic languages have a lot of redundant tactics.  One can use a powerful tactic like `simp` or a surgical tactic like `rewrite`.  There are also a lot of symmetries.  Sometimes one can apply two rewrites in either order, or one has unlimited choice what to name variables.  While some of these symmetries can be broken by hand, many are too subtle to manually account for.  Lean GPT-f for example, has a habit of spamming `intro n`, `intro x`, `intro m`, … if it is confident `intro` is the way to go.

It is no secret that multiple paths can lead to exponential blowup in “explore”-heavy proof searches like breadth first search, and even in initial stages of Monte Carlo tree search.  While some exploit-heavy search frameworks like depth-first are good at handling redundancy, this is true only if not too much backtracking I needed.  If, say, `intro n` turns out to be a bad choice, the algorithm also needs to learn that `intro m` is also a bad choice.  Part of this is just an issue that information doesn’t normally pass from one part of a tree search to another.

All this symmetry leads to having many paths which arrive at roughly equivalent proofs (or intermediate goals).  But since they are not exactly equivalent, hash set are not a feasible solution.  This can make the search space intractable.  Given the immense choice in actions, it is not surprising that proof search often has difficulty in producing proofs of only a few tactics in length due to combinatorial explosion. 

There may be a lot of duplication in a proof.  The same (or similar) goals can come up multiple times in a proof search, either (1) in the same proof, or (2) in a different attempt at the same proof.  While most graph search algorithms maintain sets of visited nodes, it is much harder to characterize different nodes which are close but not the same).  Again, this is an issue that information doesn’t travel between different nodes of a proof search for many algorithms.

I think as sexy as it is to improve the policies (through better neural networks and better data), it seems paramount to me to fundamentally reevaluate proof search.  Here are many ideas on how to improve search.

### Symmetry breaking and search pruning through proof compression
The DreamCoder paper provides some interesting ideas that could be used to speed up search with RL in almost any combinatorial setting.  While the specifics of DreamCoder are not (easily) directly applicable to most theorem proving environments, there is a general recipe that goes as follows:
* Build proofs using the current search algorithm.
* Compress those proofs as much as possible
* Learn only from the most compressed proofs.  (This breaks symmetries and passes information between different parts of a proof search as I’ll explain.)
* (Optional) Extract information from completed proofs, e.g. lemmas, which can be used later to increase compression.

Now, what does it mean to compress a proof.  In most policy-based proof-search frameworks, each proof step comes from a probabilistic policy which generates the proof step.  Each generated proof step therefore comes with a probability from 0 to 1.  (In Metamath and Lean GPT-f, this is the probability of the generated proof step string.  In DeepHOL, ProverBot9001, and ASTactic this is the product of the probabilities of all the components of the tactic.)  The probability of a whole proof is the product of the probabilities for each proof step (where each proof step is generated conditioned on some partial information in the proof like the current goal).  From an information theoretic perspective, the information (in bits) of a proof with probability `p` is `-log_2(p)`, so the higher the probability, the less the information (bits) needed to encode that proof.

By training on maximally compressed proofs, we can aid our search in many ways which heavily prune the search space thus allowing us to discover deeper proofs:
* The neural network starts to become more strongly opinionated on irrelevant details, like `intro n`, `intro m` or the order of certain rewrites.  This leads to less exploration and more exploitation when it comes to such details.  That leads to a deeper search.
* The network will tend to avoid useless combinations of steps, like applying commutativity twice in a row only to get back to where it started, or applying a theorem which doesn’t change the goal substantially.
* The network will avoid large proofs with repeated steps if it can.
* The network can pass information from one search state to another, coupling their probabilities and reducing their joint probability. Specifically,  as the network is trained to be better on one search state, it will also be better on other similar search states.  (However, see the next few sections for how to compress proofs with repeated steps even more by more closely coupling the proof states found in a proof to the overall proof attempt.)
* If the same proof is attempted many times, and one learns not only from the completed proofs but from any completed subgoals, once one subgoal is learned it becomes part of the training data, therefore that subgoal (and any similar proof states in the proof search) becomes easier to solve on the next pass.  (An online reinforcement learning algorithm that doesn’t involve training, like a sequential NN or a k-NN, could even make this more practical for real-time proof search.)
* If one stores lemmas, or even discovered proofs for common subgoals, during RL, then it might be possible to use those as information to speed up proofs the next time those exact situations are encountered again.  If these lemmas can be made more abstract they would also cover a greater number of cases.  See the DreamCoder paper for ideas on abstracting common subroutines.

When we realize how important compression is, we realize little things that can be done to improve RL:
* RL is not optional, but an integral part of deep proof search.
* During RL, the goal of proof search should be to find maximally compressed proofs, not just any proof.  Even after a proof is found, the search should continue for a bit more (within compute budget) to compress it more (or prove that there is no more compressible proof like in DreamCoder).
* Since the goal is to find a high probability proof, this should be taken into account in the proof search algorithm.  Specifically proof search becomes a shortest path finding algorithm, where the length of a proof is the sum of the negative log probabilities of each step.  At any partial stage of the proof search, an approximated length of a path going through that node is the length of the path (sum of negative log probabilities from the policy) plus the negative log of the probability of completing the proof from that node.  One obvious approach here is A* search.  Also, MCTS can be easily modified to search for shortest paths.  (At early stages of search when the algorithm isn’t very good, more exploration may be needed.  This can be done with added noise or small punishments for each proof step.)
* Learn from solved subgoals.
* Just as a paragraph can be better modeled (and hence compressed) as a single string instead of multiple disjoint sentences, similarly, a proof can be better modeled (and compressed) as a single proof instead of a disjoint collection proof steps.  The main idea is that different parts of the proof can depend on each other.  (Even if they don’t logically depend on each other, they could require similar proofs and similar dependencies.  Also, there is a general rule of thumb that usually every hypothesis is used at least once in a proof.  Just using this rule alone could significantly improve proof search!)  See the next few sections for ideas on how to accomplish this.
* So far, when talking about the probability of a proof, we are considering the probability that a proof would appear if it was randomly generated using our probabilistic policy.  Of course, in a proof search, we may have to go through a number of proof attempts before reaching the right proof.  We could instead think of the amount of information in our proof as the amount of information needed to discover the proof through a proof search (probably with a penalty for long searches).  If our proof search was controlled by a learnable AI agent or could pass information when backtracking, that could provide a mechanism to compress the proof search instead of just the proof.  See the next few sections for ideas.
### Compute the main idea of a proof as a vector
It is commonly recognized that a lot of proofs contain a few big ideas and then a lot of follow-your-nose boilerplate.  Good expositions of proofs often front-load those big ideas.  Here is an idea how to encode the “main idea” of a proof into a vector which can be learned.

Imagine we have a generative model for a proof.  Now, let’s take that generative model and condition it on a vector which is computed from the top-level theorem statement.  If this whole system is differentiable, one can train this vector (and the effect it has on latter proof stages).  This hidden vector encodes the “main idea” of the proof in some sense.  Of course each subgoal of the proof can also have a hidden vector associated with it.

Notice, all we are describing with this “main idea” vector is a sequential model, like an RNN or Transformer, where each step of the proof depends on hidden variables associated with previous steps of the proof.

And as we commented above, by generating the proof with a single model, we can likely compress it more leading to a better model and a better tree search.  For example, maybe there are many repeated goals in this proof.  The “main idea vector” can specify how to handle them and pass that information to all of the children.  Or maybe the same premise is used multiple times in the proof.  Again the vector can indicate this. 

Thinking of Transformers, one could imagine filling in some parts of the proof tree and then have other unfilled parts attend not only to their parent nodes (as described above) but to previously filled in parts.  This again, would allow for communication between different parts of the proofs and likely increase compression.

Another approach to passing information is to update these hidden vector states in real time when backtracking.  (A long time ago, Deepmind wrote a paper about an adaptive form of MCTS where a hidden vector for each state is updated over the course of the algorithm.)  In this case, one could pass information between different nodes of a proof search not just a proof tree.
### An example
So let’s make this previous idea of using a sequence model more formal.  Imagine a search space with states —2, -1, 1, 0, and 1, 2.  We start at 0, can move “LEFT” or “RIGHT” and want to get to 2.  We first move left to -1, then move left to -2.  We decide to backtrack starting at 0 and then move right to 1, and right to 2.  

We train a Transformer or other sequence model on this rollout of the search with the following string.
```text
| 0 LEFT -1 | -1 LEFT -2 || 0 RIGHT 1 || 1 RIGHT 2
```
This helps the model learn the right policy and the action-state dynamics.  Here I’m using `|` to mean “this is what happened”, and `||` to mean “this is what should have happened”.  (If one can mask out the loss of various tokens, that is another approach.)  Additional training can also be used to supply values.

Now when one is in the middle of search, they use the results of the search so far as the prompt, e.g.
```text
| 0 LEFT -1 | -1 LEFT -2 ||
```
The sequence model sees the `||` and it tries to suggest the best next state and action to try.  (Again, we can also add in values to this training data and the model will return those as well.)

(However, it should be noted, that if one keeps around other branches of the search, then one risks violating some of the independence assumptions that go into say MCTS algorithms, and that needs to be taken into account when designing the search.  Of course, one could just have the sequence model control the entire search.)

### Compute a discrete strategy for a proof
Having a “main idea” of a proof seems nice in theory, but it also flies in the face of the accepted wisdom that for many theorems there isn’t a canonical way to go about proving a theorem.  Many times a lot can be learned from multiple proofs as long as they are sufficiently different.  We want to capture that fundamental choice in a vector.

Instead of (or in addition to) a continuous “main idea” vector, one could have a discrete vector (or vector) which encode different strategies.  There are a number of possible approaches, but here are two flavors:
1. At the top-level of the proof (or at each subgoal) the agent must make action where they pick a number 1 to N.  This action doesn’t directly correspond to a proof step, but instead is just a signal that it uses for later proof steps.  This action is directed by the agent’s policy algorithm just like other proof steps.  (See papers on discrete representations in reinforcement learning, like [Dreamer v2](https://arxiv.org/abs/2010.02193))
2. The same as above, but the agent is forced to try strategy 1 first until a certain point in the proof step, and then strategy 2, etc.  I haven’t worked this out as much, but the idea is that this would encode a deterministic proof finding algorithm.

It would probably still make sense to have a “main idea vector” (which again in nothing more than a hidden state) computed after this strategy signal to specify how that strategy should be executed given the current goal.
 
Now one of the big advantages of both of these approaches is that the agent can be even more opinionated, conditioned on the above strategy signal.  For example, one of the strategies might signal just a straight-forward-follow-your-nose-using-the-definitions sort of proof.  Once that is indicated, then the agent will have high confidence on every further step after that.  So high of confidence hopefully, that it should be easy to either (1) follow ones nose to the goal even if it is deep, or (2) give up this strategy entirely.  Another strategy might specify a totally different approach like applying a single lemma from the library to finish the proof.  Another might specify that one needs to perform a chain of rewrites, etc.  Unlike where it picks one tactic at a time, this strategy is global.  (Yes, some tactics do hint at global strategies, but not always.  It is better to have that mechanism build into the agent.)
### Compute the algorithm of a proof
Sometimes a proof can be described by an algorithm and this would be the ultimate way to compress a proof (indeed this is the point of Kolmogorov complexity).  Further, this is exactly what tactics are used for in tactic-based theorem provers.  It is not out of the realm of possibility that in the future AI agents will just discover tactics instead of individual proofs.  They will build up a library of tactics.  This in many ways is the dream of neuro-symbolic AI, and likely without something like this, we won’t be able to achieve our wildest dreams.
### Turn search into something adversarial (and self-supervised)
Often a proof search gets into a state which is not only difficult to prove, but provably false.  To my knowledge, no algorithm currently tries to actually show a proof subgoal is false, but one could.  It is just another proof to solve.

Treat this like a two player game.  Alice picks a proof step which generates zero, one, or multiple subgoals.  If that proof step succeeds, Alice gets a point.  Otherwise, Bob picks one of those subgoals which seems most difficult to prove.  They both only consider this one goal.  In version 1 of this game, that is the game.  They continue like that until Alice completes solves a goal, or a timeout is reached.

In version 2 of the game, after Bob’s pick, Bob has a choice: (1) Let Alice continue picking tactics for that subgoal, or (2) try to prove the subgoal is false.  If option (2) is picked, then Alice and Bob switch roles.  Bob is now trying to prove the negation of the goal, and Alice is the adversary.  This continues until one of the players solves the goal in question getting a point, or there is a timeout in which they tie.  (I may need to think a bit more on how to make this zero sum in the right way.  We don’t want to encourage Alice to just do trivial rewrites of a goal to avoid making a false statement.)  (For this idea I am reminded of Daniel Huang’s [On Learning to Prove](https://arxiv.org/abs/1904.11099) which has similar ideas on turning theorem proving into a two player game for training AI, and might provide a better version than my game.)

Note, this game won’t prove the theorem, but it will provide a specific adversarial game which would train both a good policy and value function.  If the theorem is provable, then Alice has a winning strategy in both versions of the game.  Moreover, in version 2 of the game, if Alice makes any false goals, and Bob exploits that, then we get good data even if a proof isn’t found since we solved a different proof and the policy learns how not to get into false situations.

It would also provide a good way to self-supervise learn a lot of “common” mathematical knowledge that may be lacking in the theorem proving library.  Consider “obvious facts” like  `x` is not less than itself, not all groups are commutative, not all functions are linear, etc.  It would also very quickly solidify general rules of thumb, such as proofs of a theorem typically use all the hypotheses at least once.  Moreover, it would solidify the theorem prover with number sense and sense about other common examples, since the most common counterexamples are just plugging in obvious examples like 0, 1, the identity function, and so on.

While I think ideas like this have been floated around for a while, with language models it is now possible to implement them in full.  The reason is that to prove the negation of a goal, one needs to come up with a number of counterexample terms.  This is difficult without a language model.

It would also be easier to do this in classical logic.  In constructive logic, many goals are neither provably true nor provably false.  Nonetheless, one can still have constructive counterexamples, so it is possible to do something similar, but that is more complicated.
### Systematic analysis of neural search
It makes sense to (1) do more analysis of search methodologies, and (2) create benchmarks for learning search.  

There are many papers on search methodologies (both neural and not) both in theorem proving and outside in other combinatorial, game playing, and RL settings. It would be good to organize that knowledge more.  For example, there are papers exploring search in mazes, Rubik’s cube, generating computer code, Montezuma’s Revenge, free form games, and simulated robots.   (Montezuma’s Revenge, for example, has a huge problem with lots of actions which lead to roughly the same states, and papers like [Go-Explore] provide interesting ideas which could be adapted here.)

One could also create simple benchmarks which are similar in character to theorem proving, but not as difficult to work with.  They could have many (semi-)symmetric actions, multiple ways to get to the similar (but not exactly the same) states, duplicate subgoals, etc.
### Rethinking proof search
We really shouldn’t assume that the order we are choosing actions of visiting subgoals is even close to optimal.  It is well known that search can significantly be improved by changing the objects being searched and how we divide up the search space.

See for example [Subgoal Search For Complex Reasoning Tasks](https://arxiv.org/abs/2108.11204) for one encouraging result.  I imagine there are a lot of possibilities here were we continuously search for intermediate and lemmas.  The Isabelle dataset would help to shed light on this situation I think since that has a very different paradigm of proof step and proof search.

In the end if we really want to solve big theorems, we have to make big jumps in reasoning, and this will involve radical changes in how we think of theorem proving as an AI activity.
 
Even if we keep with the current paradigm of using tactics to create subgoals, we still have a lot of choice in what order we process those subgoals.  If a proof creates multiple subgoals, does one prove them in order they are given, prove them using the most difficult first (since this will most likely cause the algorithm to backtrack), the easiest first, the medium difficult ones first?  (Currently my thought is that we should prove the most difficult first, but I’m willing to be persuaded to change my mind on this.)  Also, many tactic frameworks allow us to apply the same tactics too many subgoals at ones.
### Speeding up search steps
While the most important way to speed up a search is via better dividing up of the search space, where one has to visit fewer nodes (since it is often an exponential speedup), it should be noted that there are a few good ideas on how to significantly speed up the processing of each node as well.

Some of the best neural networks are Transformers.  GPT-f uses them, as do other theorem proving projects.  They offer a lot of potential, but have quadratic complexity (`O(N^2)` where `N` is the input plus output length).  I think in a few years, we will have good transformer alternatives which are closer to `O(N log N)`.  This will speed up proof step generation and also will allow longer sequence lengths making it possible to implement some of the ideas from above where we condition a proof step on the whole partial proof (or even on the whole global context of a proof).

The three most expensive operations in proof search are (1) encoding a proof state, (2) using that encoding to generate a proof step, (3) plugging that step into our theorem prover to get a new proof state.  
```text
  State -[encoder NN]-> StateEncoding -[NN]-> Value
    |                        |
+---+                        | [NN]
|                            v
| Action <-[decoder NN]- ActionEncoding
|   |
+---+
    | [theorem prover]
    v
  State2 -[encoder NN]-> StateEncoding2 -[NN]-> Value2
                             ...
```
All three can, in theory, be eliminated by directly approximating the step of reasoning in latent space.  The key is to use _world models_.  In a world model, one doesn’t need to constantly work with the theorem prover itself, but with a continuous vector representation of the proof state. There are a few good world model papers like MuZero.

The specific idea is as follows, instead of computing the proof step directly, we compute a discrete representation of our action, (for example a categorical value 1 through N or maybe a vector of categorical values).  At every point where we have to choose an action, we choose this discrete representation which deterministically (using the full context) determines the proof step.  In the long path, we use that discrete representation (plus the vector representation of the current proof state and the theorem proving library) to generate the full proof step, then plug that proof step into our model, and finally encode the new proof state into a vector representation.  But we also train a simple feed forward neural network (or similar) to compute the vector representation of the next proof state from the discrete action representation as well as the current proof state vector.  This bypasses all of the long steps and allows us to compute one step of reasoning in latent space entirely with fast models.  In this way, we can now perform reasoning in the world model several steps ahead without having to touch the theorem prover or generate complete tactic commands.
```text
State -[encoder NN]-> StateEncoding -[NN]-> Value
                           |
                  +--------+
                  |        |[NN]
                  |        v
                  |   ActionEncoding (Discrete)
                  |        |
                  +--------+ 
                           |[NN]
                           v
                      StateEncoding2 -[NN]-> Value2
                          ...
```

I think this would also go well with the ideas above about compressing proofs with vector representations or discrete representations.  Those discrete representations were for the whole proof, but it is a similar idea, and similarly this discrete representation of a proof step acts as another form of compression.

Actually, if this above idea can be made to work it could be really good.  The big left over piece of the MuZero paper was that in their setup, they hard codes certain things specific to chess.  They hard coded the shape of the action space, as well as gave their network an architectural inductive bias relevant to the chess board.  This makes it more difficult to adapt to novel scenarios.  But if all game states and moves were just encoded by plain text, then there is nothing special about chess here.  Any game or combinatorial problem could be put in place of chess.

For creating the categorial representation of the actions, see papers on discrete representations in reinforcement learning, like [Dreamer v2](https://arxiv.org/abs/2010.02193).  Also see Szegedy’s [Mathematical Reasoning in Latent Space](https://arxiv.org/abs/1909.11851) for a least one experiment showing that this can actually work in theorem proving.
