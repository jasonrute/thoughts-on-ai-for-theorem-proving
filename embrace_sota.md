## Embracing new SoTA trends in AI
#draft/aitp_thoughts
- - - -
I think there are two views when it comes to new technologies.  One is that new technologies are untested and have a shiny factor that eventually doesn’t live up to their promise.  One rarely hears about neural Turing machines any more, and while Transformers are all the rage even for image processing, CNNs are much more battle tested.  For many applications, logistic regression is fast and easy to implement and does just as well as the big hitters.

Further in an area like theorem proving, where inference speed matters a lot, older simpler algorithms like kNN and Boosted trees are orders of magnitude faster.  Even faster are hand-coded algorithms.  The ATP field has been developed over decades and one shouldn’t expect a giant black box to replace it anytime soon, especially in terms of efficiency for the sorts of problems these algorithms are specifically designed to do.

On the other side of that coin, certain newer technologies are showing promise over and over again in a wide variety of tasks.  Transformers, while they have their obvious O(n^2) disadvantages, they continue to show incredible potential.  A trend of the newer models is that one purposely doesn’t hard code much inductive bias, instead replacing hand-crafted knowledge with learned knowledge.  Unlike a graph neural network, a transformer doesn’t need to be told the graph structure of a formula.  Moreover, it also learns less obvious things like the complicated interrelations between variables, functions, etc.

The other big trend is that all pipelines are differentiable when possible, which allows features to be learned instead of hard coded.  One doesn’t have to manually convert a formula to a vector, loosing information along the way.  A language model neural network works directly on the formula,  automatically creating features through data.

A third big trend is being creative with gathering data.  It is not without doubt that deep learning requires large amounts of data, but the hottest ideas in deep learning keep showing that this isn’t a problem.  There are so many ways to get data.  Transfer learning shows that by taking pre-trained models on large datasets, one can very efficiently fine tune a model in a different setting using much less data.  Similarly, it has been shown (even in theorem proving) that training on tangentially related auxiliary data can improve a model as well.  Self-supervised learning has shown that there are many ways to train on raw un-labeled data by just training on self-guided tasks with that data.  (Theorem proving and other formal settings have a huge amount of potential here.)  Reinforcement learning, especially in grounded settings like theorem proving, basically provides an unlimited supply of training data.  Finally, large language models and multimodal models show it’s possible to basically just soak of knowledge from the internet and use that productively without any additional training on the task at hand.  We are starting to get closer to human-level ability to just pull knowledge from everywhere.

The fact that these same powerful techniques are so flexible and the same ideas are so ubiquitous, it makes it really easy (if not cheap) to deploy modern AI.   Working with raw text and differentiable models, makes pipelines easier to build and maintain.  Using large pre-trained Transformer models provide an economy of scale where all the data gathering and infrastructure work is done up front.  A HuggingFace transformer is as easily available (just as available as say a SQL database) for testing ideas and for large scale deployment.   This can greatly accelerate research and development.

The idea of those who champion neural networks and their newest incarnations are reminded of Deep Blue vs AlphaZero.  Deep Blue was the crowning achievement in AI of its time.  It was a massive brute force search built on extensive hand-crafted chess knowledge.  It worked and it was the prominent model of chess AI until AlphaZero came along.  AlphaZero showed that there was another paradigm, one which didn’t require hand crafted knowledge, but instead could learn from data alone.

Of course there are those which hypothesis that modern AI is in as much danger of reaching its local maximum as first wave AI was.  We obviously don’t want a black box differentiable tool when we can have an algorithm.  The algorithm is faster and better in every way.

The really question is, how do we accomplish this neuro-symbolic AI?  Some, like Daniel Selsam, hypothesis that we should build flexible programing frameworks that allow arbitrary choices.  These choices can be handled by AI agents (possibly a single powerful AI agent).  Others, like many in the current deep learning community, think that extensions of the current models can themselves learn to reason and code, providing solutions to problems.  Already Open AI Codex and similar models are doing this.  Other more specific tools like DreamCoder offer a mix of more tailored and principled approach to computer program creation.  

My opinion is that we are firmly in the second wave of AI.  While there are many areas where older AI systems excel, everything will soon get touched by deep learning, if not replaced.  It is important to benchmark and compare deep learning (which still may struggle) with simpler ideas which work well, but I believe (and I think I have reason to) that in a few years, a lot of current AIs will be outdone by neural based models.

Having said that, I think one should always be on the lookout for fusion of neural and symbolic ideas.  I think such fusions will naturally bring us into the third wave of AI.  I’m not claiming that will be human level cognition, but I think many neuro-symbolic ideas will flow out of current deep learning research.  Enough people are interested in the fusion even if they have vastly different views on how it will work.  Also, enough people are making concrete benchmarks test these ideas.  Maybe there will be a big ah-ha moment or maybe it will be more natural progression, but I don’t think we are going to reach an AI winter before having good neuro-symbolic tools.

In this regard, I think it is good to (cautiously) embrace new ideas backed up with good research.  Here are what I think are some of the most promising ideas.

### Advancements in RL
There have been many wonderful advancements in deep reinforcement learning.  Almost all of them are at least tangentially relevant to theorem proving and most haven’t been explored to their full extent.

One powerful idea is _world models_ in which one learns the action dynamics from data.  While this may at first seem silly in formal theorem proving since we already know the action dynamics (specifically how a proof state changes if you apply a proof step), it has been shown (for example in the MuZero paper) that world models significantly speed up searching through actions.  (I have some specific ideas on how to implement MuZero style world models further in here.)

The fundamental problem in realistic reinforcement scenarios is that it is very difficult to explore the state space efficiently.  There are many ideas here such as Go-Explore, curiosity methods, and so on.

Reinforcement learning is one of the best ways to gain knowledge, but even on-policy reinforcement learning requires continually retraining a model.  For small models this is fine, but as models get larger, this is more and more difficult.  It’s starting to be realized that one can replace a lot of gradient descent learning tasks with sequence models.  The hidden representations of the sequence model learn data in a way similar to gradient decent updating of weights.  This provides a lot of possibilities with fast on-policy RL which can even happen in real time.

RL can quickly get stuck if it is given a problem which is too hard.  One approach is to develop a curriculum where one starts with easier problems first.  The difficulty is creating curriculums automatically, and there are a lot of good ideas in this space.  One is to have two players and an adversary.  The adversary tries to come up with a problem that one player can solve and the other can’t.

A big part of the AlphaStar AI was population-based training where there are many agents which all specialize in different areas.  I haven’t seen as much of this sort of research lately (maybe it is too computationally expensive), but there are a lot of possibilities there.

### Transformers and Large language models
Transformers (as mentioned above) are quite popular and for good reason.  They work remarkably well at many sorts of text and sequence tasks.  There have been a handful of papers which have shown that Transformers can solve a number of combinatorial problems just through supervised learning.

[An aside.  I’m quite sure I could train an autoregressive decoder Transformer to completely solve sudoku puzzles.  I’m surprised no one has done this, but I’m 95% sure it is quite doable with today’s technology.]

_Large language models_ (or foundations models?) are all the rage now.  Most are built on top of large Transformer models.  While Transformers are the workhorse of the large language model, the true beauty is in the training.  By using a few well-designed self-supervised tasks (like next token prediction) and a huge, diverse dataset of Internet text, we begin to have language models whose output starts to look and sound like human text.



### Multimodal models
Combining images and text
Combining video and sound
Language models already are multi-modal
### Long sequence models
The largest problem with sequence models like Transformers is that they have a very limited context length (either due to memory or time complexity).  The tricks we can do with large language models work great, but aren’t that scalable to really long sequences.

Consider a simple example from the Lean theorem prover.  Suppose we want to build a tactic proof of `∀ (m n : ℕ), foo m n = foo n m`.
There are two simple ideas on how to use language models.  The first is to have a language model construct the proof outwrite.  With a prompt
```lean
example : ∀ (m n : ℕ), foo m n = foo n m := begin
```
Our model fills starts to fill in a proof like
```lean
  intro m n,
  induction m,
  induction n,
  refl,
  ...
```
This would be great, but it is asking a lot of the model.  Most humans use intermediate information provided by Lean, like the subgoals.

The other current way is to just treat each tactic step as a language modeling problem.  Given the prompt, 
```lean
<GOAL>
m n : ℕ
⊢ foo m n = foo n m
<TACTIC>
```
The model outputs a tactic `induction m`.  This can then be used as part of a larger tree search.  However, we are missing out on a lot of context here.  Even if it isn’t logically necessary, it is helpful to know where a proof has already been.  Instead, we could provide the prompt:
```lean
example : ∀ (m n : ℕ), foo m n = foo n m := begin
<GOAL> ⊢ ∀ (m n : ℕ), foo m n = foo n m </GOAL>
intro m n,
<GOAL> m n : ℕ  ⊢ foo m n = foo n m </GOAL>
```
This would give more information.

Moreover, to really know what to do here, it is helpful to know some information about `foo` such as its type or definition.  Lean obviously knows this information since it is part of the global context.  Why shouldn’t it also be provided to our model either as information in front, or as metadata about the tokens in the prompt.  Following this sort of logic, we will quickly blow up our context size as we add in more and more data to the prompt.

However, really long language models are to the rescue!  The only problem is that they don’t yet exist.   But I have every confidence that they will very very soon.

Here are two candidates that I’ve found in ICLR papers:  The first, is called S3 and it is basically an RNN, with O(L) or O(L log L) runtime (where L is the length of the sequence), and O(1) memory (when used in RNN form, it also has a convolutional form).  But unlike other RNNs, it is really good at memorizing long strings.  One could run it linearly over the global context up to the theorem one wishes to prove (forgetting that import structures of math libraries are not linear, but DAGs).  If it is that good, it would have actually picked up a lot of information along the way.  Indeed, a really good long sequence model shouldn’t need to be retrained (fine-tuned) on new data.  Instead it will just “learn” from that data by updating its internal hidden vector representation.  Since we know that fine-tuning a neural network stores information efficiently in the network weights, it isn’t surprising that this would also be true of the hidden weights of an RNN if we can find the right RNN variant.

The second candidate is the Memorizing Transformer.  It is like a Transformer, but it stores everything it has seen in a kNN external memory.  It doesn’t store the raw data per say, but efficient vector representations of the data.  It has an attention layer like a Transformer, but the query not only can look up information in the local context, but that same query also searches over data in the kNN external memory.  This means that in a proof search, the entire global context could be available for efficient querying.  Again, the idea is that one doesn’t have to retrain a model to learn data.  One learns by just running the model over the data.

Really, everywhere that models learn from data, we may now train powerful neural-network based models by “memorizing” the data in an efficient way.  This especially includes reinforcement learning, which historically has required (expensive) retraining on every step.  We can now have models which learn from their mistakes instantly, that memorize everything they experience, and everything they have tried so far.  We can have models that build up knowledge over time (such as proving their own theorems), and which adapt to new changing situations (meta-learning).  Imagine a chat bot which remembers your conversation and adapts in real time to you without any complicated ad-hoc engineering.  Or imagine an agent which scans over your code base and then gives you advice on coding, using your specific in-house libraries.  Or imagine an agent which is exploring mathematics, learning, and growing in its knowledge, developing its own personal library of facts.

Now, in fairness, we already have such models.  kNN-based systems have been around for a long time, and there are decent kNN-based theorem provers, like Tactician.  But actually mixing memorization and neural networks more seamlessly would provide these tools with a lot more power and flexibility.
### Self-supervised learning
