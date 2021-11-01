## User stories: What is our goal, really?
#draft/aitp_thoughts
- - - -
What is the goal of AI theorem proving?  What things do we hope to accomplish in 1 year?  5 years?  Our lifetimes?  Without answers to these questions, we can easily talk past each other.  

One approach in software engineering and product design to answer such question is to make user stories outlining how a user would use a product from beginning to end.  Here are various user stories, roughly from most the ambitious projects to the most realistic.

(I’ve added other user stories for related topics like auto-formalization and rediscovery of mathematics in a different location.)

### User story: The AI Mathematician
Alice, a human, mathematician, consults with her AI counterpart.  This AI does basically everything a mathematician does.  It proves theorems, it discovers connections between different areas, it develops new areas of mathematics, complete with definitions and terminology.  It explains those areas to others (both human and AI) so that they can understand it better.  It is comfortable working in natural language and doesn’t need to write formal proofs.  It just gets things in the same way that a human mathematician does.

This seems far fetched, and in many ways it is.  I don’t see this happening anytime soon, but it is good to have this as inspiration.
### User story: The AI Formal Mathematician
Bob, is a computer scientists and mathematician who is comfortable with formal mathematics. He is working with an AI which is remarkably good at proving formal theorems.  This AI is built on top of a large body of formal mathematics.  The AI (which may not be one agent, but a population of agents) is continually compressing and learning from this formal library and working on theorems that Bob gives it to solve.  It is a collaborative process in which the agent forms conjectures, finds important lemmas, scans the literature (both formal and informal).  Right now, Bob is working on getting the agent to solve Big Theorem X.  The agent seems to be coming up with brilliant suggestions and other mathematicians are brought in to make sense of them and push the agent along.  Papers have already been published on results found by Bob and his AI.

While the complete picture here is likely AGI level and won’t happen anytime soon, there is already a slight glimmer of hope that maybe we can continually move in this direction.
### User story: Fill in the proofs over night
Carol is trying to formalize a well-known, but controversial theorem she proved.  She first tried to do it all by herself but it is just too much work.  Instead she switched to just giving an outline with the hope that others will fill in the details.  David on the other hand manages a database of formal abstracts, these are formal theorems complete with definitions, but often missing proofs.  They both partners with some ML researchers who have a powerful reinforcement learning agent.  That agent searches for formal proofs and whenever it finds any, it learns from the proofs it has found and uses those to find more proofs.  This agent isn’t particularly fast, it works over days, weeks, and moths.  After a quick start where it filled in a bunch of easy theorems, it is slowly building up both an internal database of little facts and useful knowledge, as well as slowly finding proofs of the target theorems (both from Carol’s outline and David’s formal abstracts).  It both fills in complete proofs from established formal facts, as well as partial proofs which prove a theorem from other facts in the library which haven’t been formally proven.  For the more challenging theorems, Carol and David fill in more details and outlines to make it easier for the agent.

I think we are almost there.  Various RL for theorem proving projects basically already do this.  The only challenge is scaling this up.
### User story: A smart theorem prover buddy
Ernie is a proof engineer.  He isn’t as experienced as he would like to be, and he is building proofs on top of a large codebase.  Built into some IDE or some language server is his AI buddy.  This buddy helps with all aspects of formal theorem proving: writing code, filling in boiler plate, completing term proofs, tactic proofs, finding useful lemmas.  It is like a modern IDE, but on steroids.  Basically anything that Ernie considers trivial this agent knows how to do.  This agent also knows his personal code base pretty well and can has a variety of search functions which helps Ernie find exactly the theorem he needs from the library.

I think many aspects of this are getting really close.  We have better and better AI proof search, and AI library search, and AI assisted programming tools.  The question is if we have enough interest to build this out for a single ITP, or if we can create a general system like OpenAI Codex which is language independent.
### User story: The Industrial strength theorem prover
Francesca works in industry and Gina works in a particular area of pure mathematics.  Both have discovered that they each have a particular class of problems for which they would like an automated theorem prover to assist them.  Each of these problems is sufficiently different that a general algorithm isn’t really possible, but also close enough related that they have personally developed a good intuition for solving these sorts of problems.  They have even taught their colleagues.  They collect a number of examples with proofs, and a larger number of examples without proofs.  They then train an AI agent to formally solve these problems.

Whether or not this is easy is going to depend a lot on the type of problems we are trying to solve.  Problems which are routine,  boiler-plate, and common and don’t involve much trickery (or brute force search) will fall pretty quickly with current methods.  Problems which are more diverse, require a lot of general mathematical background knowledge, or require creative tricks will be harder.

### What is holding us back?
In many of these cases (especially the less ambitious ones) the goals are pretty clear:  Given a formal system of mathematics, a library of formal theorems and definitions, and demonstrations of proofs, construct an agent which can solve formal theorems.   Many modern ML approaches, often inspired by AlphaGo and its descendants, show very promising results in this area.  This has happened time and time again.

However, there is still a sense in which we are hitting a brick wall:
1. These systems find only shallow proofs.  They don’t explore the space of proofs well, nor are they particularly trained well to do this.
2. These agents use a library of formal mathematics that serves humans well, but they are missing massive amounts of simple mathematical common-sense knowledge.
3. They don’t adapt (nor are they built to adapt) to new information, be that new definitions, or information which comes up during their running.

### What can we do better
Deep learning is entirely driven by data.  We have to find data and then exact whatever we can out of that data.  Having said that, one doesn’t have to necessarily go out and track down and label new data.  Raw data, self supervised data, and data gained through RL is just as valuable and often more.  There are two big tricks in AI:
1. Developing games to make an agent discover better data.  (This is through RL on self-experience data, self-supervised learning on big data and raw data, transfer learning from unrelated data, distillation / student / teacher learning on already learned data, and multimodal learning on different modalities of data.)
2. Using better models which are able to extract information from data.  (Transformers, for example, handle large amounts of data in short term memory, learn inductive biases from data, work with many modalities of data, and can learn unlabeled data easily through simple tasks.  Other, task specific models have relevant built-in inductive biases.  Long language models and other models which can adapt quickly to new information are desperately needed.)

### What can we do better specifically?
Let’s again explore those three areas where we were struggling and see how both better data and better models can help:

#### Search
For search, the challenge is to know what do do next and know if one is on the right track.  RL has a well-proven track record for improving search.  Small tweaks to RL can make a big difference here, by making it easier for the model to find good high-quality data for search.  Indeed much of the RL literature is about how to improve the data finding to have the model not get stuck in local situations.

Also for search, one does not want the model to be constrained in the sort of actions it can make.  This requires better models which are able to better use all available information and output generically structured data.  Language models excel at this.  Further, there is a lot of information that is applicable to search in theorem proving, information about the current state, the past actions.  By being able to better use this information, in say long sequence models, we can better compress the data we have available to us and make more informed decisions on longer moves.

New model designs also let us think about search fundamentally differently.  There is no reason that we can’t have an ML agent which handles every aspect of the search, such as what states to explore next.
#### Mathematical knowledge
For mathematical knowledge, the problem is really a lack of data.  First, there is a lot of general math knowledge out there on the Internet or in other theorem provers, and we shouldn’t be afraid to use it.  Moreover, there is a lot of low level mathematical information available in theorem provers to use.

But also, there is so much mathematical knowledge to be had just from reinforcement learning.  Imagine the simple case where an agent reaches a goal which is false.  First, it could learn this through RL and a value function.  But, it still wouldn’t be able to distinguish a goal which is false from one which is just hard to prove.  By devising a game, we could have an agent learn to find counterexamples to goals.  Not only would this game help a model find false goals, but it instills mathematical knowledge to the model.  This is just scratching the surface of what knowledge we can find through self-supervised learning.

Also, in RL, if a model finds a proof of some part of a theorem, that alone is a lemma that the agent can now prove.  Moreover, we could even try to add this lemma to the library to be used for future iterations.

As for models, language models make it easier to incorporate external mathematical knowledge, and also make it easier for the model to learn new types of mathematical knowledge.  For example, if we play the game where we try to show a goal is false, we usually need to find a counterexample, and language models make that easy.
#### Adaption
RL is certainly a form of adaptation, but it is in many ways it is too slow.  Another form of adaptation is to train a model on many related settings.  The model doesn’t specialize to any of them.

An important part of adaptation is being able to access new information.  Current neural networks are really bad at this, but new sequence models are coming down the line that will really help with this I think.  This could fundamentally change the way we think about what it means to “learn” or even to “train” an AI agent.
