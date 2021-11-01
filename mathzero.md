# MathZero: What does “zero” mean?
#draft/aitp_thoughts
- - - -
A number of people have proposed a project which I will call “MathZero”.  The inspiration is AlphaZero, a Go and Chess playing AI that learned to play the game from just the rules of the game.  A follow-up work MuZero, learned to play from just demonstrations of the game.

This has inspired the questions, “Can we have an AI agent which rediscovers mathematics all by itself?”  “What would this alien mathematics look like?”

Again, with a lot of these questions it is easy to be vague, but what does it mean to re-discover mathematics, and what does “all by itself” mean?  I have no doubts that we can have agents eventually learn new and interesting mathematics that go beyond current human knowledge.  However, when it comes to re-learning mathematics, we quickly run into issues of either having nothing concrete to grasp onto, or of leaking our current way of thinking.

Let’s say we have the goal of rediscovering that there are primes and that the primes are infinite.  What would motivate an agent to learn this proof, and how can one provide that motivation without also providing subtle hints along the way to the answer.  I’m afraid this becomes quite challenging.

Humans discovered mathematics not in a vacuum but as a tool to deal with some of the most fundamental aspects of human life:
* Counting people and things.  A fundamental reality of the world we live in is that stuff by and large comes in discrete units that can be seperately moved and manipulated including people, animals, stones, apples, etc.
* Dividing up money and property (say to children).  Whole number divisibility quickly comes into play here when dividing up indivisible objects like animals.  Geometric length and area also comes into play.
* Understanding the physics of the world around us.  Simple questions about limits, conservation laws (and hence equations), area all come into play.

Our human experience gives us so much intuition about mathematics.  Of course addition is commutative when you think about adding money, and multiplication is commutative when you think about area.  Of course, if you divide up a shape into two, the areas sum to the original.

On way to get an agent to learn mathematics from scratch is to give it human experiences.  The problem here is two fold.  We can give an agent a vast collection of human data, but surely there is already mathematics inside that data.  Or we can give an agent a simulator which simulates economic transactions, physical laws, geometric constructions, and so on but again we are explicitly choosing mathematical knowledge to put into that simulation (with possibly a desired end goal of certain theorems we want the agent to learn).

A different way to start with minimal knowledge is to start with just the cold axioms of some formal system.  The agent starts to prove basic theorems and slowly builds up knowledge.  But without any concrete goals or motivation, I doubt an agent could discover anything of value.

### Mathematical games
