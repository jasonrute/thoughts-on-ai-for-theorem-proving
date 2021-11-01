## Fast vs. slow (online vs. offline) learning
#draft/aitp_thoughts
- - - -
Humans process information at vastly different rates
* Our reflexes respond before the signal even gets to our brains.
* We can complete a sentence or recognize an object without any thought.
* We can store a list of objects in our short term memory.
* We can remember things long term
* Important information we remember our whole lifetime (and dreams seem to be an important mechanism here for preserving this information).
* Human knowledge is built over generations, slowing being refined.
* Our brains come with certain instincts built-in via millennia of evolution.

In the same way, information can come into a learned model at different speeds.  
* A neural network or machine learning model can respond to a short input instantly.
* Sequence models can take in very large inputs with a lot of information, and hold that information in memory long enough act on it.
* kNNs and other retrieval mechanisms directly store data which can be quickly retrieved through various mechanisms.
* Small neural networks can be trained really quickly, even in real time.
* Pre-trained medium and large neural networks can be quickly fine-tuned (seconds, minutes, hours, days depending on the size of the network and size of the data).
* Medium size neural networks can be trained in minutes, hours, or days depending on their size and the amount of training data, but they can be  hour or day.
* Large neural networks hold a huge amount of information but require many resources to train from scratch.

Different aspects of fast verse slow:
* Permanent vs. forgettable
* Old knowledge vs new knowledge
* Fixed vs adaptable
* Development vs production environments
