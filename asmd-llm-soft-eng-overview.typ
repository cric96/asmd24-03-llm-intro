#import "@preview/touying:0.6.1": *
#import "themes/theme.typ": *
#import "@preview/fontawesome:0.6.0": *
#import "@preview/ctheorems:1.1.3": *
#import "@preview/numbly:0.1.0": numbly
#import "utils.typ": *

// Pdfpc configuration
// typst query --root . ./example.typ --field value --one "<pdfpc-file>" > ./example.pdfpc
#let pdfpc-config = pdfpc.config(
    duration-minutes: 30,
    start-time: datetime(hour: 14, minute: 10, second: 0),
    end-time: datetime(hour: 14, minute: 40, second: 0),
    last-minutes: 5,
    note-font-size: 12,
    disable-markdown: false,
    default-transition: (
      type: "push",
      duration-seconds: 2,
      angle: ltr,
      alignment: "vertical",
      direction: "inward",
    ),
  )


#show: theme.with(
  aspect-ratio: "4-3",
  footer: self => self.info.author + ", " + self.info.institution + " - " + self.info.date,
  config-common(
    // handout: true,
    preamble: pdfpc-config, 
  ),
  config-info(
    title: [Large Language Models],
    subtitle: [An Overview for Software Engineers],
    author: [Gianluca Aguzzi],
    email: "gianluca.aguzzi@unibo.it",
    date: datetime.today().display("[day] [month repr:long] [year]"),
    institution: [Università di Bologna],
    // logo: emoji.school,
  ),
)

#set text(font: "Source Sans Pro", weight: "regular", size: 20pt)
#show math.equation: set text(font: "Fira Math")
#show strong: set text(weight: "bold", fill: rgb("#005587"))
#show emph: set text(style: "italic", fill: rgb("#00a3e0"))
#set underline(stroke: 1.5pt + rgb("#005587"), offset: 2pt)

#title-slide()

= Introduction 

// == Today Lesson in a Nutshell
// #align(center)[
//    #image("figures/meme.jpg", width: 60%)
//  ]


== Today Lesson
- *Goal:* Understand the fundamentals of Natural Language Processing (NLP) and Language Models (LM).

  - From a "practical" and "software engineering" perspective.
  - Understanding the basic concepts and the common tasks.
  - Different providing services (remote vs local).
  - How to use them from API and libraries.
  - How to "tune" them for specific (soft. eng.) tasks.
- *Note:*
  - We will not dive too much into the details of the algorithms and the mathematics behind them.
  - For this, please refer to the resources provided and the course on NLP.
- *Next:*
  - Vertical focus on the use of LLM in Software Engineering.
    - AI-assisted programming (e.g., code generation, code completion, etc.)
    - Vibe coding
    - Best practices for using LLMs in software engineering tasks.
  - And, also, the use of Software Engineering to build better AI-based app.
  - Research oriented directions)

#focus-slide[
  Ok, but #underline[why?]
]

== NLP & Software Engeneering -- Why BTW?
#align(center)[
  #image("figures/tweet-1.png")
]

== NLP & Software Engeneering -- Why BTW?
#align(center)[
  #image("figures/andreji.png", width: 80%)
]

== NLP & Software Engeneering -- Why BTW?
#align(center)[
  #image("figures/copilot.png", width: 40%)

]
#align(center)[
  #image("figures/copilot copy.png", width: 50%)
]

== NLP & Software Engeneering -- Why BTW?
#align(center)[
  #image("figures/soft-eng-improvements.png", width: 60%)
]

== Software 1.0 to Software 3.0

#text(size: 16pt)[
#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1em,
  [
    #align(center)[
      #underline[*Software 1.0*]
      #v(0.5em)
      _Rule-based (Code)_
    ]
  ],
  [
    #align(center)[
      #underline[*Software 2.0*]
      #v(0.5em)
      _Machine Learning (NN)_
    ]
  ],
  [
    #align(center)[
      #underline[*Software 3.0*]
      #v(0.5em)
      _Prompt Engineering (LLM)_
    ]
  ],
  [
    ```python
    def sentiment(text):
      good = ["great", "good"]
      bad = ["bad", "awful"]
      score = 0
      for w in text.split():
        if w in good: score += 1
        if w in bad: score -= 1
      return score > 0
    ```
  ],
  [
    ```python
    model = nn.Sequential(
      nn.Embedding(1000, 16),
      nn.LSTM(16, 32),
      nn.Linear(32, 1),
      nn.Sigmoid()
    )
    # Requires dataset
    # and training loop
    model.fit(X_train, y_train)
    ```
  ],
  [
    ```python
    prompt = f"""
    Classify sentiment:
    '{text}'
    Output Positive or Negative.
    """
    response = llm.generate(
      prompt=prompt
    )
    ```
  ]
)
]
== NLP & Soft. Eng. -- Why Should We Care?
- The Software Engineering landscape is *rapidly evolving*:
  - AI pair programmers (like Copilot) are becoming ubiquitous tools
  - LLMs can now handle tasks previously requiring human expertise:
    - Intelligent code completion
    - Automated documentation generation
    - Assisted refactoring and optimization
    - Test case generation
  - Key questions for modern developers:
    - What will be our role in this AI-augmented future?
    - How can we best leverage NLP to enhance our productivity?
    - Which skills remain uniquely human in software development?
  - Understanding this technology isn't optional—it's essential for staying relevant

= Natural Language Processing and (Large) Language Models
#focus-slide()[
#align(center)[
  #text(size: 28pt, weight: "bold")[Natural Language Processing (NLP)]
  
  #v(1em)
  
  #text(size: 20pt)[
    A subfield of artificial intelligence that focuses on #underline[_understanding_], #underline[_interpreting_], and #underline[_generating_] human language.
  ]
  #v(1em)
]
#align(left)[
  #text(size: 18pt, weight: "bold")[Resources]
  #set list(indent: 1em)
  #set text(font: "Source Sans Pro", weight: "regular", size: 18pt)
  - #link("https://github.com/keon/awesome-nlp?tab=readme-ov-file")
  - #link("https://github.com/brianspiering/awesome-dl4nlp")
  - #link("https://nlpprogress.com/")
  - #link("https://www.youtube.com/watch?v=LPZh9BOjkQs")
  ]
]


== Natural Language Processing

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    #underline[*Goal*]
    #v(0.5em)
    Identify the #underline[structure] and #underline[meaning] of _words_, _phases_, and _sentences_ in order to enable computers to #underline[understand] and #underline[generate] human language.
  ],
  [
    #underline[*Why?*]
    #v(0.5em)
    Improve _human-computer_ interaction, closing the gap between _human communication_ and _computer "understanding"_.
  ]
)

#v(1em)
#underline[*Applications (all around us)*]
#v(0.5em)
#grid(
  columns: 3,
  gutter: 1em,
  [
    - _Chatbots_
    - _Machine Translation_
    - _Speech Recognition_
  ],
  [
    - _Sentiment Analysis_
    - _Question Answering_
    - _Code Generation_
  ],
  [
    - _Image Captioning_
    - _Summarization_
    - _Text Classification_
  ]
)


== Natural Language Processing

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    #underline[*Challenges*]
    #v(0.5em)
    - *Ambiguity:* Multiple meanings for words/phrases.
    - *Context:* Meaning shifts with context (linguistic, cultural).
    - *Syntax:* Sentence structure affects meaning.
    - *Sarcasm/Idioms:* Non-literal language interpretation.
  ],
  [
    #underline[*Approaches*]
    #v(0.5em)
    - *Rule-Based:* Hand-crafted linguistic rules (e.g., #link("https://en.wikipedia.org/wiki/Georgetown-IBM_experiment")[Georgetown–IBM]).
    - *Statistical:* Probabilistic language modelling (e.g., hidden Markov model)#footnote("Mérialdo, B. (1994). Tagging English Text with a Probabilistic Model. Comput. Linguistics, 20(2), 155–171.")
    - *ML/Deep Learning:* Algorithms learn from data; neural networks model complex patterns (RNN#footnote("Yin, W., Kann, K., Yu, M., & Schütze, H. (2017). Comparative Study of CNN and RNN for Natural Language Processing. CoRR, abs/1702.01923. Retrieved from http://arxiv.org/abs/1702.01923"), LSTM#footnote("Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Comput., 9(8), 1735–1780. doi:10.1162/NECO.1997.9.8.1735"), GRU#footnote("Dey, R., & Salem, F. M. (2017). Gate-variants of Gated Recurrent Unit (GRU) neural networks. IEEE 60th International Midwest Symposium on Circuits and Systems, MWSCAS 2017, Boston, MA, USA")) 
    //- _Goal_: Find a *Language Model* that understands and generates human language.
  ]
)

#focus-slide()[
  #align(center)[
  #text(size: 28pt, weight: "bold")[What is a Language Model?]
  
  #v(1em)
  
  #text(size: 20pt)[
    A #underline[_machine learning_] model that aims to #underline[predict] and #underline[generate] *plausible* text.
  ]


  #align(
    center,
    block[
      #image("figures/llm-nutshell.png", width: 80%)
    ]
  )
  ]
]
== Language Models

#underline[*Fundamental Idea*]
#v(0.5em)
Text is a sequence of words, and language models learn the *probability distribution* of a word given the previous words in context.

#v(1em)
#underline[*Simple Example*]
#v(0.5em)
#align(center)[
  #text(size: 24pt)[_The software engineer was very happy with the <\*>_]
]

#pause

#align(center)[
  $arrow.b$
  
  #text(size: 20pt)[_The software engineer was very happy with the *coffee*._ (80%)]
  
  #pause
  #text(size: 20pt)[_The software engineer was very happy with the *unit-tests*._ (15%)]
  #pause
  #text(size: 20pt)[_The software engineer was very happy with the *codebase*._ (0.0001%)]
]

== Language Models -- Phases

*1. Tokenization:* split raw text into discrete, manageable units (tokens).
- _Example:_ "Unbelievable!" $arrow.r$ `["Un", "believ", "able", "!"]`

*2. Embedding:* map a fixed set of discrete tokens into dense numerical vectors (continuous space).
- _Example:_ `["Un"]` $arrow.r$ `[0.25, -0.75, 0.5, ..., 1.0]` (captures semantic meaning).

*3. Modelling:* learn the contextual relationships and probability distributions of tokens.
- _Example:_ $P("able" | "Un", "believ") = 0.95$

*4. Generation/Decoding:* sample from the predicted probabilities to produce the final output.
- _Example:_ Given $P("able")=0.95, P("ably")=0.01$, select "able".

#v(0.5em)
*Note:* Modern LLMs integrate these phases into a massive, end-to-end pipeline. However, these phases can also be used as standalone solutions (e.g., using just a Tokenizer to count tokens, or just an Embedding model for #underline[semantic] search).

== Tokenization

=== #underline[Tokenization: Breaking Text into Pieces]
#v(0.5em)
Splitting text into discrete subword units (tokens) for the model to process and generate.
#link("https://platform.openai.com/tokenizer")
#v(1em)

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    #underline[Example (BPE)]
    #v(0.5em)
    "Tokens are subwords!"

    #text(size: 16pt)[
      Split as: ["Tok", "ens", " are", " sub", "words", "!"]
      
      Mapped to IDs (e.g., GPT-5):
      - Tokens (30325)
      -  are (553)
      -  sub (1543)
      -  words (10020)
      - ! (0)
      - ...
    ]
  ],
  [
    #underline[In Practice (State-of-the-Art)]
    #v(0.5em)
    //- Standard algorithms: Byte-Pair Encoding (BPE), SentencePiece.
    - Modern vocabulary sizes up to ~250k (Gemini 3.1).
    - Used bidirectionally: encodes input prompts and decodes text during generation.
    - Unseen text is dynamically split into known subword fragments.
    - Special tokens manage flow: `<|begin_of_text|>`, `<|eot_id|>`.
  ]
)

== Embedding
=== #underline[Embedding: Converting Tokens to Contextual Vectors]
#v(0.5em)
Translating token IDs into dense numerical arrays that capture semantic meaning in context.
#v(1em)

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    #underline[Example (Context Matters)]
    #v(0.5em)
    The token "bank" gets a different vector based on surrounding text:

    #text(size: 16pt)[
      "River *bank*" -> [0.12, -0.85, 0.33, ...]

      "Secure *bank*" -> [-0.45, 0.22, 0.91, ...]
      
      Properties:
      - Deeply contextual (vector shifts based on the sentence).
      - Geometric distance = Semantic similarity.
    ]
  ],
  [
    #underline[In Practice (State-of-the-Art)]
    #v(0.5em)
    - Highly dimensional: typically 1024 to 12288+ dimensions per token.
    - Standard models: OpenAI `text-embedding-3`, open-source BGE or Nomic.
    - Computed dynamically on-the-fly inside modern Transformers.
    - Replaced static embeddings (Word2Vec) to capture nuance.
    - Serves as the continuous mathematical input for Self-Attention.
  ]
)

== Embedding -- Visual Example
#align(
  center,
  block[
    #image("figures/embedding-meaning.png", width: 100%)
  ]
)

== Modelling -- How?
#align(
  center,
  block[
    #image("figures/text-generation.png", width: 100%)
  ]
)


== Modelling -- How?
#align(
  center,
  block[
    #image("figures/cnn-text.png", width: 100%)
  ]
)

== Modelling -- CNN and RNN Limitations

#underline[*Limitations of Traditional Approaches*]
#v(0.5em)
- *RNN:* Long-term dependencies are hard to capture
- *RNN:* Slow to train; not suitable for large-scale data
- *CNN:* Fixed-size input window; not suitable for variable-length text
- *Both:* Struggle with large-scale parallelization
- *Solution:* _Multi-head self-attention_ — the core of _transformers_

#v(1em)
Transformers overcome these limitations by:
- Processing entire sequences in parallel
- Using attention to weigh token importance
- Capturing relationships across arbitrary distances
- Enabling efficient training on massive datasets

== Transformers -- Visual
#align(
  center,
  block[
    #image("figures/TransformerBasedTranslator.png", width: 100%)
  ]
)
== Transformers Architecture

#underline[*Transformers: State-of-the-Art for Language Models*]
#v(1em)

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    #underline[*Architecture Types*]
    #v(0.5em)
    - *Encoder-only:*
      - Creates embeddings from input text
      - Use: Classification, token prediction
      - Examples: BERT, RoBERTa
      
    - *Decoder-only:*
      - Generates new text based on context
      - Use: Continuations, chat responses
      - Examples: *GPT family, LLaMA*
  ],
  [
    #underline[*Full Transformers*]
    #v(0.5em)
    - Contains both encoder and decoder
    - *Encoder:* Processes input into intermediate representation
    - *Decoder:* Converts representation into output text
    - Example use case: Translation
      - English → Intermediate representation → French
    - Examples: *T5, BART, Marian MT*
  ]
)

== Transformers -- Self-attention

#underline[*Self-attention: The Key to Context Understanding*] -- #link("https://bbycroft.net/llm")
#v(1em)

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    #underline[*The Core Question*]
    #v(0.5em)
    For each token, self-attention asks:

    "How much does each other token affect the interpretation of this token?"
    
    - Attention weights determine token relationships
    - "Self" means within the same input sequence
    - Enables context-aware understanding
  ],
  [
    #underline[*Example*]
    #v(0.5em)
    "The animal didn't cross the street because it was too tired."
    
    - Each word pays attention to all others
    - Pronoun "it" is *ambiguous*
    - Self-attention reveals: "it" refers to "animal"
    - Resolves dependencies across *arbitrary distances*
  ]
)

== Self-attention -- Visual

#align(
  center,
  block[
    #image("figures/self-attention.png", width: 50%)
  ]  
)
- Self-attention calculates *weighted relationships* between every token
- These relationships reveal which parts of text should *influence* each token
- Attention weights are *learned parameters* during model training
- Multi-head attention allows model to focus on *different relationship types* simultaneously
- This mechanism captures both *local and long-range dependencies*

== Text Generation: From Probabilities to Text 

#underline[*Text Generation: Turning Model Output into Human-Readable Text*]
#v(1em)

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    #underline[*The Generation Process*]
    #v(0.5em)
    1.  Model receives a prompt or seed text (e.g., "The cat sat on the...")
    2.  Model predicts probabilities for the *next* token based self-attention encoding
    3.  A token is selected from this distribution based
    4.  Selected token is added to the sequence
    5.  Process repeats until stopping criterion is met
    
    *Key Idea:* Building a sequence one token at a time
  ],
  [
    #underline[*Decoding Strategies*]
    #v(0.5em)
    - *Greedy:* Always choose the highest probability token
    - *Random:* Sample from the probability distribution
    - *Top-k:* Sample from k most likely tokens
    - *Top-p/Nucleus:* Sample from the smallest set with probability > p
    - *Beam Search:* Track multiple candidate sequences
  ]
)

== Text Generation: Temperature

#underline[*Temperature: Controlling Randomness*]
#v(1em)

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    #underline[*How Temperature Works*]
    #v(0.5em)
    - Modifies probability distribution before sampling
    - Applied by dividing logits by temperature value
    - Softmax function then applied to get new probabilities
    
    *Formula (Simplified):*
    
    `probabilities = softmax(logits / temperature)`
  ],
  [
    #underline[*The Effect of Temperature*]
    #v(0.5em)
    - *High (≥ 1.0):* Flatter distribution, more random and creative
    - *Low (≈ 0.2):* Sharper distribution, more coherent but repetitive
    - *Zero:* Equivalent to greedy decoding
    
    *Analogy:* Temperature controls the "spice level" of the text
  ]
)

== Text Generation: Example

#underline[*Example: Generating Text with Different Temperatures*]
#v(1em)

#grid(
  columns: (1fr),
  gutter: 1em,
  [
    *Prompt:* The quick brown fox...
    
    *Temperature = 0.2:* The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox... (Repetitive)
    
    *Temperature = 0.7:* The quick brown fox jumps over the lazy dog. It was a sunny day, and the fox was happy.
    
    *Temperature = 1.2:* The quick brown fox dances with a sparkly unicorn under a rainbow made of cheese, giggling!
    
    _Note: These examples are illustrative. The actual output depends on the specific model._
  ]
)


== Large Language Model (LLM)
#focus-slide()[
  #align(center)[
    #text(size: 28pt, weight: "bold")[Large Language Model (LLM)]
    
    #v(1em)
    
    #text(size: 20pt)[
      A _language model_ with a _large_ number of parameters, trained on a _large_ corpus of text.
    ]
  ]
]
== LLM -- Implementation Strategies

- *Transformers* as the foundational architecture, characterized by:
  - Long-range context (_Attention_)
  - Efficient large-scale training (_Parallelization_)
  - Model growth (_Scalability_)
- *Pretraining:* Involves training the model on a vast corpus of text to learn a wide range of language patterns and structures.
- *Fine-tuning:* Refines the pretrained model for specific tasks, enhancing its applicability and performance on targeted applications.

== LLM -- Self-Supervised Phase

#underline[*Self-Supervised Learning: Learning directly from data without human annotations*]
#v(1em)

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    #underline[*What Makes It Self-Supervised?*]
    #v(0.5em)
    - Data creates its *own supervision signal*
    - No human annotations or labels needed
    - Model learns to predict parts of its input
    - Example: "The \_\_ of sleepy town weren't \_\_" #fa-arrow-right() "people, happy"
    - Leverages *natural structure* in language itself
  ],
  [
    #underline[*Advantages*]
    #v(0.5em)
    - Uses *unlimited* text data from the internet
    - Scales efficiently with more data and compute
    - Creates rich representations of language
    - Learns grammar, facts, reasoning, and more
    - Forms foundation for downstream adaptation
  ]
)

== LLM -- Training Paradigms

#underline[*The Learning Cake: An analogy to describe the layered approach in training methodologies.*]
#v(1em)

#grid(
  columns: (1fr),
  gutter: 1em,
  [
    - *Self-supervised Learning:* Models learn patterns from unlabelled data, reducing the need for expensive annotations. Ideal for initial _understanding_ of language structures.
    
    - *Supervised Learning:* Enhances accuracy with labeled data, crucial for tasks requiring specific outcomes like _classification_ and _translation_.
    
    - *Reinforcement Learning:* Adapts through trial and error using rewards, fine-tuning decision-making skills in scenarios like _dialogue generation_ (chatbots).
  ]
)

== LLM -- Paradigm Shift
#align(center)[
  #image("figures/llm-idea.jpg", width: 80%)
]
- LLMs are foundational for Modern NLP !!

== LLM -- Foundational Models (GenAI)
- A *Foundational Model* is a large model that serves as the basis for a wide range of downstream applications.
- It is related to several generative AI models (diffusion, transformers, etc.).
#align(center)[
  #image("figures/foundation.jpg", width: 60%)
]

== Difference with Traditional Models

#underline[*Traditional ML Pipeline vs. Foundation Model Approach*]
#v(1em)

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    #underline[*Traditional ML*]
    #v(0.5em)
    - Task-specific datasets
    - Models built for single purposes
    - Linear development pipeline
    - Requires retraining for new tasks
    - Limited transfer of knowledge
  ],
  [
    #underline[*Foundation Model Approach*]
    #v(0.5em)
    - General knowledge acquisition first
    - Adaptation to downstream tasks
    - Efficient knowledge transfer
    - Zero/few-shot capabilities
    - Single model, multiple applications
  ]
)

- Adaptation is a kind of "transfer learning" to other tasks
- With foundational LLMs, this adaptation may not require additional learning:
  - LLMs function as zero-shot or few-shot learners (more details later)
  - With just the right instructions (prompts), they can perform a wide range of tasks
- This represents a fundamental paradigm shift in AI and NLP development

== LLM -- Scalability
#align(center)[
  #image("figures/over-year.jpg", width: 100%)
]

== LLM -- Scalability
#align(center)[
  #image("figures/image-size.png", width: 100%)
]

== LLM -- Emergent Properties#footnote("Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., ... & Zhou, D. (2022). Emergent abilities of large language models. arXiv preprint arXiv:2206.07682.")
#align(center)[
  #grid(
    columns: 2,
    gutter: 1em,
    [#image("figures/small.jpg", width: 70%)],
    [#image("figures/medium.jpg", width: 100%)]
  )
  
  #v(1em)
  #image("figures/big.jpg", width: 80%)
]

== LLM -- State-of-the-art foundational models

- "Open" vs "Closed" models
  - Open -> are available for public use and research (you have access to the model and its parameters).
  - Closed -> are proprietary and not available for public use or research (you can just use the API).
- *GPT-\** (Closed): generative Pre-trained Transformer (OpenAI)
  - State-of-the-art in _language generation_ and _translation_.
  - GPT > 4 is multi-modal, capable of processing text, images, and audio.

- *Gemini* (Closed): most capable multi-modal model from Google.

- *Llama \** (Open): Large Language Model Meta AI (Meta) 
  - One of the first open source LLMs with a relevant number of parameters.

- *DeepSeek* (Open): a large-scale, open-source LLM with very low cost for training.
- *Mistral#footnote("jiang2023mistral")/Mixtral#footnote(link("https://huggingface.co/docs/transformers/model_doc/mixtral"))/Falcon#footnote("almazrouei2023falcon")* (Open): several completely open and transparent models from several companies.

#align(center)[
  More at: #link("https://lifearchitect.ai/models-table/")
]

== LLM Applications: Chatbots #footnote[#link("https://openai.com/blog/chatgpt")]
#align(center)[
  #image("figures/image.svg", width: 100%)
]

== LLM Applications: Medical Diagnosis #footnote[#link("https://sites.research.google/med-palm/")]
#align(center)[
  #image("figures/palm-med.png", width: 100%)
]

== Robotics #footnote[#link("https://deepmind.google/discover/blog/a-generalist-agent/")]
#align(center)[
  #image("figures/generalistic-agent.jpeg", width: 90%)
]

== LLM Concerns -- Training Cost
#align(center)[
  #image("figures/training-cost.jpg", width: 80%)
]

== LLM Concerns -- Privacy
#align(center)[
  #image("figures/italy-privacy-concern.png", width: 100%)
]

== LLM Concerns -- Hallucination
_Hallucination_ is the generation of text that is not grounded in reality.
#align(center)[
  #image("figures/hallucinations.png", width: 100%)
]

= LLM in Practice -- API and Prompt Engineering

== Interact with LLMs


- Via direct API: using the weights of the model.
  - [+] Full access to the model, it can be also fine-tuned
  - [-] Sometimes you do not have access to model weights (e.g., GPT-3)
  - [-] Sometimes even if the model is open, it is too _big_ to be used in a local environment (e.g., Falcon 180b)

- Via HTTP API: using a _web service_ that wraps the model.
  - OpenAI as reference#footnote(link("https://platform.openai.com/"))
  - [+] Easy to use
  - [+] Can be used in _any_ environment
  - [+] Can also be used with _local_ models (e.g., ollama)
  - [+] it supports both synchronous and asynchronous requests

== Ollama
#align(center)[
  #image("figures/ollama.png", width: 30%)
  
  #link("https://ollama.com/")
]

- A platform that wraps LLMs in a web service.
  - More than 60 models available.
  - Allow customizing the model.
  - Native performance for LLMs.
  - Support both embedding and generation tasks.
  - It is possible to set several parameters (e.g., temperature, top-k, etc.)

- How to use?
  - Pull a model: `ollama pul llama3.2`
  - Use the model: `ollama run llama3.2`
  - Start a web service: `ollama serve`


== LangChain
#align(center)[
  #image("figures/logo.png", width: 50%)
  
  #link("https://github.com/langchain-ai/langchain")
]

- A framework for developing applications powered by language models.
- Features:  
  - Support several API providers (e.g., OpenAI, Ollama, etc.)
  - Support the combination of several processing steps (e.g., prompting, chaining, etc.)
  - Support the context retrivial (e.g., RAG)
- In this course we will use the Java version of the framework.
  - #link("https://github.com/langchain4j/langchain4j")
#focus-slide()[
  == Demo
  #link("https://github.com/cric96/asmd-llm-code")
]

== LLM in Practice -- Prompt Engineering

- *Prompt Engineering* is the art of crafting the right instructions for the model to perform a specific task.

- *Prompts* are the input text that guides the model's behavior.
  - They can be as simple as a few words or as complex as a full paragraph.
  - They can be used to steer the model towards a specific task or style of output.
  - They can be used to provide context or constraints for the model's output.

- For an overview of the best practices in prompt engineering, see the #link("https://www.promptingguide.ai/")[Prompt Engineering Handbook].

== Prompt Engineering -- Zero Shot

#underline[*Zero-shot Learning allows a model to perform a task without any training examples.*]
#v(1em)

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    #underline[*Key Concepts*]
    #v(0.5em)
    - Model is given a prompt describing the task
    - Uses pre-existing knowledge from training
    - No examples or fine-tuning required
    - Works for tasks never explicitly taught
  ],
  [
    #underline[*Example*]
    #v(0.5em)
    *Intent:* Sentiment classification
    
    *Prompt:* 
    
    ``` Classify the sentiment of this review as positive, negative, or neutral: 'This restaurant had amazing food but terrible service.```
    
    *Output:* 
    ```
    Negative/Mixed
    ```
  ]
)

== Prompt Engineering -- Few Shot

#underline[*Few-shot Learning perform a task with just a few examples provided directly in the prompt.*]
#v(1em)

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    #underline[*Key Concepts*]
    #v(0.5em)
    - Model is given a prompt with a few labeled examples
    - Examples demonstrate the desired pattern or format
    - Examples act as implicit instructions for the task
    - Model learns to mimic the pattern in new cases
    - Significantly improves performance over zero-shot
    - Bridges general knowledge and specific task requirements
  ],
  [
    #underline[*Example*]
    #v(0.5em)
    *Intent:* Classifying emails
    
    *Prompt:* 
    ```
    Classify the following emails as SPAM or NOT SPAM:
    Email: "Congratulations! You've won $1,000,000 in our lottery!" -> SPAM
    Email: "Meeting scheduled for tomorrow at 2pm in the conference room." -> NOT SPAM
    Email: "Your account has been compromised."
    Classification: 
    ```
    
    *Output:* "SPAM" (model learns the pattern from examples)
  ]
)

== Prompt Engineering -- Chain of Thought

#underline[*Chain of Thought is a technique that involves chaining multiple prompts together to guide the model's output.*]
#v(1em)

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    #underline[*Key Concepts*]
    #v(0.5em)
    - Encourages model to show step-by-step reasoning
    - Explicitly asks model to "think aloud" before answering
    - Breaks complex problems into manageable steps
    - Dramatically improves accuracy for reasoning tasks
    - Reduces logical errors by exposing intermediate thinking
    - Particularly effective for math, logic, and multi-step problems
  ],
  [
    #underline[*Example*]
    #v(0.5em)
    *Intent:* Math word problem solving
    
    *Prompt:* 
    ```
    Solve this step-by-step:
        
        If John has 5 apples and gives 2 to Mary, then buys 3 more, 
        how many apples does John have now?
        ```
        
        #text(weight: "bold")[Output:] 
        ```
        Let me solve this step-by-step:
        1. John starts with 5 apples
        2. John gives 2 apples to Mary: 5 - 2 = 3 apples
        3. John buys 3 more apples: 3 + 3 = 6 apples
        John now has 6 apples.
        ```
      ]
    )
  
)

== Prompt Engineering -- Self-Consistency

#underline[*Self-Consistency is a technique that involves asking the model to generate multiple outputs and ensure they are consistent with each other.*]
#v(1em)

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    #underline[*Key Concepts*]
    #v(0.5em)
    - Generates multiple independent solutions to a problem
    - Cross-checks these outputs for logical consistency
    - Reduces contradictions and reasoning errors
    - Valuable for complex reasoning and math problems
    - Improves reliability through logical coherence
    - Often implemented via majority voting among solutions
  ],
  [
    #underline[*Example*]
    #v(0.5em)
    *Intent:* Reasoning with multiple paths
    
    *Prompt:* 
    ```
    Q: Today I have 6 apples. Tomorrow I buy 3 more.
    How many apples do I have?
    A: 9
    Q: Today I have 6 apples. Yesterday I ate 6 apples, How many apples
    do I have?
    A: 6
    Q: Today I have 6 apples. Tomorrow I buy 3 more. Yesterday I ate 6
    apples, How many apples do I have?
    ```
    *Output:* 
    ```
    Answers: 6, 6, 9 => 6
    ```
  ]
)

== Prompt Engineering -- Advanced Techniques

- *Retrivial Augmented Generation (RAG):* Allow to enrich the prompt with additional information from a knowledge base.
  - Used to provide context or constraints for the model's output.
  - It reduces `hallucination` and improves the quality of the generated text.
- *ReAct*: To generate both reasoning traces and task-specific actions in an interleaved manner. 
  - It is used to improve the reasoning capabilities of the model.
  - It is particularly effective for math, logic, and multi-step problems.
    - It use function calls to guide the model in the reasoning process.

- Some of these techniques are implemented in the LangChain framework.
- We will see more about these techniques in the next sessions

== Conclusion
- LLMs are *revolutionizing* the field of NLP and AI
- This has also a significant *impact on Software Engineering*
- Today: we have seen the *basics of LLMs* and how to interact with them
  - We have seen the importance of *prompt engineering*
  - We have seen how to *interact with them* through APIs
- Next Lesson:
  - Focus on the *use of LLMs in Software Engineering*
  - Divide by specific *tasks and applications*
- *Lab*:
  - *Hands-on* with LLMs
  - Use of *LangChain and Ollama*
  - *Prompt Engineering* practice