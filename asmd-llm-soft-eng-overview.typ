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
#show quote.where(block: true): it => block(
  fill: rgb("#f4f8fa"),
  inset: 1em,
  radius: 0.2em,
  stroke: (left: 4pt + rgb("#005587")),
  text(style: "italic", it)
)

#title-slide()

= Introduction 

// == Today Lesson in a Nutshell
// #align(center)[
//    #image("figures/meme.jpg", width: 60%)
//  ]


== Today Lesson
- *Goal:* Understand the fundamentals of Natural Language Processing (NLP) and Language Models (LM).

  - Adopting a #underline[practical] and #underline[Software Engineering] perspective.
  - Understanding basic concepts, architectures, and common tasks.
  - Different providing services (_remote_ vs _local_).
  - How to use them from API and libraries.
  - How to "tune" them for specific tasks.
- *Note:*
  - We will not dive too much into the details of the algorithms and the mathematics behind them.
    - For this, please refer to the resources provided and the course on NLP.
- *Next:*
  - Vertical focus on the use of LLM (and Generative AI) in Software Engineering.
    - #underline[AI-assisted programming] (e.g., code generation, code completion, etc.)
    - #underline[Vibe coding]
    - Best practices for using LLMs in software engineering tasks.
  - And, also, the use of Software Engineering to build better AI-based app.
  - Research oriented directions (e.g., multi-agent communication).

#focus-slide[
  #text(size: 40pt, weight: "bold")[
    Ok, but #underline[why?]
  ]
]
== NLP & Software Engeneering -- Why BTW?
#align(center)[
  #image("figures/unicorns.png")
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
#quote(block: true)[
  AI is becoming a standard in developers’ lives: 85% of developers regularly use AI tools for coding and development, and 62% rely on at least one AI coding assistant, agent, or code editor.#footnote(link("https://blog.jetbrains.com/research/2025/10/state-of-developer-ecosystem-2025/"))
]
#align(center)[
  #image("figures/dev-ai-society.png", width: 60%)
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
  - AI #underline[pair programmers] (like Copilot) are becoming *ubiquitous* tools
  - LLMs can now handle tasks previously requiring human expertise:
    - Smart _code completion_
    - Automated _documentation generation_
    - Assisted refactoring and optimization
    - Test case generation
  - Key questions for modern developers?:
    - What will be our *role* in this AI-augmented future? 
    - How can we best *leverage* NLP to enhance our productivity?
    - Which skills remain #underline[_uniquely_ human] in software development? 
    - _Spoiler_: I do not have the answers, unfortunately #emoji.face.sweat
  - Understanding this technology isn't optional—*it's essential for staying relevant*

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
    Improve _human-computer_ interaction, closing the gap between _human communication_ and _computer #underline["understanding"]_.
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

  #v(0.5em)

  #text(size: 18pt)[
    When a language model is used to _generate_ new content (text, code, images, ...), it is called a *Generative AI* model.
  ]

  #align(
    center,
    block[
      #image("figures/llm-nutshell.png", width: 80%)
    ]
  )
  ]
]

== On Generative AI
- *Discriminative AI*:
  - Models that #underline[classify] or #underline[predict] data (e.g., Regression, Classification).
  - Focus on the boundary between classes (e.g., _"Is this a cat?"_).

- *Generative AI*:
  - A #underline[paradigm shift] enabled by Deep Learning.
  - Models designed to *create* #underline[new content] (text, images, audio, code).
  - Focus on the structure of the data itself to generate new instances (e.g., _"Generate a cat."_).

#image("figures/genai.png")
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

#align(center)[
  #let phase-box(title, desc, example) = block(
    fill: rgb("#f4f8fa"),
    stroke: 1pt + rgb("#005587"),
    radius: 0.4em,
    inset: 0.4em,
    width: 100%,
    align(left)[
      #text(size: 17pt)[*#title*] \
      #text(size: 14pt)[#desc] \
      #v(0.5em)
      #text(size: 12pt, fill: rgb("#555555"))[_Ex:_ #example]
    ]
  )

  #grid(
    columns: (1fr, auto, 1fr),
    rows: (auto, auto, auto),
    gutter: 0.4em,
    align: center + horizon,
    phase-box(
      "1. Tokenization", 
      "Split raw text into discrete units.", 
      ["Unbelievable!" $arrow.r$ `["Un", "believ", "able", "!"]`]
    ),
    text(fill: rgb("#005587"), size: 1.2em)[#fa-arrow-right()],
    phase-box(
      "2. Word Embedding", 
      "Map tokens into dense numerical vectors.", 
      [`["Un"]` $arrow.r$ `[0.25, -0.75, 0.5, ..., 1.0]`]
    ),
    
    [], [], text(fill: rgb("#005587"), size: 1.2em)[#fa-arrow-down()],

    phase-box(
      "4. Generation", 
      "Sample from probabilities to produce output.", 
      [Given $P("able")=0.95$, select "able".]
    ),
    text(fill: rgb("#005587"), size: 1.2em)[#fa-arrow-left()],
    phase-box(
      "3. Modelling", 
      "Learn contextual relationships and probabilities.", 
      [$P("able" | "Un", "believ") = 0.95$]
    )
  )
]

#text(size: 17pt)[
  *Note:* Here we use tokens and words interchangeably for illustration purposes. In practice, #underline[tokens] are the actual units processed by the model.
]

#text(size: 17pt)[
  *Note:* Modern LLMs integrate these phases into a massive, end-to-end pipeline (e.g., transformers) that learns all components jointly during training.
]

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

== Word Embedding
=== #underline[Word Embedding: Converting Tokens to Semantic Vectors]
#v(0.5em)
Translating token IDs into *dense numerical arrays* that capture *semantic meaning* in context.
#v(1em)

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    #underline[Example]
    #v(0.5em)
    Dog, Cat, and Car might be represented as:
    #text(size: 16pt)[
      - *Dog*: `[0.25, -0.75, 0.5, ..., 1.0]`
      - *Cat*: `[0.30, -0.70, 0.45, ..., 0.95]`
      - *Car*: `[-0.10, 0.20, -0.30, ..., 0.50]`
      
      If the model understands that *Dog* and *Cat* are *more similar* to each other than to *Car*, the vectors will reflect that (e.g., *smaller distance* between Dog and Cat).
    ]
  ],
  [
    #underline[In Practice]
    #v(0.5em)
    - Modern embeddings typically are computed in the *same model architecture* (e.g., inside the *transformer layers*).
    - In the past, they were often *pre-trained separately* (e.g., Word2Vec, GloVe).
    - *GPT-2* uses *768 dimensions*, while *DeepSeek V3* has *7,168*.
  ]
)



== Modelling -- Approaches

#underline[*Several approaches to model language:*]
#v(1em)

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1em,
  [
    #underline[CNN]
    #v(0.5em)
    - Fixed-size sliding windows over text
    - Good at capturing _local patterns_
    - Limited by fixed receptive field
  ],
  [
    #underline[RNN / LSTM / GRU]
    #v(0.5em)
    - Process tokens _sequentially_
    - Can capture order and context
    - Struggle with _long-range_ dependencies
  ],
  [
    #underline[*Self-Attention*]
    #v(1.5em)
    - Each token attends to _all_ others
    - Captures _arbitrary-distance_ relationships
    - Fully _parallelizable_
  ],
)

#align(center)[
  #image("figures/cnn-text.png", width: 80%)
]

== Modelling -- Self-Attention
  //#underline[*Self-attention: The Key to Context Understanding*]
#v(0.5em)

#grid(
  columns: (1fr, 1.2fr),
  gutter: 1.5em,
  align: horizon,
  align(center)[#image("figures/self-attention.png", width: 95%)],
  [
    #underline[*The Core Question*]
    #v(0.3em)
    For each token: _"How much does each other token affect its interpretation?"_
    - Attention weights determine token relationships
    - Captures *arbitrary-distance* dependencies
    #v(0.5em)
    #underline[*Example*]
    #v(0.3em)
    "The animal didn't cross the street because *it* was too tired."
    - "it" is *ambiguous* — self-attention resolves it to "animal"
    - Multi-head attention captures *different relationship types* simultaneously
  ]
)


== Self-attention -- Mechanics

#text(size: 18pt)[
  *Input:* A sequence of token embeddings (vectors).

  *Output:* A sequence of *context-aware* vectors. Each vector is a mixture of information from other tokens, weighted by their relevance (e.g., resolving "it" to "animal").

  *How it works (conceptually):*
  - *Roles:* Each token is projected into three views:
    - *Query:* What information the token is looking for.
    - *Key:* What information the token contains.
    - *Value:* The actual content to be passed along.
  - *Attention:* Compute similarity between Queries and Keys to find relevance.
  - *Aggregation:* Compute the weighted sum of Values based on attention scores.
]
== Modelling -- Self-Attention
#image("figures/more-self-attention.png")



== Transformers -- Visual

- *Transformers* are the dominant architecture for LLMs.
- They merge *word embedding* and *modelling* into a single end-to-end system.
- The architecture relies on layers of *multi-head self-attention* and *feedforward networks*:
  - *Multi-head attention* captures various relationships (e.g., syntax, semantics) simultaneously.
  - *Feedforward layers* introduce non-linearity for complex pattern learning.
- It is highly *parallelizable*, allowing training on massive datasets.
- The output is a vector for each token, which results in a probability distribution to generate the next token in the sequence.
#image("figures/end-to-end.png", width: 100%)

#link("https://poloclub.github.io/transformer-explainer/")

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

== Contextual Embedding
- If you remove the last layer of a transformer, you get a *contextual embedding* for each token.
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
    - *Highly* dimensional: typically 1024 to 12288+ dimensions per token.
    - Standard models: OpenAI `text-embedding-3`, open-source BGE or Nomic.
    
  ]
)

#align(center)[ 
#link("https://dashboard.cohere.com/playground/embed")
]
== Contextual Embedding -- Visual Example
#align(
  center,
  block[
    #image("figures/embedding-meaning.png", width: 100%)
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
#align(center)[
    #image("figures/overall-idea.png", width: 85%)
]
== LLM -- Self-Supervised Phase

#v(1em)

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    What Makes It *Self-Supervised?*
    #v(0.5em)
    - Data creates its *own supervision signal*
    - No human annotations or labels needed
    - Model learns to predict parts of its input
    - Example: "The people of sleepy town weren't \_\_" #fa-arrow-right() "happy"
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

== LLM -- Training Pipeline

Modern LLM training is typically organized as a *pipeline* involving three main phases:

#v(1em)

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1em,
  [
    #set align(center)
    #underline[*1. Pretraining*]
    #v(0.5em)
    #set align(left)
    - *Goal:* Learn general language patterns & world knowledge.
    - *Signal:* Next-token prediction on massive unlabeled corpora.
    - *Result:* fluent, but not instruction-following.
    #v(0.5em)
    #text(size: 16pt, style: "italic", fill: rgb("#555555"))[
      "What continuations are likely in text?"
    ]
  ],
  [
    #set align(center)
    #underline[*2. Instruction Tuning*] 
    #v(1em)
    #set align(left)
    - *Goal:* Follow instructions and specific tasks.
    - *Signal:* Labeled "prompt $arrow.r$ ideal response" pairs.
    - *Result:* *"Instruction-tuned"* model.
    #v(0.5em)
    #text(size: 16pt, style: "italic", fill: rgb("#555555"))[
      "What should I do when asked?"
    ]
  ],
  [
    #set align(center)
    #underline[*3. Alignment*] 
    #v(0.5em)
    #set align(left)
    - *Goal:* Match human preferences (safety, helpfulness).
    - *Signal:* Human preference comparisons (A vs B).
    - *Result:* safe & helpful assistant.
    #v(0.5em)
    #text(size: 16pt, style: "italic", fill: rgb("#555555"))[
      "What response is preferred/safe?"
    ]
  ]
)

#v(1em)
*Note:* Steps 2 and 3 are often iterated to fix regressions and target specific domains.

== LLM -- Foundational Models and Paradigm Shift
#align(center)[
  #image("figures/llm-idea.jpg", width: 80%)
]
- A *Foundational Model* is a large model that serves as the basis for a wide range of downstream applications.


== Traditional ML Pipeline vs. Foundation Model Approach

#v(1em)

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    #align(center)[*Traditional ML*]
    #v(0.5em)
    - Task-specific datasets
    - Models built for #underline[single] purposes
    - Linear development pipeline
    - Requires *retraining* for new tasks
    - Limited #underline[transfer of knowledge]
  ],
  [
    #align(center)[*Foundation Model Approach*]
    #v(0.5em)
    - *General knowledge* acquisition first
    - Adaptation to downstream tasks
    - Efficient #underline[knowledge transfer]
    - Zero/few-shot capabilities --- more later
  ]
)

- Adaptation is a kind of "transfer learning" to other tasks
- With foundational LLMs, this adaptation may not require additional learning:
  - LLMs function as *zero-shot* or *few-shot* learners (more details later)
  - With just the right instructions (*prompts*), they can perform a wide range of tasks
- This represents a *fundamental paradigm shift* in AI and NLP development

== LLM -- Scalability
#align(center)[
  #image("figures/model-over-time.png", width: 100%)
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


== LLM Applications in a Nutshell

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1em,
  align: top,
  [
    #align(center)[
      #image("figures/image.svg", height: 6em, fit: "contain")
      #v(0.5em)
      *Chatbots & Assistants*
    ]
    #v(0.5em)
    - General purpose interaction
    - Content generation
    - Coding assistants
  ],
  [
    #align(center)[
      #image("figures/palm-med.png", height: 6em, fit: "contain")
      #v(0.5em)
      *Specialized Domains*
    ]
    #v(0.5em)
    - Medical diagnosis (Med-PaLM)
    - Legal analysis
    - Scientific research
  ],
  [
    #align(center)[
      #image("figures/generalistic-agent.jpeg", height: 6em, fit: "contain")
      #v(0.5em)
      *Embodied AI*
    ]
    #v(0.5em)
    - Robotics control
    - Generalist agents (Gato)
    - Multi-modal interaction
  ]
)

== Critical Concerns and Limitations

#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 1em,
  align: top,
  [
    #align(center)[
      #image("figures/training-cost.jpg", height: 6em, fit: "contain")
      #v(0.5em)
      *Training Costs*
    ]
    #v(0.5em)
    - Massive compute requirements
    - Environmental impact
    - Centralization of power
  ],
  [
    #align(center)[
      #image("figures/italy-privacy-concern.png", height: 6em, fit: "contain")
      #v(0.5em)
      *Privacy & Legal*
    ]
    #v(0.5em)
    - Data leakage risks
    - Copyright infringement
    - GDPR compliance
  ],
  [
    #align(center)[
      #image("figures/hallucinations.png", height: 6em, fit: "contain")
      #v(0.5em)
      *Reliability*
    ]
    #v(0.5em)
    - #underline[Hallucinations:] Confidently stating false facts
    - Lack of grounding
    - Bias and fairness issues
  ]
)

= LLM in Practice -- API and Prompt Engineering

== LLM -- State-of-the-art foundational models

- "Open" vs "Closed" models
  - Open -> are available for public use and research.
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