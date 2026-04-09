#import "@preview/touying:0.7.1": *
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
  - First, understanding basic concepts, architectures, and common tasks.
  - Then going on how to use them in practice (e.g., APIs, libraries, etc.) and how to "tune" them for specific tasks.
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
  AI is becoming a standard in developers' lives: 85% of developers regularly use AI tools for coding and development, and 62% rely on at least one AI coding assistant, agent, or code editor.#footnote(link("https://blog.jetbrains.com/research/2025/10/state-of-developer-ecosystem-2025/"))
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
Text is a sequence of words (namely, a *prompt*), and language models learn the *probability distribution* of a word given the previous words in context.

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

== LLM -- Self-Supervised Phase
#image("figures/idea-for-training.png")

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
- With foundational LLMs, this adaptation may not require additional learning
  - The parameters are #underline[freezed], and the model is used as-is with just the right instructions.
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
    - Massive compute requirements (petabytes of data, exaflops of compute)
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

= LLM in Practice

== LLM — Models

- *Open weights*: model parameters (weights) are released; you can _run_ and _fine-tune_ locally, but training code and training data may remain #underline[undisclosed].
- *Open source*: model code (and often the full training pipeline) is released; you can _inspect/modify/retrain/redistribute_ under the license (weights and data may still differ in openness).
- *Open data*: training dataset (or a meaningful, reusable disclosure of it) is released; enables stronger _reproducibility/auditing_, but is comparatively #underline[rare] in practice.
- *Closed*: #underline[no] public code/weights/data; typically _API access only_ (vendor-controlled).

Key distinction: *open weights* ⇒ _mainly usage_; *open source* ⇒ _modifiability_; *open data* ⇒ _data transparency/reproducibility_.

== LLMs -- Example of each type

- *GPT-\** (#underline[Closed]): OpenAI's flagship series. Known for strong _language generation_ and _multi-modal_ capabilities.
  - OpenAI also offers *open-weight* variants (e.g., _GPT-4.1 nano_).
- *Llama \** (#underline[Open weights]): Meta's family of LLMs. Notable for being among the #underline[first] large-parameter models available for _public use_.
- *Gemini \** (#underline[Closed]): Google's flagship series, comparable to GPT.
  - *Gemma* is the _open-weights_ counterpart of Gemini.
- Several others: *Claude* (Anthropic, strong at _code_), *Mistral* (one of the few _European_ models), *DeepSeek*, *Qwen*, etc.
- They all share the same core idea -- #underline[sequence in, next token out] -- but differ in _architecture details_, _training data_, and _fine-tuning techniques_.

== LLM -- 'Providers'
- LLMs may have *billions* of parameters
  - Typically require _specialized hardware_ (GPUs)
  - Rule of thumb: *1B parameters* $approx$ #underline[double] the amount of *GPU memory* for inference
  - Consider for instance _DeepSeek-V3_, which has *~671B parameters* and requires at least #underline[800 GB] of GPU memory for inference (e.g., 8$times$ A100 80 GB).
- Therefore, it is common to use LLMs via a *provider* that offers access to the model through an #underline[API].
  - The provider is responsible for _hosting_ the model, _managing_ the infrastructure, and _exposing_ an interface for users to interact with it.
  - Access is typically provided through a *RESTful API*, allowing users to send requests and receive responses in a standardized format (e.g., JSON).
- Obviously, it is also possible to run LLMs *locally* (when the architecture and weights are available -- more details in a minute), but it is #underline[not always feasible] due to the computational requirements.

== LLM -- 'Providers' 
#image("figures/llm-provider.png")

== Cost Models and Access

#underline[*Pay-as-you-go model*] — Users charged based on API usage:
- Pricing based on _tokens processed_ (input + output) per million tokens
- Some providers offer subscription plans or enterprise agreements
- Typically, work with an API key that tracks usage and billing

#v(1em)
#underline[*Current Pricing Examples (March 2026):*]
- *OpenAI:* GPT-5 at \$1.25/\$10.00 per 1M tokens (input/output)
- *Anthropic:* Claude Sonnet 4.5 at \$3.00/\$15.00 per 1M tokens
- *Google:* Gemini 2.5 Flash at \$0.15/\$0.60 per 1M tokens

#v(1em)
#underline[*Cost Optimization Strategies:*]
- _Batch Processing:_ 50% discount for non-real-time workloads (24-hour turnaround)
- _Prompt Caching:_ Up to 90% savings on repeated content (e.g., Anthropic)
- _Combined savings:_ Batch API + caching can achieve 95% cost reduction

#v(1em)
#underline[*Access Tiers and Limits:*]
- _Free tiers_ available with limited usage for experimentation
- _Tiered rate limits_ (e.g., Google: 15 RPM free $arrow.r$ 1000+ RPM Tier 2)
- Higher tiers unlocked by spending thresholds (\$250+) or time

== LLMs -- Other Axes
- LLMs also vary along other axes:
  - *Size*: _small_ (less than 3B, #underline[local usage], _edge AI_) vs. _large_ (more than 10B, requires #underline[specialized hardware]).
  - *Modality*: _text-only_ vs. _multi-modal_ (e.g., text + images + audio).
  - *Specialization*: _general-purpose_ vs. _domain-specific_ (e.g., medical, legal, code).
- #underline[How to choose the right model?]
  - Consider the *task requirements* (e.g., do you need _multi-modal_ capabilities?).
  - Consider the *computational resources* available (e.g., can you run a _large model_ locally?).
  - Consider *privacy and security* needs (e.g., do you want to avoid sending data to an _external API_?).
  - Consider the *cost* of using a model via _API_ vs. running it _locally_.

#align(center)[
  More at: #link("https://lifearchitect.ai/models-table/")
]

== API Endpoints & Interaction Models

- #underline[*Core Mechanism:*] HTTP POST requests with a JSON payload (inputs + config like `temperature`).
- #underline[*Chat Completions:*] The modern standard. Uses an _array of structured messages_ (roles: `system`, `user`, `assistant`) to maintain dialogue context.
  - _Examples:_ `v1/chat/completions` (OpenAI), `v1/messages` (Anthropic).
- #underline[*Embeddings:*] Converts text into _high-dimensional vectors_. As input, they take a list of strings; as output, they return a list of vectors (one per input string).

#v(0.5em)
#underline[*Streaming Responses (Token-by-Token)*]
- *How it works:* Output is streamed incrementally via *Server-Sent Events (SSE)* (triggered by `"stream": true`).
- *Why it matters:* Drastically reduces #underline[*Time to First Token (TTFT)*], giving immediate visual feedback and preventing HTTP timeouts.


#focus-slide()[
  = Demo
  Google AI Studio
]

== Interacting with Local LLMs

While cloud APIs are undeniably convenient, running LLMs *locally* or on #underline[self-hosted infrastructure] offers unique advantages: you gain #underline[*full control*], strict #underline[*data privacy*] (no data leaves your machine!), and *zero recurring API costs*.

#v(0.5em)
- #underline[*Our Focus:*] 
  - We target _turnkey platforms_ that provide an #underline[*OpenAI-compatible API*] out-of-the-box.
  - These tools encapsulate the complexity of model execution, making them #underline[extremely easy] to set up and integrate into standard Software Engineering workflows.
  - The goal is to treat local LLMs just like _standard microservices_.

#v(0.5em)
- #underline[*Out of Scope:*] 
  - _Low-level manual loading_ of models from scratch (e.g., writing raw PyTorch / HuggingFace Transformers pipelines).
  - Orchestration of complex, highly-distributed serving frameworks meant for massive enterprise data center clusters (unless trivialized).

== What a Local LLM Engine Should Offer

When self-hosting an LLM, a good engine should abstract away the complexity of model execution:

- *API Compatibility:* Provide a standardized REST API (often drop-in compatible with OpenAI's format).
- *Hardware Abstraction:* Automatically handle GPU offloading, memory management, and execution optimization.
- *Model Management:* Simplify discovering, downloading, and updating model weights.

*Common Solutions:*
- *Ollama:* Wraps LLMs in a background service, optimized for native local performance and easy CLI management.
- *LM Studio:* Focuses on prototyping with a user-friendly GUI to discover models and test different prompts.
- *vLLM / TGI:* Enterprise-grade engines focused on high throughput and scalability for production workloads.

== Enabling Local Execution: Quantization

*Quantization* is the key to running large models locally. It reduces the memory and computational requirements by representing model weights with lower precision (e.g., from 16-bit floats to 8-bit or 4-bit integers).

- *The Impact on VRAM Requirements (Approximate):*
  - #underline[*7B / 8B parameters*]: ~14GB (16-bit) $arrow.r$ *~4.5GB* (4-bit) — _Runs on a standard laptop._
  - #underline[*32B parameters*]: ~64GB (16-bit) $arrow.r$ *~18GB* (4-bit) — _Runs on a Mac M-series or a consumer GPU (e.g., RTX 4090)._
  - #underline[*70B parameters*]: ~140GB (16-bit) $arrow.r$ *~40GB* (4-bit) — _Runs on high-end workstations (e.g., 2x 24GB GPUs or Mac Studio)._
- *The Benefits:*
  - *Lower Memory Footprint:* Makes otherwise inaccessible models fit into available RAM/VRAM.
  - *Faster Inference:* Reduces memory bandwidth bottlenecks (often the primary constraint for LLMs).
- *The Trade-offs:*
  - May introduce slight losses in model accuracy scaling with aggressive shrinking (e.g., < 4 bits).
  - Requires specific file formats (like GGUF or AWQ) optimized for the targeted inference engine.

== Quantization 
#align(center)[
#image("figures/quantization.png", width: 80%)
]

== Ollama
#align(center)[
  #image("figures/ollama.png", width: 20%)
  
  #link("https://ollama.com/")
]

- *What is it?* A lightweight platform to run LLMs locally as a web service.
  - *Vast library:* 60+ ready-to-use models (e.g., Qwen3).
  - *High performance:* Leverages local hardware for near-native execution speed.
  - *Versatile:* Supports both text generation and embedding tasks.
  - *Customizable:* Easy configuration of parameters (e.g., temperature) and system prompts.


== Ollama -- API Example

`ollama serve` -- starts a local server at `http://localhost:11434` with endpoints for generation and embeddings.

`ollama pull qwen3:0.6b` -- downloads the Qwen3 model for local use.

`ollama run qwen3:0.6b` -- starts an interactive REPL for testing prompts directly in the terminal.

#align(center)[
  #block(
    fill: rgb("#282a36"),  // Dark background simulating a terminal
    stroke: 1pt + rgb("#6272a4"),
    radius: 0.4em,
    inset: 1em,
    width: 90%,
    align(left)[
      #text(fill: rgb("#f8f8f2"), font: "Fira Code", size: 14pt)[
        #text(fill: rgb("#50fa7b"))[>>>] Hello, World! \
        #text(fill: rgb("#6272a4"))[Thinking...] \
        #text(fill: rgb("#6272a4"), style: "italic")[Okay, the user said "Hello, World!" and I need to respond. Let me start by acknowledging their message. Since it's a simple greeting, a friendly response is appropriate. Maybe say "Hello, World!" and offer help. Let them know I'm here to assist...] \
        #text(fill: rgb("#6272a4"))[...done thinking.] \
        \
        #text(fill: rgb("#ffb86c"))[Assistant:] Hello, World! 😊 How can I assist you today?
      ]
    ]
  )
]


== LLM - Reasoning/Thinking models
-  LLMs perform significantly better on #underline[complex reasoning tasks] when allowed to *"think step by step"*.
- Newer models are natively *trained* to generate #underline[intermediate reasoning steps] before outputting the final answer.
-Drastic improvements in *accuracy* and logical coherence, with a significant reduction in reasoning errors.
#align(center)[
  #image(width: 80%, "figures/reasoning.png")
]

// Helper for chat-style conversation blocks
#let chat-block(messages, width: 90%) = align(center)[
  #block(
    fill: rgb("#282a36"),
    stroke: 1pt + rgb("#6272a4"),
    radius: 0.4em,
    inset: 0.8em,
    width: width,
    align(left)[
      #text(font: "Fira Code", size: 12pt, fill: rgb("#f8f8f2"))[
        #messages
      ]
    ]
  )
]

#let sys(content) = text(fill: rgb("#bd93f9"), weight: "bold")[System: ] + text(fill: rgb("#f8f8f2"))[#content]
#let usr(content) = text(fill: rgb("#50fa7b"), weight: "bold")[User: ] + text(fill: rgb("#f8f8f2"))[#content]
#let bot(content) = text(fill: rgb("#ffb86c"), weight: "bold")[Assistant: ] + text(fill: rgb("#f8f8f2"))[#content]

== How to Guide LLMs? — The Ambiguity Problem

LLMs are incredibly powerful, but they need the *right instructions* to perform well on specific tasks.
Human language is inherently #underline[ambiguous] — the same prompt can lead to wildly different outputs.

#v(0.3em)
#text(size: 14pt)[
#underline[*Same prompt, three different results:*]
#chat-block(width: 95%)[
  #usr[Write a function to filter a list.] \
  
]

#v(0.3em)
#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 0.8em,
  [
    #block(
      fill: rgb("#282a36"),
      stroke: 1pt + rgb("#6272a4"),
      radius: 0.4em,
      inset: 0.6em,
      width: 100%,
      align(left)[
        #text(font: "Fira Code", size: 11pt)[
          #text(fill: rgb("#ffb86c"), weight: "bold")[Result 1] \
          #text(fill: rgb("#6272a4"), style: "italic")[Python] \
          #text(fill: rgb("#f8f8f2"))[List comprehension filtering integers > 10]
        ]
      ]
    )
  ],
  pause,
  [
    #block(
      fill: rgb("#282a36"),
      stroke: 1pt + rgb("#6272a4"),
      radius: 0.4em,
      inset: 0.6em,
      width: 100%,
      align(left)[
        #text(font: "Fira Code", size: 11pt)[
          #text(fill: rgb("#ffb86c"), weight: "bold")[Result 2] \
          #text(fill: rgb("#6272a4"), style: "italic")[JavaScript] \
          #text(fill: rgb("#f8f8f2"))[`.filter()` removing null values from objects]
        ]
      ]
    )
  ],
  pause,
  [
    #block(
      fill: rgb("#282a36"),
      stroke: 1pt + rgb("#6272a4"),
      radius: 0.4em,
      inset: 0.6em,
      width: 100%,
      align(left)[
        #text(font: "Fira Code", size: 11pt)[
          #text(fill: rgb("#ffb86c"), weight: "bold")[Result 3] \
          #text(fill: rgb("#6272a4"), style: "italic")[C++] \
          #text(fill: rgb("#f8f8f2"))[`std::copy_if` extracting even numbers]
        ]
      ]
    )
  ],
)
]

#v(0.3em)
#align(center)[
  #text(size: 18pt, style: "italic", fill: rgb("#555555"))[Same prompt, three completely different interpretations — language, data type, and filter logic all vary.]
]

== How to Guide LLMs? — Humans vs. Models

#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    #underline[*Between Humans*]
    #v(0.5em)
    We naturally disambiguate through shared context:
    - *Role:* We know _who_ we are talking to (e.g., Senior Dev helping a Junior).
    - *Context:* We know _what_ we are working on (e.g., the data-processing module).
    - *Outcome:* We know _what_ we need (e.g., remove duplicates from a dataset).
  ],
  [
    #underline[*With LLMs*]
    #v(0.5em)
    Models lack implicit context — they must be #underline[explicitly] guided:
    - *Role:* Must be stated in the prompt (e.g., _"You are an expert Python developer."_).
    - *Context:* Must be provided as input (e.g., _"We are building a REST API in FastAPI."_).
    - *Outcome:* Must be clearly specified (e.g., _"Return a function that removes duplicates."_).
  ],
)

#v(1em)
#align(center)[
  The discipline of crafting these explicit instructions is called *prompt engineering*.
]

== Prompt Engineering

*Prompt Engineering* is the discipline of crafting inputs to guide LLMs toward desired outputs.
- *Goal:* Reduce ambiguity and constrain the model's probability space.
- *Key:* Iterative refinement of clear, specific, and structured instructions.

#v(0.5em)
#underline[*Prompt Structure (General)*]
#v(0.3em)
A prompt typically consists of two key components:
- *System instructions:* High-level guidelines that set the model's overall behavior and persona (e.g., _"You are a helpful assistant that provides concise answers."_).
- *User input:* The specific request or question the model needs to address (e.g., _"What is the capital of France?"_).

#v(0.5em)
#underline[*Why It Matters*]
#v(0.3em)
- Prompts range from a few words to structured, multi-paragraph instructions.
- They can steer the model toward a specific _task_, _style_, _format_, or _constraint_.
- Small changes in phrasing can #underline[significantly] influence the model's output quality and relevance.

For an overview of best practices, see the #link("https://www.promptingguide.ai/")[Prompt Engineering Handbook].

== Prompt Engineering -- Best Practices

#text(size: 18pt)[
#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    #underline[*1. Set Clear Goals*]
    #v(0.3em)
    - Use _action verbs_ to specify the desired action
    - Define the _length_ and _format_ of the output
    - Specify the _target audience_
    #v(0.5em)
    #underline[*2. Provide Context*]
    #v(0.3em)
    - Include relevant _facts and data_
    - Reference specific _sources or documents_
    - Define _key terms_ and _constraints_
  ],
  [
    #underline[*3. Be Specific*]
    #v(0.3em)
    - Use _precise language_, avoid ambiguity
    - _Quantify_ requests whenever possible
    - _Break down_ complex tasks into steps
    #v(0.5em)
    #underline[*4. Iterate and Experiment*]
    #v(0.3em)
    - Try different _phrasings_ and _keywords_
    - Adjust the _level of detail_
    - Test different _prompt lengths_
  ],
)
]
#v(0.5em)
#align(center)[
  #text(size: 16pt, style: "italic", fill: rgb("#555555"))[See also: #link("https://cloud.google.com/discover/what-is-prompt-engineering")[Google Cloud — What is Prompt Engineering?]]
]

== Prompt Engineering -- Zero-Shot

#underline[*Zero-shot prompting: direct instructions with no examples.*]
#v(0.5em)
The model relies _entirely_ on its pretraining knowledge and the instructions in the prompt. This is the most common interaction mode (e.g., everyday chat).

#v(0.5em)
#underline[*General Structure*]
#v(0.3em)
#text(size: 16pt)[
  + *Role* — Who the model should act as.
  + *Context* — Background information relevant to the task.
  + *Task instructions* — What the model should do.
  + *Output format* — How the result should be structured.
]

#v(0.5em)
#underline[*When to Use*]
#v(0.3em)
- Tasks where the model already has strong knowledge (e.g., translation, summarization, classification).
- Quick prototyping — no need to prepare examples upfront.
- Simple, well-defined requests where ambiguity is low.

== Prompt Engineering -- Zero-Shot Examples (1/2)

#text(size: 14pt)[
#underline[*1. Sentiment Classification*]
#chat-block[
  #sys[You are a sentiment analysis expert.] \
  \
  #usr[Classify the sentiment of this review as Positive, Negative, or Mixed: \ "This restaurant had amazing food but terrible service."] \
  \
  #bot[Mixed]
]

#v(0.5em)
#underline[*2. Code Generation*]
#chat-block[
  #sys[You are a senior Python developer. Always include type hints and docstrings.] \
  \
  #usr[Write a function that takes a list of integers and returns only the even numbers using a list comprehension.] \
  \
  #bot[```python
def filter_even(nums: list[int]) -> list[int]:
    """Return only even numbers from the input list."""
    return [n for n in nums if n % 2 == 0]
```]
]
]

== Prompt Engineering -- Zero-Shot Examples (2/2)

#text(size: 14pt)[
#underline[*3. Summarization*]
#chat-block[
  #sys[You are a technical writer who explains concepts in plain language.] \
  \
  #usr[Summarize the following in exactly two sentences for a non-technical audience: \ "TCP/IP is a layered protocol suite that governs how data packets are addressed, transmitted, routed, and received across networks..."] \
  \
  #bot[TCP/IP is a set of rules that computers use to send and receive information over the internet. It breaks data into small packages, sends them along the best route, and reassembles them at the destination.]
]

#v(0.5em)
#underline[*4. Structured Data Extraction*]
#chat-block[
  #sys[You are a data extraction assistant. Always respond with valid JSON.] \
  \
  #usr[Extract name, email, and company from: \ "Hi, I'm Alice from Acme Corp. Reach me at alice\@acme.com."] \
  \
  #bot[```json
{"name": "Alice", "email": "alice@acme.com", "company": "Acme Corp"}
```]
]
]

== Prompt Engineering -- Few-Shot

#underline[*Few-shot prompting: provide examples to demonstrate the desired pattern.*]
#v(0.5em)
Instead of only describing the task, you _show_ the model what you expect through a small number of input $arrow.r$ output pairs.

#v(0.5em)
#grid(
  columns: (1fr, 1fr),
  gutter: 2em,
  [
    #underline[*Key Concepts*]
    #v(0.3em)
    - Examples act as *implicit instructions* — the model learns the pattern and mimics it.
    - Significantly improves performance over zero-shot for _ambiguous_ or _domain-specific_ tasks.
    - Number of examples matters: typically *2–5* shots are enough; more can help but consume context.
    - Examples should be *representative* and *diverse* to avoid bias.
  ],
  [
    #underline[*When to Use*]
    #v(0.3em)
    - The desired output format is *unusual or strict* (e.g., a custom schema).
    - The task is *ambiguous* and a verbal description alone is insufficient.
    - The model underperforms in zero-shot mode for the specific task.
    - You want to enforce a consistent *style or tone* across outputs.
  ],
)

== Prompt Engineering -- Few-Shot Examples (1/2)

#text(size: 14pt)[
#underline[*1. Email Classification*]
#chat-block[
  #sys[You classify emails as SPAM or NOT SPAM.] \
  \
  #usr[Classify the following emails: \
  \
  Email: "Congratulations! You've won \$1,000,000 in our lottery!" $arrow.r$ SPAM \
  Email: "Meeting at 2pm tomorrow in the conference room." $arrow.r$ NOT SPAM \
  \
  Email: "Your account has been compromised. Click here immediately." $arrow.r$] \
  \
  #bot[SPAM]
]

#v(0.5em)
#underline[*2. Custom Entity Extraction*]
#chat-block[
  #usr[Extract the programming language and framework from each sentence: \
  \
  "We built the API in Go with Gin." $arrow.r$ Language: Go, Framework: Gin \
  "The frontend uses TypeScript and React." $arrow.r$ Language: TypeScript, Framework: React \
  \
  "Our pipeline is written in Rust using Actix." $arrow.r$] \
  \
  #bot[Language: Rust, Framework: Actix]
]
]

== Prompt Engineering -- Few-Shot Examples (2/2)

#text(size: 14pt)[
#underline[*3. Tone/Style Control*]
#chat-block[
  #sys[You rewrite informal developer messages into formal, professional language.] \
  \
  #usr[Rewrite the sentence in a formal tone: \
  \
  "This app is kinda buggy." $arrow.r$ "The application exhibits several defects." \
  "The server crashed again lol." $arrow.r$ "The server experienced an unexpected failure." \
  \
  "Yo the deploy broke prod." $arrow.r$] \
  \
  #bot["The deployment introduced a critical issue in the production environment."]
]

#v(0.5em)
#underline[*4. Commit Message Generation*]
#chat-block[
  #usr[Generate a conventional commit message from the diff description: \
  \
  "Added null check in UserService.getById" $arrow.r$ fix(user): add null check in getById to prevent NPE \
  "Renamed Config to AppConfig and moved to config pkg" $arrow.r$ refactor(config): rename Config to AppConfig and relocate to config package \
  \
  "Added retry logic with exponential backoff to HttpClient" $arrow.r$] \
  \
  #bot[feat(http): add retry logic with exponential backoff to HttpClient]
]
]

== From Prompting to Applications

So far we have explored how to *interact with LLMs* through prompt engineering — crafting the right instructions to get useful outputs.

#v(0.5em)

But building real *applications* on top of LLMs requires more than just prompting:

#v(0.3em)

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    #underline[*Challenges*]
    #v(0.3em)
    - Switching between _providers_ (OpenAI, Ollama, Gemini…) without rewriting code
    - #underline[Chaining] multiple LLM calls and processing steps together
    - Integrating _external data sources_ (databases, APIs, documents)
    - Managing _conversation history_ and _context windows_
  ],
  [
    #underline[*What We Need*]
    #v(0.3em)
    - A *unified abstraction layer* over different LLM providers
    - Composable *pipelines* for multi-step tasks
    - Built-in support for _tool use_ and _external integrations_ (more on this later)
    - A principled way to build #underline[agentic] workflows
  ],
)

#v(0.5em)
#align(center)[
  #fa-arrow-right() We need a *framework* that bridges the gap between _prompt engineering_ and _software engineering_.
]

== LangChain4J

#grid(
  columns: (auto, 1fr),
  gutter: 1.5em,
  align: horizon,
  [
    #image("figures/logo.png", width: 8em)
  ],
  [
    A *Java framework* for building LLM-powered applications.
    #v(0.3em)
    Inspired by the original Python #link("https://github.com/langchain-ai/langchain")[LangChain] library, redesigned to be _idiomatic_ for the Java ecosystem (also drawing ideas from #link("https://www.llamaindex.ai/")[LlamaIndex]).
    #v(0.3em)
    #align(left)[#link("https://github.com/langchain4j/langchain4j")]
  ],
)

#v(0.5em)

#text(size: 18pt)[
#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    #underline[*Core Features*]
    #v(0.3em)
    - *Unified provider API* — swap between OpenAI, Ollama, Gemini, Anthropic, etc. with _one-line config changes_
    - *AI Services* — compose multi-step LLM workflows (prompt $arrow.r$ call $arrow.r$ parse $arrow.r$ act)
    - *Tool integration* — let the model call _external functions_ (search, DB queries, REST APIs) — the basis for *agentic AI*, more later
  ],
  [
    #underline[*Why LangChain4J?*]
    #v(0.3em)
    - First-class *Java/Kotlin* support with type-safe APIs
    - Built-in *memory* management for multi-turn conversations
    - Native *embedding stores* for Retrieval-Augmented Generation (RAG)
    - Active community and _Spring Boot_ integration
  ],
)
]

== LangChain4J -- Main abstraction:

#image("figures/langchain4j-abstract.png")

== LangChain4J -- Main abstraction

- *ChatModel*: the model used to generate responses (e.g., OpenAI, Ollama, Gemini).
- *ChatMessage*: the input messages (system, user, assistant) that define the conversation context.
- *Response*: the output generated by the model, which can be further processed or parsed. It also contains metadata about the generation process (e.g., tokens used, latency).
- *ModelBuilder*: a fluent API for configuring and instantiating `ChatModel` instances with specific parameters (e.g., temperature, max tokens, provider).
- Similar architecture persists for `EmbeddingModel`, with appropriate adjustments for the different input/output formats.

== LangChain4J -- Example Usage

First import the dependencies:
```scala
libraryDependencies += "dev.langchain4j" % "langchain4j" % "1.11.0"
libraryDependencies += "dev.langchain4j" % "langchain4j-ollama" % "1.11.0"
```

Create and configure the model:
```java
final ChatModel model = OllamaChatModel.builder()
    .baseUrl(LlmConstants.OLLAMA_BASE_URL)
    .logRequests(true)
    .logResponses(true)
    .modelName(LlmConstants.CHAT_MODEL_SMOLLM)
    .numPredict(LlmConstants.MAX_PREDICT_TOKENS)
    //.temperature(0.0)
    //.topK(1)
    .build();
```

And then interact with the model:
```java
final UserMessage message = UserMessage.userMessage("Say Hello!");
var response = model.chat(message);
```

#focus-slide()[
  == Demo
  #link("https://github.com/cric96/asmd-llm-code")
]

== Managing Input and Output
Integrating LLMs into software requires bridging the gap between *structured domain models* and *unstructured natural language*.

- *Input (Prompt Construction):* Translating the application state (e.g., a game `Board`) into text. This is often done using *Prompt Builders* or *Templates* that inject dynamic values at runtime to create _context-aware prompts_.
- *Output (Response Parsing):* The LLM returns _free-form text_. You must extract *structured data* from it (e.g., `row,col` coordinates) using techniques like *Regular Expressions*, *JSON parsing*, or dedicated *output parsers*.
  - Modern LLMs can be trained to produce *strictly formatted* outputs (e.g., JSON) to simplify parsing---this is called *structured output*, which also accepts a _schema definition_ to validate the output format.
- *Resilience (Validation & Fallbacks):* LLMs are *non-deterministic* and can produce invalid output (hallucinate bad moves, break format). Production code must include *retry mechanisms*, *data validation*, and *safe fallbacks* to guarantee stability.

#focus-slide()[
= Demo
 Tic Tac Toe AI Player
]
 
== Open Questions (Next Lectures & Discussion)
- *Testing LLM Integrations:*
  - We tested our code by mocking the LLM API calls.
  - But how do we verify that the *actual model* behaves as expected in reality?
  - How do we test for *edge cases*, non-determinism, and hallucinated *failure modes*?
- *Evaluating LLM Applications:*
  - How can we systematically compare the performance of *different models* or *prompting strategies*?
  - Do we need new *metrics* to score subjective natural language outputs instead of exact matches?
- *AI-Assisted Quality Assurance:*
  - Can we use LLMs themselves to generate *validation pipelines* or bootstrap *test cases*?
  - How do we integrate these checks securely into continuous integration pipelines?

== Conclusion
- *What we covered:*
  - *LLM Fundamentals:* Core concepts, capabilities, and limitations.
  - *Interfacing:* Working with remote APIs and local execution engines.
  - *Prompt Engineering:* Crafting prompts to effectively steer model behavior.
  - *Application Development:* Building LLM-powered apps using frameworks like LangChain4J.
- *What's next:*
  - Enabling LLMs to interact with external tools and live data sources.
  - Introducing *Agentic AI* — where models autonomously call APIs, query databases, and execute code to solve complex goals.
  - Exploring how autonomous AI agents are transforming modern software engineering pipelines.
- *Lab Session:*
  - Hands-on practice: Building a simple LLM-powered application using Ollama and LangChain4J.

== Resources (1/2): Learn & Visualize
- *Visualizers & Playgrounds:*
  - *OpenAI Tokenizer*: See text-to-token encoding. \ #link("https://platform.openai.com/tokenizer")
  - *Transformer Explainer*: Visualizer of a GPT model's internal operations. \ #link("https://poloclub.github.io/transformer-explainer/")
  - *Cohere Embeddings*: Test semantic similarity. \ #link("https://dashboard.cohere.com/playground/embed")
- *Prompt Engineering:*
  - *Anthropic Interactive Tutorial*: Hands-on course on steering models. \ #link("https://github.com/anthropics/prompt-eng-interactive-tutorial")
  - *Prompt Engineering Handbook*: Guide by DAIR.AI on prompt techniques. \ #link("https://www.promptingguide.ai/")
- *Background & Theory:*
  - *Large Language Models explained briefly*: Concise video explanation. \ #link("https://www.youtube.com/watch?v=LPZh9BOjkQs")
  - *NLP Progress*: SOTA tracking across NLP tasks. \ #link("https://nlpprogress.com/")

== Resources (2/2): Tools & Code
- *Models & Runtimes:*
  - *Ollama*: Standard tool to run open-weight models locally. \ #link("https://ollama.com/")
  - *LifeArchitect.ai Models Table*: Huge spreadsheet of current LLMs. \ #link("https://lifearchitect.ai/models-table/")
- *Lists & Collections:*
  - *Awesome NLP* & *Deep Learning for NLP*: Curated lists of NLP libraries. \ #link("https://github.com/keon/awesome-nlp") & #link("https://github.com/brianspiering/awesome-dl4nlp")
- *Building Applications:*
  - *LangChain4J*: Java framework for LLM orchestration. \ #link("https://github.com/langchain4j/langchain4j")
  - *Course Demo Repository*: Code examples for LLM-powered apps. \ #link("https://github.com/cric96/asmd-llm-code")
