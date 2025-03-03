#import "@preview/touying:0.5.2": *
#import themes.metropolis: *
#import "@preview/fontawesome:0.1.0": *
#import "@preview/ctheorems:1.1.2": *
#import "@preview/numbly:0.1.0": numbly

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

// Theorems configuration by ctheorems
#show: thmrules.with(qed-symbol: $square$)
#let theorem = thmbox("theorem", "Theorem", fill: rgb("#eeffee"))
#let corollary = thmplain(
  "corollary",
  "Corollary",
  base: "theorem",
  titlefmt: strong
)
#let definition = thmbox("definition", "Definition", inset: (x: 1.2em, top: 1em))
#let example = thmplain("example", "Example").with(numbering: none)
#let proof = thmproof("proof", "Proof")

#show: metropolis-theme.with(
  aspect-ratio: "4-3",
  footer: self => self.info.author + ", " + self.info.institution + " - " + self.info.date,
  config-common(
    // handout: true,
    preamble: pdfpc-config, 
  ),
  config-info(
    title: [Leveraging Large Language Models in Software Engineering],
    subtitle: [LLM Primer and Overview],
    author: [Gianluca Aguzzi],
    date: datetime.today().display("[day] [month repr:long] [year]"),
    institution: [Università di Bologna],
    // logo: emoji.school,
  ),
)

#set text(font: "Fira Sans", weight: "light", size: 18pt)
#show math.equation: set text(font: "Fira Math")

#title-slide()

= Introduction 

== Today Lesson in a Nutshell
#align(center)[
    #image("figures/meme.jpg", width: 60%)
  ]


== Today Lesson (Seriously)
- #text(weight: "bold")[Goal:] Understand the fundamentals of Natural Language Processing (NLP) and Language Models (LM).

  - From a ``pratictal'' perspective.
  - Understading the basic concepts and the common tasks.
  - How to use them from API and libraries.
  - How to ``tune'' them for specific (soft. eng.) tasks.
- #text(weight: "bold")[Note:]
  - We will not dive too much into the details of the algorithms and the mathematics behind them.
  - For this, please refer to the resources provided and the course on NLP.

== NLP & Soft. Eng. -- Why BTW?
#align(center)[
  #image("figures/copilot.png", width: 80%)

]
== NLP & Soft. Eng. -- Why BTW?
#align(center)[
  #image("figures/copilot copy.png", width: 100%)
]

== NLP & Soft. Eng. -- Why BTW?
#align(center)[
  #image("figures/soft-eng-improvements.png", width: 50%)
]
== NLP & Soft. Eng. -- Why Should We Care?
- The Software Engineering landscape is *rapidly evolving*: #fa-rocket()
  - AI pair programmers (like Copilot) are becoming ubiquitous tools #fa-robot()
  - LLMs can now handle tasks previously requiring human expertise: #fa-brain()
    - Intelligent code completion #fa-code()
    - Automated documentation generation #fa-file-alt()
    - Assisted refactoring and optimization #fa-wrench()
    - Test case generation #fa-vial()
  - Key questions for modern developers: #fa-question-circle()
    - What will be our role in this AI-augmented future? #fa-user-cog()
    - How can we best leverage NLP to enhance our productivity? #fa-chart-line()
    - Which skills remain uniquely human in software development? #fa-fingerprint()
  - Understanding this technology isn't optional—it's essential for staying relevant #fa-lightbulb() #fa-exclamation()

= Natural Language Processing and (Large) Language Models

== Natural Langauge Processing
#align(center)[
  #text(size: 28pt, weight: "bold")[Natural Language Processing (NLP)]
  
  #v(1em)
  
  #text(size: 20pt)[
    A subfield of artificial intelligence that focuses #emph[understanding], #emph[interpreting], and #emph[generating] human language.
  ]
  #v(1em)
]
#text(size: 18pt, weight: "bold")[Resources]
  #set list(indent: 1em)
  - #link("https://github.com/keon/awesome-nlp?tab=readme-ov-file")
  - #link("https://github.com/brianspiering/awesome-dl4nlp")
  - #link("https://nlpprogress.com/")
  - #link("https://www.unibo.it/it/studiare/dottorati-master-specializzazioni-e-altra-formazione/insegnamenti/insegnamento/2023/412644")
  - #link("https://www.youtube.com/watch?v=LPZh9BOjkQs")


== Natural Language Processing

#block(
  fill: rgb("#c5e0d880"), 
  width: 100%,
  inset: 1em,
  radius: 8pt,
  stroke: (paint: rgb("#9aafa9"), thickness: 1pt),
  [
    #text(weight: "bold")[Goal]
    
    Identify the structure and meaning of #emph[words], #emph[phases], and #emph[sentences] in order to enable computers to understand and generate human language.
  ]
)

#block(
  fill: rgb("#e6e6e6"),  // Light gray
  width: 100%,
  inset: 1em,
  radius: 8pt,
  stroke: (paint: rgb("#c7c5c5"), thickness: 1pt),
  [
    #text(weight: "bold")[Why?]
    
    Improve #emph[human-computer] interaction, closing the gap between #emph[human communication] and #emph[computer understanding].
  ]
)

#block(
  fill: rgb("#e6e6e6"),  // Light gray
  width: 100%,
  inset: 1em,
  radius: 8pt,
  stroke: (paint: rgb("#c7c5c5"), thickness: 1pt),
  [
    #text(weight: "bold")[Applications #text(weight: "bold")[(all around us)]]
    
    #grid(
      columns: 3,
      gutter: 1em,
      [
        - #emph[Chatbots]
        - #emph[Machine Translation]
        - #emph[Speech Recognition]
      ],
      [
        - #emph[Sentiment Analysis]
        - #emph[Question Answering]
        - #emph[Code Generation]
      ],

      [
        - #emph[Image Captioning]
        - #emph[Summarization]
        - #emph[Text Classification]
      ]
    )
  ]
)


== Natural Language Processing

#block(
fill: rgb("#FDE8E9"),  // Light gray
  width: 100%,
  inset: 1em,
  radius: 8pt,
  stroke: (paint: rgb("#d4c3c4"), thickness: 1pt),
  [
    #text(weight: "bold")[Challenges]
    
    - #text(weight: "bold")[Ambiguity:] Multiple meanings for words/phrases.
    - #text(weight: "bold")[Context:] Meaning shifts with context (linguistic, cultural).
    - #text(weight: "bold")[Syntax:] Sentence structure affects meaning.
    - #text(weight: "bold")[Sarcasm/Idioms:] Non-literal language interpretation.
  ]
)

#block(
  fill: rgb("#e6e6e6"),  // Light gray
  width: 100%,
  inset: 1em,
  radius: 8pt,
  stroke: (paint: rgb("#c7c5c5"), thickness: 1pt),
  [
    #text(weight: "bold")[Approaches]
    
    - #text(weight: "bold")[Rule-Based:] Hand-crafted linguistic rules (e.g., #link("https://en.wikipedia.org/wiki/Georgetown-IBM_experiment")[Georgetown–IBM]).
    - #text(weight: "bold")[Statistical:] Probabilistic language modelling (e.g., hidden Markov model)#footnote("Mérialdo, B. (1994). Tagging English Text with a Probabilistic Model. Comput. Linguistics, 20(2), 155–171.")
    - #text(weight: "bold")[ML/Deep Learning:] Algorithms learn from data; neural networks model complex patterns (RNN#footnote("Yin, W., Kann, K., Yu, M., & Schütze, H. (2017). Comparative Study of CNN and RNN for Natural Language Processing. CoRR, abs/1702.01923. Retrieved from http://arxiv.org/abs/1702.01923"), LSTM#footnote("Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Comput., 9(8), 1735–1780. doi:10.1162/NECO.1997.9.8.1735"), GRU#footnote("Dey, R., & Salem, F. M. (2017). Gate-variants of Gated Recurrent Unit (GRU) neural networks. IEEE 60th International Midwest Symposium on Circuits and Systems, MWSCAS 2017, Boston, MA, USA")) 
    - _Goal_: Find a *Language Model* that understands and generates human language.
  ]
)
== Language Models
#align(center)[
  #text(size: 28pt, weight: "bold")[What is a #text(weight: "bold")[Language Model]?]
  
  #v(1em)
  
  #text(size: 20pt)[
    A #emph[machine learning] model that aims to predict and generate *plausible* text.
  ]


  #align(
    center,
    block[
      #image("figures/llm-nutshell.png", width: 80%)
    ]
  )
  
]
== Language Models
#block(
  fill: rgb("#c5e0d880"), 
  width: 100%,
  inset: 1em,
  radius: 8pt,
  stroke: (paint: rgb("#9aafa9"), thickness: 1pt),
  [
    #text(weight: "bold")[ Fundamental Idea]
    
    Text is a sequence of words, and language models learn the #emph[probability distribution] of a word given the previous words in context.
  ]
)

#block(
  fill: rgb("#e6e6e6"),  // Light gray
  width: 100%,
  inset: 1.2em,
  radius: 8pt,
  stroke: (paint: rgb("#c7c5c5"), thickness: 1pt),
  [
    #text(weight: "bold")[ Simple Examples]
    
    - #emph[The customer was very happy with the <\*>]
    - #emph[The customer was very happy with the #text(weight: "bold", fill: rgb("#555555"))[service].]
    - #emph[The customer was very happy with the #text(weight: "bold", fill: rgb("#555555"))[product].]
  ]
)

#block(
  fill: rgb("#e6e6e6"),  // Light gray
  width: 100%,
  inset: 1.2em,
  radius: 8pt,
  stroke: (paint: rgb("#c7c5c5"), thickness: 1pt),
  [
    #text(weight: "bold")[Common Tasks]
    
    - #emph[Text Generation:] Complete or continue text based on a prompt
    -  #emph[Classification:] Categorize text (sentiment, topic, intent)
    - #emph[Question Answering:] Find answers within a context
    - #emph[Summarization:] Condense longer texts into summaries
    -  #emph[Translation:] Convert text between languages
  ]
)
== Language Models -- Phases


*1. Tokenization:* split text into words, phrases, symbols, etc.
- _Example:_ "Hello, world!" #fa-arrow-right() ["Hello", ",", "world", "!"]

*2. Embedding:* convert words into numerical vectors.
- _Example:_ "Hello" #fa-arrow-right() [0.25, -0.75, 0.5, 1.0]
- It is possible to use pretrained embeddings (e.g., Word2Vec, BERT).

*3. Modelling:* learn the probability of a word given the previous words.
- _Example:_ P("world" | "Hello,") #fa-arrow-right() 0.8

*4. Generation/Classification:* use the model to generate text or classify it.
- _Example for Generation:_ Input: "The weather is" #fa-arrow-right() Output: "sunny."
- _Example for Classification:_ Input: "This is a spam email." #fa-arrow-right() Output: Spam

#v(0.5em)

#text(weight: "bold")[Note:] each NLP solution can use different techniques for each phase.

== Tokenization
#block(
  fill: rgb("#c5e0d880"), 
  width: 100%,
  inset: 1em,
  radius: 8pt,
  stroke: (paint: rgb("#9aafa9"), thickness: 1pt),
  [
    #text(weight: "bold")[Tokenization: Breaking Text into Pieces]
    
    Splitting text into discrete units (tokens) for the model to process.
    #link("https://platform.openai.com/tokenizer")
  ]
)

#grid(
  columns: 2,
  gutter: 1em,
  [
    #block(
      fill: rgb("#e6e6e6"),
      width: 100%,
      height: 65%,
      inset: 1em,
      radius: 8pt,
      stroke: (paint: rgb("#c7c5c5"), thickness: 1pt),
      [
        #text(weight: "bold")[Example]
        
        "I heard a dog bark loudly at a cat"
        
        #text(size: 16pt)[
          Tokenized as: {1, 2, 3, 4, 5, 6, 7, 3, 8}
          
          Where:
          - I (1)
          - heard (2)
          - a (3)
          - dog (4)
          - bark (5)
          - loudly (6)
          - at (7)
          - cat (8)
        ]
      ]
    )
  ],
  [
    #block(
      fill: rgb("#e6e6e6"),
      width: 100%,
      height: 65%,
      inset: 1em,
      radius: 8pt,
      stroke: (paint: rgb("#c7c5c5"), thickness: 1pt),
      [
        #text(weight: "bold")[In Practice]
        
        - Tokens can be words, subwords, or characters
        - Modern models use subword tokenization
        - Vocabulary size: typically 30k-100k tokens
        - Out-of-vocabulary words get split into known tokens
        - Special tokens: [START], [END], [PAD], [MASK]
      ]
    )
  ]
)

== Embedding
#block(
  fill: rgb("#c5e0d880"), 
  width: 100%,
  inset: 1em,
  radius: 8pt,
  stroke: (paint: rgb("#9aafa9"), thickness: 1pt),
  [
    #text(weight: "bold")[Embedding: Converting Words to Numbers]
    
    Translating tokens into numerical vectors that capture *semantic meaning*.
  ]
)

#grid(
  columns: 2,
  gutter: 1em,
  [
    #block(
      fill: rgb("#e6e6e6"),
      width: 100%,
      height: 65%,
      inset: 1em,
      radius: 8pt,
      stroke: (paint: rgb("#c7c5c5"), thickness: 1pt),
      [
        #text(weight: "bold")[Example]
        
        "dog" might be represented as:
        
        #text(size: 16pt)[
          [0.2, -0.4, 0.7, 0.1, ...]
          
          Properties:
          - Similar words have similar vectors
          - "dog" is closer to "puppy" than to "table"
          - Vector dimensions capture semantic features
          - Enables mathematical operations on words
        ]
      ]
    )
  ],
  [
    #block(
      fill: rgb("#e6e6e6"),
      width: 100%,
      height: 65%,
      inset: 1em,
      radius: 8pt,
      stroke: (paint: rgb("#c7c5c5"), thickness: 1pt),
      [
        #text(weight: "bold")[In Practice]
        
        - Vectors typically have 100-1000 dimensions
        - Word2Vec, GloVe: Static embeddings
        - BERT, GPT: Contextual embeddings
        - Enables semantic operations:
          "king" - "man" + "woman" ≈ "queen"
        - Forms foundation for downstream tasks
      ]
    )
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
#text(weight: "bold")[Limitations of Traditional Approaches]

- #text(weight: "bold")[RNN:] Long-term dependencies are hard to capture
- #text(weight: "bold")[RNN:]  Slow to train; not suitable for large-scale data
- #text(weight: "bold")[CNN:] Fixed-size input window; not suitable for variable-length text
- #text(weight: "bold")[Both:] Struggle with large-scale parallelization
- #text(weight: "bold")[Solution:] #fa-lightbulb() #emph[Multi-head self-attention] — the core of #emph[transformers]

Transformers overcome these limitations by: #fa-rocket()
- Processing entire sequences in parallel #fa-bolt()
- Using attention to weigh token importance #fa-balance-scale()
- Capturing relationships across arbitrary distances #fa-project-diagram()
- Enabling efficient training on massive datasets #fa-database()

== Transformers -- Visual
#align(
  center,
  block[
    #image("figures/TransformerBasedTranslator.png", width: 100%)
  ]
)
== Transformers Architecture
#block(
  fill: rgb("#c5e0d880"), 
  width: 100%,
  inset: 1em,
  radius: 8pt,
  stroke: (paint: rgb("#9aafa9"), thickness: 1pt),
  [
    #text(weight: "bold")[Transformers: State-of-the-Art for Language Models]
  ]
)

#grid(
  columns: 2,
  gutter: 1em,
  [
    #block(
      fill: rgb("#e6e6e6"),
      width: 100%,
      height: 70%,
      inset: 1em,
      radius: 8pt,
      stroke: (paint: rgb("#c7c5c5"), thickness: 1pt),
      [
        #text(weight: "bold")[Architecture Types]
        
        - #text(weight: "bold")[Encoder-only:]
          - Creates embeddings from input text
          - Use: Classification, token prediction
          - Examples: BERT, RoBERTa
          
        - #text(weight: "bold")[Decoder-only:]
          - Generates new text based on context
          - Use: Continuations, chat responses
          - Examples: *GPT family, LLaMA*
      ]
    )
  ],
  [
    #block(
      fill: rgb("#e6e6e6"),
      width: 100%,
      height: 70%,
      inset: 1em,
      radius: 8pt,
      stroke: (paint: rgb("#c7c5c5"), thickness: 1pt),
      [
        #text(weight: "bold")[Full Transformers]
        
        - Contains both encoder and decoder
        - #text(weight: "bold")[Encoder:] Processes input into intermediate representation
        - #text(weight: "bold")[Decoder:] Converts representation into output text
        - Example use case: Translation
          - English → Intermediate representation → French
        - Examples: *T5, BART, Marian MT*
      ]
    )
  ]
)

== Transformers -- Self-attention

#block(
  fill: rgb("#c5e0d880"), 
  width: 100%,
  inset: 1em,
  radius: 8pt,
  stroke: (paint: rgb("#9aafa9"), thickness: 1pt),
  [
    #text(weight: "bold")[Self-attention: The Key to Context Understanding]
  ]
)

#grid(
  columns: 2,
  gutter: 1em,
  [
    #block(
      fill: rgb("#e6e6e6"),
      width: 100%,
      height: 60%,
      inset: 1em,
      radius: 8pt,
      stroke: (paint: rgb("#c7c5c5"), thickness: 1pt),
      [
        #text(weight: "bold")[The Core Question]
        
        For each token, self-attention asks:

        "How much does each other token affect the interpretation of this token?"
        
        - Attention weights determine token relationships
        - "Self" means within the same input sequence
        - Enables context-aware understanding
      ]
    )
  ],
  [
    #block(
      fill: rgb("#e6e6e6"),
      width: 100%,
      height: 60%,
      inset: 1em,
      radius: 8pt,
      stroke: (paint: rgb("#c7c5c5"), thickness: 1pt),
      [
        #text(weight: "bold")[Example]
        
        "The animal didn't cross the street because it was too tired."
        
        - Each word pays attention to all others
        - Pronoun "it" is *ambiguous*
        - Self-attention reveals: "it" refers to "animal"
        - Resolves dependencies across *arbitrary distances*
      ]
    )
  ]
)

== Self-attention -- Visual

#align(
  center,
  block[
    #image("figures/self-attention.png", width: 50%)
  ]  
)
- Self-attention calculates *weighted relationships* between every token #fa-arrows-alt()
- These relationships reveal which parts of text should *influence* each token #fa-lightbulb()
- Attention weights are *learned parameters* during model training #fa-cogs()
- Multi-head attention allows model to focus on *different relationship types* simultaneously #fa-layer-group()
- This mechanism captures both *local and long-range dependencies* #fa-project-diagram()

== Transformers -- Training
#block(
  fill: rgb("#c5e0d880"), 
  width: 100%,
  inset: 1em,
  radius: 8pt,
  stroke: (paint: rgb("#9aafa9"), thickness: 1pt),
  [
    #text(weight: "bold")[Self-Supervised Learning]: The Key to LLM Training -- #link("https://bbycroft.net/llm")
  ]
)

#grid(
  columns: 2,
  gutter: 1em,
  [
    #block(
      fill: rgb("#e6e6e6"),
      width: 100%,
      height: 65%,
      inset: 1em,
      radius: 8pt,
      stroke: (paint: rgb("#c7c5c5"), thickness: 1pt),
      [
        #text(weight: "bold")[What Makes It Self-Supervised?]
        
        - Data creates its *own supervision signal*
        - No human annotations or labels needed
        - Model learns to predict parts of its input
        - Example: "The \_\_ of sleepy town weren't \_\_" #fa-arrow-right() "people, happy"
        - Leverages *natural structure* in language itself
      ]
    )
  ],
  [
    #block(
      fill: rgb("#e6e6e6"),
      width: 100%,
      height: 65%,
      inset: 1em,
      radius: 8pt,
      stroke: (paint: rgb("#c7c5c5"), thickness: 1pt),
      [
        #text(weight: "bold")[Advantages]
        
        - Uses *unlimited* text data from the internet
        - Scales efficiently with more data and compute
        - Creates rich representations of language
        - Learns grammar, facts, reasoning, and more
        - Forms foundation for downstream adaptation
      ]
    )
  ]
)
