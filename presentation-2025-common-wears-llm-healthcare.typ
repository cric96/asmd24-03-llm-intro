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
  aspect-ratio: "16-9",
  footer: self => self.info.author + ", " + self.info.institution + " - " + self.info.date,
  config-common(
    // handout: true,
    preamble: pdfpc-config, 
  ),
  config-info(
    title: [Small Langauge Model for Personalized Tasks],
    subtitle: [],
    author: [Gianluca Aguzzi],
    date: datetime.today().display("[day] [month repr:long] [year]"),
    institution: [Università di Bologna],
    // logo: emoji.school,
  ),
)

#set text(font: "Fira Sans", weight: "light", size: 18pt)
#show math.equation: set text(font: "Fira Math")

#title-slide()
/*
Presentation Flow 

LLM What and Main alternatives
 => Big and Closed, best performing
 => Small and Open, best for privacy, good compromise
LLM Why
 => Performance
 => Applied in healthcare

LLM How? 
  => First fast, self-supervised
  => Second phases, adaptability
    => in context learning (RAG, prompt eng, etc)
    => fine tuning
      => efficient
      => RL

Small LLM what
  => not a clear defintiion, as "for big data"
  what small is?
  => but they have some interesting application

Small LLM where
  Personalalized heatlcare:
    A,b,c

Use Case => Chatbot for hypertensive patients:
  Requirements: 
  Validation (two phase)
    => automatic
    => by physicians

  First test:
    => small LLM have similar performance w.r.t. big for simplier task with go

SLM in Common wears perspective:
- Combine my expertise

SLM: cooperative
=> SLM can be used collaborating with other SLM to reach a collective performamnce
e.g., physician during 
How?
  => coordination
  => communication
  => Orchestration

*/

= Introduction

== LLM -- What
- A *Language model* is a model which is able to generate text, given a prompt.

#align(
  center,
  block[
    #image("figures/llm-nutshell.png", width: 80%)
  ]
)

- Training #fa-arrow-right() #h(0.4em) large corpora of text in a *self-supervised* way.
- Two main alternatives:
  - Large and Closed #fa-arrow-right()#h(0.4em)  best performing, but privacy concerns (e.g., GPT-\*, Claude, Gemini, etc.),
  - Small and Open #fa-arrow-right() #h(0.4em) best for privacy, good compromise (e.g., LLaMa, Gemma, Phi, etc.)

== LLM -- Why
- *Performance*: LLMs are the state-of-the-art reaching (super-)human performance in many tasks.
- *Generalist*: LLMs can be adapted to many tasks, including educational, healthcare, etc.

#components.side-by-side[
    #image("figures/med-palm-example.png",)
][
    #image("figures/copilot.png",)
][
    #image("figures/notebook-lm.png",)
]

== LLM -- How
- A *generalist* (or _foundational_) *model* is trained first, then adapted to a specific task.

#align(center)[
  #image("figures/shift.png", width: 50%)
]

- Adaptation can be done in two phases:
  - *In-context* learning #fa-arrow-right() #h(0.4em) _RAG_, Prompt Engineering, etc.
  - *Fine-tuning*  #fa-arrow-right() #h(0.4em)  Parameter-efficient fine-tuning, RLHF, etc.

== Retrival Augmented Generation (RAG) -- one slide summary
#align(center)[
  #image("figures/RAG.png")
]

== Small LM -- What
- We call these models "small" because they have fewer parameters than larger LLMs.
  - They can often run on a single GPU or modest hardware.
  - "Small" is relative; typical parameter counts may range from a *few hundred million* to a *few billion*.

- Several open source projects
  - *LLaMa* 3.2 (Meta)#footnote[https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/]: 1 to 3 billion parameters, showing good performance on many tasks and they are compatible with smarhgpone
  - *Gemma* 2 (Google)#footnote[https://ai.google.dev/gemma?hl=it]: 2 to 9 billion parameters
  - *Phi* (meta)#footnote[https://news.microsoft.com/source/features/ai/the-phi-3-small-language-models-with-big-potential/]: 1 to 3 billion parameters

== Small LM -- Why 
- *Privacy*: Smaller models can be fully trained on sensitive data *in-house*, protecting _confidentiality_.
- *Efficiency*: Reduced _compute_ and _memory requirements_ lower costs and _energy_ usage.
- *Real-time*: _Faster_ inference speeds enable on-device, _immediate_ predictions and responses.

#align(center)[
=== #h(0.2em) Kind of a _manifesto_ for *Edge AI*
]

#align(
  center,
  block[
    #image("figures/llama3.2b.png", width: 40%)
  ]
)

== Small LM -- Where


- *Coding*
  - CodeGemma#footnote[https://ai.google.dev/gemma/docs/codegemma?hl=it], CodeLLaMa#footnote[https://github.com/meta-llama/codellama], etc.
  - *Code completion*, *documentation*, *refactoring*, etc.
  - Faster and local processing.

- *Personalized healthcare*: #footnote[https://research.google/blog/advancing-medical-ai-with-med-gemini/]
  - *Diagnosis* and *treatment* recommendations,
  - *Patient monitoring* and *feedback*,
  - *Healthcare chatbots* and *assistants*.

- We made first effort in the latter, particularly focusing on the *chatbot* for *hypertensive patients*.

== Use Case: Chatbot for Hypertensive Patients

- Privacy: *Personal data* is stored *locally* and kept *secure*.
#align(center)[
  #image("figures/chatbot.png", width: 40%)
]
- Evaluation: *Two-phase validation*.
  - *Automatic* evaluation: *BLEU* score and *G-Eval*#footnote[Liu, Yang et al. “G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.” Conference on Empirical Methods in Natural Language Processing (2023).] score
  - *Physician* evaluation: Through a randomized controlled trial.

== Use Case: Chatbot for Hypertensive Patients
- LLM adaptation:
  - *Prompt engineering* (role-playing)#footnote[Magnini, M., Aguzzi, G., & Montagna, S. (2025). Open-source small language models for personal medical assistant chatbots. Intelligence-Based Medicine, 100197.] #fa-arrow-right() For models under ~3B parameters, hallucinations can be more frequent.
  - *Retrival augmented generation* (RAG)#footnote[Aguzzi, Gianluca, et al. "Applying Retrieval-Augmented Generation on Open LLMs for a Medical Chatbot Supporting Hypertensive Patients." Proceedings of the 3rd HC@ AIxIA 2024) co-located with the 23rd IAIxIA 2024, Bolzano, Italy, 27-28 November 2024. Vol. 3880. CEUR-WS. org, 2024.] from medical conversation #fa-arrow-right() Better performance (again with models over ~3B parameters), but needs a lot of data.
  - *Fine-tuning and data augmentation* #fa-arrow-right() Efficient and effective, also with smaller models
    - still an on going research


= Small LM in Common Wears Perspective

== Combine My Expertise
- As you may know, my background is more on engineering collective intelligence
  - *Multi-agent* Reinforcement Learning
  - *Aggregate* Computing 
  - *Swarm* Intellingence and Programming
- These prelimary works on chatbots were mainly to understand the potential of small LLMs.
- Now, I would like to focusing on the *cooperative* aspect of small LLMs.
  - More aligned with common wears perspective #fa-smile()

== Small LM -- Why?
- #fa-server() #h(0.2em) Edge AI: Small LM can be directly deployed in edge, particularly:
  - #fa-microchip() #h(0.2em) Edge Servers (> 7b)
  - #fa-mobile() #h(0.2em) Smartphones (1-3b)
  - #fa-clock() #h(0.2em) And wearables (< 1b)

- #fa-puzzle-piece() #h(0.2em) Multi-modality is a *key aspect* for the future of Edge AI:
  - Current approaches treat them in isolation #fa-arrow-right() #h(0.2em) *limited potential*
  - Need for integrated solutions #fa-arrow-right() #h(0.2em) *enhanced capabilities*

- #fa-users() #h(0.4em) The power of *cooperation*:
  - Multiple small LLMs working together as _intelligent agents_#footnote[Wang, Lei et al. “A Survey on Large Language Model based Autonomous Agents.” ArXiv abs/2308.11432 (2023): n. pag.]
  - Leveraging *collective capabilities*
  - Achieving *better performance* through collaboration and synergy

== Multi-Modality -- one slide summary
#align(center)[
  #image("figures/multi-modality.png")
]

== Small LM -- How?
- *Distributed RAG*#footnote[Li, Jiaxing et al. “EACO-RAG: Edge-Assisted and Collaborative RAG with Adaptive Knowledge Update.” ArXiv abs/2410.20299 (2024): n. pag.] #fa-arrow-right() #h(0.2em) Share local knowledge among multiple models for synergy and higher accuracy.
- *Orchestration* #fa-arrow-right() #h(0.2em) Manage different SLM tasks and unify outputs into coherent results.
  - Example 1: Multi-step reasoning with an aggregator SLM
  - Example 2: Scheduling tasks across multiple SLM agents
- *Communication* #fa-arrow-right() #h(0.2em) Maintain consistent data flow between models for stronger collaboration.
  - Example 1: Sharing partial solutions from one agent to another (due partial observability)
  - Example 2: Broadcasting a knowledge to all collaborating SLMs

== Small LM -- Operating Room Vision
// Vision for future operating rooms with cooperative SLMs
- #fa-user-md() #h(0.2em) *Operating Room of the Future*: Seamless integration of humans and AI
  - #fa-glasses() #h(0.3em) *Common-wears perspective*:
    - Smart glasses providing AR overlay of patient vitals and medical data
    - Smart watches monitoring physician stress levels and hand movements

  - #fa-robot() #h(0.2em) *Cooperative SLM Network*:
    - Real-time multi-modal analysis of medical procedures
    - Distributed knowledge sharing between specialized models
    - Contextual alerts and decision support
    
  - #fa-shield() #h(0.2em) *Benefits*:
    - Enhanced situational awareness
    - Reduced cognitive load on medical staff
    - Improved patient safety through redundant monitoring


== Conclusion
- Small LLMs are a promising technology for *personalized applications* 
  - Small footprint
  - Privacy
  - Efficiency
- Key message: *Cooperation* among Small LLMs can lead to *collective intelligence*