# CERTIFICATION CHALLENGE TASKS
---

# Task 1: Articulate the problem and the user of your application  
*Hints:*
- *Create a list of potential questions that your user is likely to ask!*
- *What is the user’s job title, and what is the part of their job function that you’re trying to automate?*

**✅ Deliverables**

1. Write a succinct 1-sentence description of the problem
2. Write 1-2 paragraphs on why this is a problem for your specific user

**✅ SOLUTION**

1. One sentence description of the problem (candiates):

Parents sometimes feel lost\guilty\stuck as to how to handle\manage challenging situations with their kids
- Lost: literally they do not know what to do or who to turn to discuss those particular challenges
- Guilty: parent managed the situation, but they are not happy about what they did and feel guilt (too harsh, too permisive, and everything in between)
- Stuck: parent is doing what he learned as child, but do not like what he is doing - needs a bettwer way that speaks true to them

2. Write 1-2 paragraphs on why this is a problem for your specific user

[c] Parents will always encounter challenging situations with their kids. Kids are hardwired to test boundaries (and that’s a good thing); it’s one of the ways they discover the world and their place in it. Today’s parents were once kids themselves, and what they saw from their own parents is, in most cases, their main reference point for how to "parent." The problem is that, in multiple instances, many of today’s parents do not agree with or like what they experienced as children. They want to do things differently, but they may not be sure how to act on that desire, even if they clearly realize they want change. Worse, they may not have anyone in their close circle to talk about it with in a way that provides psicological safety. It is not easy to admit for anyone when you feel lost, guilty or stuck.

---

# Task 2: Articulate your proposed solution

*Hint:*  

- *Paint a picture of the “better world” that your user will live in.  How will they save time, make money, or produce higher-quality output?*
- *Recall the [LLM Application stack](https://a16z.com/emerging-architectures-for-llm-applications/) we’ve discussed at length*

1. Write 1-2 paragraphs on your proposed solution.  How will it look and feel to the user?
2. Describe the tools you plan to use in each part of your stack.  Write one sentence on why you made each tooling choice.  
3.a Where will you use an agent or agents?  
3.b What will you use “agentic reasoning” for in your app?  

**✅ SOLUTION**

1. Write 1-2 paragraphs on your proposed solution.  How will it look and feel to the user?

[j1] To adress these challenges, we introduce ParentALLm, a Positive Discipline Parenting Companion. The companion allows people to debrief the worries and challenges they
face with their kids, seeking advice on how to navegate all kind of situations, ranging from the very frustraiting (tears) to the happy annecdotal. Companion does not only covers those needs by properly managing/remembering every interaction and advice given, that focuses on the kids, but will also provide tools for the parent self-care and acceptance. Let's remember "Peceful parents - Happy Kids".

[c1] To address these challenges, we introduce ParentALLm, a Positive Discipline Parenting Companion chat application. This companion allows parents to debrief their worries and challenges, seeking advice on how to navigate all kinds of situations, from the deeply frustrating (tears) to the joyful and anecdotal. Parents will intract with the app via a chat UI, were multi turn chatting will take place as well as other type of inputs, such as moods at the begining and end of all interactions. The companion will not only supports parents by remembering and managing every interaction and piece of advice focused on their children, but also provides tools to promote self-care and self-acceptance for the parent. Let's remember "Peceful parents - Happy Kids".

2. Describe the tools you plan to use in each part of your stack.  Write one sentence on why you made each tooling choice.
    1. LLM --> OpenAI 4.1 mini: balance between performance, speed, and cost-effectiveness
    2. Embedding Model --> OpenAI's text-embedding-3-small is a lightweight, high-performance, perfect balance for the job
    3. Orchestration --> LanGraph - we have learned what the tool can do
    4. Vector Database --> Qdrant - we have learned what the tool can do
    5. Monitoring --> LanSmith - top notch tracing enabled by simply adding decorators
    6. Evaluation --> RAGAS, simple to setup and delivers killer value 
    7. User Interface --> React frontend is the top choice here considering the beautiful UIs that can be generated and that GenAI tools like Cursor can create those in minutes
    8. (Optional) Serving & Inference --> FastAPI (local host) due to it is simplicity while being super effective

3.a Where will you use an agent or agents?  
The agent will manage the multi turn chating app, as well as all the context data needed for that: long term memory, short term memory, RAG content, and web search tools.

3.b What will you use “agentic reasoning” for in your app?
The app will use the ReAct framework to define what tools to use to retriev meaninful information to reply to user queries.

---
# Task 3: Collect data for (at least) RAG and choose (at least) one external API

*Hint:*  

- *Ask other real people (ideally the people you’re building for!) what they think.*
- *What are the specific questions that your user is likely to ask of your application?  **Write these down**.*

**✅ Deliverables**

1. Describe all of your data sources and external APIs, and describe what you’ll use them for.
2. Describe the default chunking strategy that you will use.  Why did you make this decision?
3. [Optional] Will you need specific data for any other part of your application?   If so, explain.

**✅ SOLUTION**

1. Describe all of your data sources and external APIs, and describe what you’ll use them for.

Sources:
- RAG: Positive discipline blog post and well as pdf material exposing the core philosophy 
- Tools: tavily to perform relevant websearches 

2. Describe the default chunking strategy that you will use.  Why did you make this decision?

I will experiment with:
(a) Naive chunking retrieval (and in some cases no chunking, for exaple some blogs are quite short)
(b) Contextual compression retrieval
(c) BM25 retrieval
(d) Multi query retrieval
(e) Parent-child retrieval

3. [Optional] Will you need specific data for any other part of your application?  If so, explain.

Not sure yet.

---
# Task 4: Build an end-to-end Agentic RAG application using a production-grade stack and your choice of commercial off-the-shelf model(s)

**✅ Deliverables**

1. Build an end-to-end prototype and deploy it to a *local* endpoint

**✅ SOLUTION**

???

---
# Task 5: Generate a synthetic test data set to baseline an initial evaluation with RAGAS

**✅ Deliverables**

1. Assess your pipeline using the RAGAS framework including key metrics faithfulness, response relevance, context precision, and context recall.  Provide a table of your output results.
2. What conclusions can you draw about the performance and effectiveness of your pipeline with this information?

**✅ SOLUTION**

???

---
# Task 6: Install an advanced retriever of your choosing in our Agentic RAG application. 

**✅ Deliverables**

1. Describe the retrieval techniques that you plan to try and to assess in your application.  Write one sentence on why you believe each technique will be useful for your use case.
2. Test a host of advanced retrieval techniques on your application.

**✅ SOLUTION**

???

---
# Task 7: Assess the performance of the naive agentic RAG application versus the applications with advanced retrieval tooling

**✅ Deliverables**

1. How does the performance compare to your original RAG application?  Test the fine-tuned embedding model using the RAGAS frameworks to quantify any improvements.  Provide results in a table.
2. Articulate the changes that you expect to make to your app in the second half of the course. How will you improve your application?

**✅ SOLUTION**

???