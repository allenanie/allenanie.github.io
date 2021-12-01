---
published: true
layout: post
draft: true
title: Grading Complex Interactive Coding Programs with Reinforcement Learning
---

[[paper]](https://arxiv.org/abs/2110.14615)

**Summary (TL;DR):** understanding student code is hard, playing a game is easy. In our recent NeurIPS 2021 [paper](https://arxiv.org/abs/2110.14615), we reformulate homework grading as game play and introduce the Play to Grade Challenge.

Massive Online Coding Education has reached striking success over the past decade. Enabled by the fast internet speed, improved UI design (especially with code editors embedded in a browser window), and designing of a diverse set of courses tailored towards students of different coding experiences and interest levels (i.e., "Star War-themed coding challenge," "Elsa/Frozen themed for-loop writing"), non-profit organizations such as [Code.org](http://code.org/) is able to create online coding challenges that reach 60 millions of students in the US and around the world.

However, despite having well-prepared teaching material that any student can read and work through at their own pace, a teacher is still required to provide feedback and grading to the student's work. This is one of the biggest obstacles of online coding education that prevents students from underprivileged, under-resourced communities, who often lack local computer science teachers, from working through the course materials and exercises on their own. In fact, many platforms do provide detailed feedback to student assignments — many run unit testing, adopt a fill-in-blank modular coding structure to allow automatic testing of errors in student code. Unfortunately, many of the most exciting assignments, such as games or interactive apps, are not unit testable.

<figure class="image"><a href="https://media.giphy.com/media/LhC4oRHFahkxvryfQA/giphy.gif"><img style="width:288px" src="https://media.giphy.com/media/LhC4oRHFahkxvryfQA/giphy.gif"/></a></figure>

Making a game is quite exciting for the students. [Code.org](http://code.org/) provides many game assignments in their curriculum. Students write JavaScript programs in a code editor embedded in the web browser, and when they click the "run" button, the game will become playable immediately. A game assignment is often regarded as an advanced assignment: students not only need to grasp basic concepts like if-conditionals and for-loop, but also write the physical rules of the game world — calculate the trajectories of objects, resolve inelastic collision of two objects, keep track of game states; and to deal with all of these complexities, students need to use abstraction (functions/class) to encapsulate each functionality in order to manage this complex set of logic.

<figure id="f86bea94-a8dd-431f-9d4c-d95d41b14b38" class="image"><a href="https://github.com/windweller/windweller.github.io/raw/master/images/play2grade/Untitled.png"><img style="width:1808px" src="https://github.com/windweller/windweller.github.io/raw/master/images/play2grade/Untitled.png"/></a></figure>

Providing automated feedback on the code text alone can be an incredibly hard challenge, even for introductory level computer science education. Two solutions which are only slightly different in text can be very different in their behavior. As such, approaches to grading which rely only on reading the text of a student's code end up being as complex as understanding a passage of natural language. Also, some assignments allow students to write in different programming languages such as C++ or Java or Python; which creates an additional layer of difficulty and vulnerability for exploits (i.e., *the algorithm can't be more accurate grading Java programs than grading C++ programs*). Finally, because feedback is necessary for the first student working on an assignment, not just the millionth, and new assignments are often introduced. We don't have the luxury of collecting a fixed dataset and training a state-of-the-art algorithm for one assignment only.

**To summarize**:

1. Complex interactive coding programs (often as long as 50-100 lines of code) are difficult to grade.
2. Training (gold-labeled) data are very few (usually less than a dozen)
3. Some coding assignments support multi-language submissions (Python, JavaScript, Java, C++, etc.)

## The Play to Grade Challenge

Each student program specifies a Markov Decision Process (MDP), which defines a state space, action space, reward function, and transition dynamics. Because all student programs are written for one assignment, these programs would share a lot of commonalities. Regarding each student program as an MDP, we sidestep the difficulty of reading and understanding student code text, which can be long, diverse, and written in different languages. Instead, all we need is to compare the MDP specified by the student's program (student MDP) to the teacher's solution (reference MDP) and determine **if these two MDPs are the same**.

Here are three incorrect programs and what they look like:

<div id="39a98484-3341-498a-8165-4f9dc45018b5" class="column-list"><div id="e48f000f-64bc-42c1-b0b8-46f6b3694cb4" style="width:33.333333333333336%" class="column"><figure id="ca968264-9e23-456a-a1ce-49aeb488fd38" class="image"><a href="https://media.giphy.com/media/i8ITbB6QtNS67t9dk6/giphy.gif"><img style="width:240px" src="https://media.giphy.com/media/i8ITbB6QtNS67t9dk6/giphy.gif"/></a></figure></div><div id="cf5741d6-4073-46e0-ac71-dfcccb121708" style="width:33.333333333333336%" class="column"><figure id="ee1f7ae5-303b-4346-bc9c-e25844c4bb69" class="image"><a href="https://media.giphy.com/media/JuQn32VatSaW1vFCgi/giphy.gif"><img style="width:240px" src="https://media.giphy.com/media/JuQn32VatSaW1vFCgi/giphy.gif"/></a></figure></div><div id="92286933-4d57-4e52-a64b-07b99d550cbe" style="width:33.33333333333333%" class="column"><figure id="f51a5280-67bc-4c19-b04a-2a39c49ea737" class="image"><a href="https://media.giphy.com/media/KTnWxZ3VvdyGl7PpAZ/giphy.gif"><img style="width:240px" src="https://media.giphy.com/media/KTnWxZ3VvdyGl7PpAZ/giphy.gif"/></a></figure></div></div>

Each incorrect program behaves differently from the correct program:

- One program's wall does not allow the ball to bounce on it.
- Another program's goal post does not let the ball go through.
- The last program gives the player one score point whenever it bounces on any object.

Each of these errors can be caused by programs that look very different from each other; however, they will all have the same erroneous behavior when you start playing them. We can design an algorithm that learns how to "play" with a student submission in order to understand the ways in which it might be buggy. What sets this challenge apart from any other reinforcement learning challenge is the fact that a **decision** (or "**classification**") needs to be made at the end of this agent's interaction with this MDP — the decision of whether the MDP is the same as the reference MDP or not.

In terms of the practicality of this approach — most game assignments do support a playable interface (for students to debug their program or for teachers to grade them), and many coding platforms even support direct play (such as [Code.org](http://code.org/)'s "run" button). It is entirely realistic to expect a server managed by the platforms to send states to an algorithm and receive actions as responses.

If there are bugs in the student program, an ideal agent needs to visit the game states that can **trigger bugs** — a game might have hundreds of thousands of unique states, it might very well be the case that an agent can create two completely identical trajectories (i.e., a sequence of states), one from the correct and one from the incorrect program. An ideal agent needs to produce **differential trajectories** that can be used to differentiate two MDPs. Differential trajectories must contain at least one bug-triggering state if the trajectory is produced from the incorrect MDP. We can achieve this if the teacher can prepare a few incorrect programs that represent their "best guesses" of what kind of mistakes a student might make. The agent will then learn to produce these differential trajectories on these incorrect programs.

## Recognizing Bugs

In order for the ideal agent to produce differential trajectories, we need to know which state is the bug triggering state. One of the most striking departures of the Play to Grade Challenge from past work is that a bug needs to be **identified** in our setting. Many past works focus on bugs that will crash the program runtime or can easily be caught by programmatic checkups. In our challenge, we assume bugs are all behavioral (i.e., "ball which is supposed to bounce on the wall but now piercing through it instead") — these types of behavioral anomalies are difficult to specify using checkups, and it won't crash the runtime. They can only be identified by a neural network predictive model.

<figure class="image"><a href="https://github.com/windweller/windweller.github.io/raw/master/images/play2grade/Untitled%201.png"><img style="width:528px" src="https://github.com/windweller/windweller.github.io/raw/master/images/play2grade/Untitled%201.png"/></a></figure>

This additional layer of difficulty is non-trivial and is often called a chicken-and-egg problem. If the classifier can perfectly identify bug states, the Play to Grade Challenge is reduced to a simple RL task that can be solved by any RL algorithm. If the RL agent is perfect and can produce differential trajectories from two MDPs, then the Play to Grade Challenge is reduced to a supervised learning task, where the classifier can be trained on perfectly labeled trajectories.

This requires an intimate collaboration between a **supervised learning classifier** and a **reinforcement learning agent**. Though both start out behaving randomly and are very bad at their job, we propose **collaborative reinforcement learning**, where we use a random agent to produce a dataset of trajectories from the correct and broken MDP to teach the classifier. Then the classifier would assign a score to each state indicating how much the classifier believes the state is a bug-triggering state. We use this score as reward and train the agent to reach these states as often as possible for a new dataset of trajectories to train the classifier.

<figure class="image"><a href="https://github.com/windweller/windweller.github.io/raw/master/images/play2grade/Untitled%202.png"><img style="width:480px" src="https://github.com/windweller/windweller.github.io/raw/master/images/play2grade/Untitled%202.png"/></a></figure>

## What Kind of Classifier Is Better?

Building off of our MDP notation, we propose HoareLSTM and Contrastive HoareLSTM classifiers that learn to approximate the transition dynamics and reward functions of the MDPs (the name is inspired by the pre-action-post condition [Hoare triple](https://en.wikipedia.org/wiki/Hoare_logic)). We also test out other classifiers such as a variational auto-encoder, which learns to memorize all the states in the correct MDP and a noisy supervised learning fully-connected neural network classifier.

In a toy environment where the agent drives a car in a 2D plane, the simple setting is that whenever the agent drives the car into the outer rim of this plane (area between the boundary and red dotted line), a bug is triggered. The bug "behavior" that needs to be recognized by the classifier is: when a bug-state is reached, the car's physical movement is altered, resulting in back-and-forth movement around the same location – reflecting the idea that the car is "stuck." The bug classifier needs to recognize the resulting states (position and velocity) of the car being "stuck." In this setting, there is only one type of bug. In the later section, when we grade homework Bounce, we will have a diverse set of bug behaviors that the classifier needs to capture, therefore more difficult and requiring classifiers that can solve this toy environment perfectly. Most classifiers do well when the agent only drives a straight line (single-direction agent). However, when the agent randomly samples actions at each state (a typical requirement for RL agents), simpler classifiers can no longer differentiate bug and non-bug states with high accuracy.

<figure class="image"><a href="https://github.com/windweller/windweller.github.io/raw/master/images/play2grade/Untitled%203.png"><img style="3406px" src="https://github.com/windweller/windweller.github.io/raw/master/images/play2grade/Untitled%203.png"/></a></figure>

If we increase this setting to be slightly harder — where the agent needs to drive the car into box areas marked by red dotted lines, we can test if collaborative reinforcement learning improves both the RL agent and the classifier. To our surprise, by running one round of collaborative training (CT Round 1 in the figure), every classifier becomes better at classifying bug vs. non-bug states. By using the badly trained classifier from the initial round, even with potentially incorrect markings on states, the agent still learned to visit bug states, which in turn created a better dataset to train a better classifier. The improvement is substantial. Variational auto-encoder started only with 32% precision but increased to 74.8% precision after 2 rounds of collaborative training.

<figure class="image"><a href="https://github.com/windweller/windweller.github.io/raw/master/images/play2grade/Untitled%204.png"><img style="width:3434px" src="https://github.com/windweller/windweller.github.io/raw/master/images/play2grade/Untitled%204.png"/></a></figure>

If we take a closer look at the trajectories produced by the agent, we can see that the agent quickly learns to only visit the states that are potentially bug-triggering even after 1 round of collaborative training.

<figure class="image"><a href="https://github.com/windweller/windweller.github.io/raw/master/images/play2grade/Untitled%205.png"><img style="width:3276px" src="https://github.com/windweller/windweller.github.io/raw/master/images/play2grade/Untitled%205.png"/></a></figure>

## Grading Bounce

We collect a large dataset from [Code.org](http://Code.org) coding assignment Bounce. Our dataset is compiled of **453,211** students, who wrote a solution to the Bounce assignment. In total, there are **711,274** submissions, where **73,611** unique programs were submitted. We provide a simulator and all of student's programs [here](https://github.com/windweller/play-to-grade/).

<figure class="image"><a href="https://media.giphy.com/media/LhC4oRHFahkxvryfQA/giphy.gif"><img style="width:288px" src="https://media.giphy.com/media/LhC4oRHFahkxvryfQA/giphy.gif"/></a></figure>

We train our agent and classifier on 10 broken programs that we wrote without looking at any of the student's submissions. The 10 programs contain bugs that we "guess" to be most likely to occur, and we use it to train 10 agents that learn to reach bug states in these 10 programs. This means that in our training dataset, we have 1 correct program and 10 broken programs. Even with only 11 labeled programs, our agent and classifier can get **99.5%** precision at identifying a bug program and **93.4-94%** accuracy overall. This demonstration shows the type of promise reformulating code assignment grading as the Play to Grade challenge will bring.

<figure class="image"><a href="https://github.com/windweller/windweller.github.io/raw/master/images/play2grade/Untitled%206.png"><img style="width:528px" src="https://github.com/windweller/windweller.github.io/raw/master/images/play2grade/Untitled%206.png"/></a></figure>

## What is Next?

We started this project by making the argument that sometimes it is far easier to grade a complex coding assignment that might have 50-100 lines of code, not by looking at the code text but by playing it. Using Bounce, we successfully demonstrated that in the simple task of identifying if a program has a bug or not (a binary task, nonetheless), we are able to achieve a striking accuracy with only 11 labeled programs.

### Multi-label Bounce

However, the journey does not end here. Feedback is not just grading, and grading is not just a pass/fail binary response. What is special about our Bounce dataset is that our gold labels actually contain all the errors the program has. If the algorithm can identify ***which bug*** is in the student's program and provide that information as feedback, the student can then debug their program with additional information. This is not addressed in our current work but is a promising direction for future algorithms to solve.

| Program | Distribution Label | Binary Error Label | Multi-Error Label | Submission Count |
| --- | --- | --- | --- | --- |
| {"when ball hits paddle": ["bounce"], ...} | Body | Correct | [] | 145636 |
| {"when ball hits paddle": ["score point"], ...} | Tail | Broken | ["whenPaddle-illegal-incrementPlayerScore", ...] | 3955 |
| ... | ... | ... | ... | ... |

### More than One Correct Solution

Oftentimes, students create solutions that are creative. Understanding and rewarding creativity is the major challenge in AI for education. Though we didn’t use the Bounce dataset to focus on the problem of understanding creativity, our work opens up an interesting angle to address this very hard challenge. In this work, we identified several unique approaches that we believe will be useful in recognizing creativity. The idea of play-to-grade could help identify the difference between a truly broken solution and one that is playable but different from the reference solution.

## For Educators

We are interested in collecting a suite of interactive coding assignments and creating a dataset for future researchers to work on this problem. Feel free to reach out to us and let us know what you would consider as important in grading and giving students feedback on their coding assignments!

## Acknowledgements

Many thanks to Emma Brunskill, Chris Piech for their guidance on the project. Many thanks to Mike Wu, Ali Malik, Yunsung Kim, Lisa Yan, and Tong Mu for their discussion and feedback. Special thanks to [code.org](http://code.org/), and Baker Franke, for many years of collaboration and generously providing the research community with data. Thanks to Stanford Hoffman-Yee Human Centered AI grant for supporting AI in education.