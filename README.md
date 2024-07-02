# Bridging the Technological Gap Workshop - 2nd Edition
Spreading technological innovations in the study of cognition and behavior of human and non-human primates <br>
4th August 2024 → 10th August 2024 at the MPI EVA

[Pierre-Etienne Martin](https://www.eva.mpg.de/comparative-cultural-psychology/staff/pierre-etienne-martin/), PhD, Max Planck Institute for Evolutionary Anthropology, Leipzig, Germany <br>
[Laura Lewis](), PhD, University of California, Berkeley, CA, USA <br>
[Hanna Schleihauf](), PhD, Utrecht University, Netherlands

## Introduction

[Workshop Webpage](https://www.eva.mpg.de/comparative-cultural-psychology/events/2024-btg2/´)

Technological breakthroughs create immense opportunities to revolutionize scientific methodologies and enable answers to questions that were previously unanswerable. Disciplines studying behavior and cognition across species are struggling to keep up with the rapid technological developments. In 2022, we organized a workshop to help bridge this technological gap. Through the workshop we (a) trained 31 early career researchers in the use of cutting-edge non-invasive technologies including motion-tracking, eye-tracking, thermal imagining, and machine learning extensions, (b) developed guidelines and common standards in the use of these methods, and (c) created an online platform with a diverse network of researchers, and (d) initiated an interdisciplinary collaboration project with most of the workshop participants involved using thermal imaging with many different species. Building on the success of the 2022 event, we plan to organize this workshop again. Our goal is to continue providing opportunities for early career researchers to gain expertise in innovative technologies, build connections between interdisciplinary and international scholars, and foster the exchange of ideas and future collaborations.

## Hands-on

### Installation

Please install Anaconda and VSCode in advance.
We advise using a 64-bit computer with GPU, CUDA drivers and 100Gb of free hard disk space.

#### Anaconda

Anaconda is a virtual environment manager. Follow the installation steps from [here](https://www.anaconda.com/download/success) according to your system. We advise installation for the user only (not as admin of for the whole system).

#### VSCode and Python extension

VSCode is a code editor with many features and extensions. Follow the installation steps from [here](https://code.visualstudio.com/Download) according to your system. You may install via Anaconda Navigator.

After installation, start VSCode and navigate to the extensions tab: <br>
![VSCode extensions](attachments/extensions.png)<br>
and install the extensions named:<br>
1. Python
2. Encryptor
3. vscode-numpy-viewer

#### Try your installation

1. Start VSCode
2. Open a VSCode terminal
3. Create a conda environment called **btg** and install numpy, opencv and matplotlib
``` bash
conda create -n btg python=3.10
conda activate btg
conda install -c conda-forge opencv matplotlib numpy natsort scipy
```
4. Select conda virtual env **btg** as "Python interpreter" (F1 with search function or ctrl + shift + p)

#### Troubleshooting

1. Conda not recognized in your terminal:

    This may happen according to the way things were installed. You can either:

    - reinstall VSCode using Anaconda Navigator and do the previous steps again

    - OR use the Anaconda Prompt and enter:

        ``` bash
        conda create -n btg python=3.10
        conda activate btg
        conda install numpy
        conda install -c conda-forge opencv matplotlib
        ```
        For Windows:
        ``` bash
        where python
        ```
        For MacOS and Linux:
        ``` bash
        which python
        ```
    Then, in VSCode, select "Select interpreter path..." in "Python interpreter" (F1 with search function or ctrl + shift + p) enter enter the path given by your previous command `where python` or `which python` from the Anaconda Prompt.

    - OR alternativly, try following [these instructions](https://stackoverflow.com/questions/64170551/visual-studio-code-vsc-not-able-to-recognize-conda-command)
    
    - OR contact your IT support.

### Sessions

#### Part I - Introduction and COMPUTATIONAL BASICS

##### Monday, August 5, 2024 – Into and Building foundations: Let’s R & git

|     |     |     |
| --- | --- | --- |
| 11:30 – 12:30 | Bret Beheim | **Hands-on**  <br>“Let’s Git” - Intro to Git and GitHub |
| 12:30 – 13:30 | **Lunch Break** |     |
| 13:30 – 15:00 | Luke Maurits | **Hands-on**  <br>Intro to R and R studio |
| 15:00 – 15:30 | **Coffee Break** |     |
| 15:30 – 17:00 | Pierre-Etienne Martin | **Hands-on**  <br>Intro to Python |

#### Part II - Training in the use of new technologies to study the human and non-human animal mind

##### Wednesday, August 7, 2024 – Gaze Tracking & Pupillometry

|     |     |     |
| --- | --- | --- |
| 10:30 – 12:30 | Tomasso Ghilardi, Franceso Poli & Guilia Serino | **Hands-on**  <br>Eye Tracking with Python |
| 12:30 – 13:30 | **Lunch Break** |     |
| 13:30 – 15:00 | Tomasso Ghilardi, Franceso Poli & Guilia Serino | **Hands-on**  <br>Eye Tracking with Python |
| 15:00 – 15:30 | **Coffee Break** |     |
| 15:30 – 17:00 | Tomasso Ghilardi, Franceso Poli & Guilia Serino | **Hands-on**  <br>Eye Tracking with Python |
| 19:00 – 21:00 | **Dinner with Speakers OR City Tour** |     |

##### Thursday, August 8, 2024 – Thermal Imaging

|     |     |     |
| --- | --- | --- |
| 15:30 – 17:00 | Pierre-Etienne Martin | **Keynote + Hands-on**  <br>Bio-TIP: Bio-Signal Retrieval from Thermal Imaging Processing |

##### Friday, August 9, 2024 – Motion Tracking

|     |     |     |
| --- | --- | --- |
| 10:00 – 11:00 | Raphaelle Malassis & Rayanne Martin | **Hands-on**  <br>Beyond response times: Tracking apes’ hand trajectory on a touchscreen |
| 11:00 – 11:30 | **Coffee Break** |     |
| 11:30 – 12:30 | Tim-Joshua Andres & Arja Mentink | **Keynote**  <br>Motion-Tracking |
| 12:30 – 13:30 | **Lunch Break** |     |
| 13:30 – 15:00 | Tim-Joshua Andres & Arja Mentink | **Hands-on**  <br>Motion-Tracking |
| 15:00 – 15:30 | **Coffee Break** |     |
| 15:30 – 17:00 | Charlotte Ann Wiltshire | **Hands-on**  <br>DeepWild: application of the pose estimation tool DeepLabCut for behaviour tracking in wild chimpanzees and bonobos |

## Sponsors

We would like to thank the Joachim Herz Foundation who made the organization of this workshop possible with their financial support.

![Logo Joachim Hez Foundation](./attachments/csm_JHS_Logo_sRGB_violett-white_extern_488c57bbe9.png)
