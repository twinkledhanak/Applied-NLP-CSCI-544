Project Title
- HW2 CSCI544 for HMM and Viterbi Algorithm for POS tagging

Description
- This project has code and output files for Greedy HMM and Viterbi algorithms, along with other tasks for HW2


Author
- Twinkle Dhanak (5150891285)


Getting Started

###############################################
A. Libraries
###############################################
1. Python3.6
2. re and json libraries. We have not used any other libraries

###############################################
B. Program Input files
###############################################

1. data/train - This file is used to generate vocabulary to be used by both models, HMM and Viterbi
2. data/dev - We evaluate the performance of both models on this file and determine accuracy
3. data/test - We predict the POS tags for sentences in this file

###############################################
C. Program Code files
###############################################

1. HW2_Greedy_Twinkle.py - This file caters to following:
- Generate vocab.txt (Task1)
- Generate hmm.json that has transmission and emission probabilities (Task2)
- Runs Greedy HMM on dev data and determines accuracy (Task3)
- Generates greedy.out file which has predicted POS tags for test data (Task3)

2. HW2_Viterbi_Twinkle.py - This file caters to following:
- Runs Viterbi on dev data and determines accuracy (Task4)
- Generates viterbi.out file which has predicted POS tags for test data (Task4)

###############################################
D. How to run the code files
###############################################

1. Keep all the data files - train , dev and test under same folder -> data
2. Ensure that the file paths are data/train , data/dev and data/test as they have been hardcoded in the program
3. Keep files HW2_Greedy_Twinkle.py and HW2_Viterbi_Twinkle.py outside 'data' directory
4. To execute , run below commands - 
python3 HW2_Greedy_Twinkle.py
python3 HW2_Viterbi_Twinkle.py


###############################################
E. Outputs
###############################################

Following files will be generated :
1. vocab.txt
2. hmm.json
3. greedy.out
4. viterbi.out

###############################################
F. Additional files
###############################################

1. PDF report
