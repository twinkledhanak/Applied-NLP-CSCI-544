# Importing all libraries
import re
import json

# Constants
input_file_path = "data/train"
dev_file_path = "data/dev"
test_file_path = "data/test"
output_fle_path = "hmm.json"
predicted_file_path = "greedy.out"
UNKNOWN = "<unk>"
start_tag = "START"
end_tag = "END"
threshold = 1 # Threshold for rare words 

###########################################################################################################################
# Function to sort dictionary in desc order
def orderDictionary(d, reverse = False):
  return dict(sorted(d.items(), key = lambda x: x[1], reverse = reverse))

def create_vocabulary(dictionary):
# We removed punctuations to increase accuracy
    train_file = open(input_file_path, "r+")
    for line in train_file:
        # If line is not a blank line, do further processing  
        if(line.strip()):
            line = line.strip()
            word = line.split("\t")[1]
            if word in dictionary:
                dictionary[word] = dictionary[word] + 1
            else:
                dictionary[word] = 1
            

    # Order the dictionary in descending order
    sorted_dict = orderDictionary(dictionary, True)

    unk_count = 0
    for key in list(sorted_dict.keys()):
        if (sorted_dict[key] <= threshold):
            del sorted_dict[key]
            unk_count = unk_count + 1

    return sorted_dict ,unk_count        
    
# Function to write Vocab output to a file
def create_vocab_file(sorted_dict,unk_count):
    output_file = open('vocab.txt','w')
    unk_entry = "<unk>"+'\t'+"0"+'\t'+str(unk_count)+'\n'
    output_file.write(unk_entry)

    count = 1
    for key in list(sorted_dict.keys()):
        record = str(key)+'\t'+str(count)+'\t'+str(sorted_dict[key])+'\n'
        output_file.write(record)
        count = count + 1

    output_file.close()

# Function to write probabilities to a file
def create_prob_json(transition,emission):
    op = {
        "transition" : transition,
        "emission" : emission
    }
    with open(output_fle_path, "w") as outfile:
        json.dump(op, outfile)
    

###########################################################################################################################

# Function to calculate data for transmission and emission probs
def calculate_data_for_prob(line):
    end_of_line = True
    for i in range(0,len(line)):
        # If line is not a blank line, do further processing 
        current_state = "" 
        next_state = ""

        if(line[i].strip()):
            line[i] = line[i].strip()
            word = line[i].split("\t")[1] 
            tag = line[i].split("\t")[2] 
        

            ## Counting the no of pos tags in document overall
            if tag in postag_dict:
                postag_dict[tag] = postag_dict[tag] + 1
            else:
                postag_dict[tag] = 1

            ## Counting the no of state transitions in every sentence key:(AtoB) , value: count 
            current_state = tag

            if (end_of_line):
                # # We have to know no of times a word was at the start of a sentence
                
                state_key = "start"+"to"+tag
                if state_key in state_dict:
                    state_dict[state_key] = state_dict[state_key] + 1
                else:
                    state_dict[state_key] = 1 
                end_of_line = False

            # If we're at last line, skip and handle state transition
            if( (i+1) != len(line)):
                #sprint(line)
                l = line[i+1].strip().split("\t")
                if (len(l) >= 3):
                    next_state = l[2]
                else:
                    next_state = 'end'     

        else :
            # line is blank
            end_of_line = True    

        

        # Create a dict to keep count between state transitions
        state_key = current_state+"to"+next_state
        if state_key in state_dict:
            state_dict[state_key] = state_dict[state_key] + 1
        else:
            state_dict[state_key] = 1 

        # Create a dict to keep count between states and observations (words)
        obs_state_key = current_state+"to"+word
        if obs_state_key in obs_state_dict:
            obs_state_dict[obs_state_key] = obs_state_dict[obs_state_key] + 1
        else:
            obs_state_dict[obs_state_key] = 1     
        
    return postag_dict,state_dict


def calculate_emission_prob_new(file_path,postag_dict,obs_state_dict,vocab):
    emission = dict()

    # Since we need the words again, we use file path
    fileVar = open(file_path,"r+")
    line = fileVar.readlines()

    for i in range(0,len(line)):
        if (line[i].strip()):
            # Line is not blank
            word = line[i].strip().split("\t")[1]

            # Create entries in emission matrix , s=NN , x = Pierre
            for i in postag_dict:
                emission_key= "("+i+","+word+")"
                state_key = i+"to"+word

                if word not in vocab:
                    word = UNKNOWN

                if state_key not in obs_state_dict:
                    obs_state_dict[state_key] = 0

                emission[emission_key] = obs_state_dict[state_key]/postag_dict[i]    
                
    print("Size of Emission dict: ",len(emission))
    return emission


def greedy_decoding(file_path,vocab,transition, emission):
    fileContent = open(file_path, "r+")
    line = fileContent.readlines()
    correct_counter = 0
    actual_lines = 0
    file_index_counter = 1
    predicted_tag = "" 
    for i in range(0,len(line)):
        if(line[i].strip()): # If line is not blank
            # For every line -
            #print("Previous predicted tag: ",predicted_tag)
            actual_lines = actual_lines + 1
            word = line[i].strip().split("\t")[1]
            if word not in vocab: 
                    word = UNKNOWN
            current_tag = line[i].strip().split("\t")[2]
            T = 0
            # Calculate transmission prob for first word
            if i == 0:
                transition_key = "(start,"+current_tag+")"
            else:
                transition_key =  "("+predicted_tag+","+current_tag+")"  

            if transition_key in transition:    
                    T = transition[transition_key] 
            
            # For all distinct 45 tags, calculate prob and select the max one
            # We select distinct tags from train file only
            max_prob = -1
            
            for key in postag_dict:
                
                E = 0
                
                # Calculate emission prob using emission dictionary
                emission_key = "("+key+","+word+")"

                if emission_key in emission:
                    E = emission[emission_key] 
                
                 
                prob = T * E
                if (max_prob < prob):
                    max_prob = prob
                    predicted_tag = key 

                #print("Key: ",key," ,Trans_key: ",transition_key," ,Trans_value: ",T," , Emit_key: ",emission_key," ,Emit_value: ",E," ,Current_prob: ",prob)
            #print("Max probability: ",max_prob," Current value: ",predicted_tag)
            

        if predicted_tag == current_tag:
                correct_counter = correct_counter + 1
        else:
            file_index_counter = 1    

    return correct_counter , actual_lines  



def calculate_transition_prob_new(postag_dict,state_dict):
    # We have to calculate values of transition using post-tag values
    transition = dict()

    #  We have to handle start state separately
    for i in postag_dict:
        transition_key = "("+"start"+","+i+")"
        state_key = "startto"+i
        if state_key not in state_dict:
            state_dict[state_key] = 0
        transition[transition_key] = state_dict[state_key]# not divding by /postag_dict[i] as we do not have postag_dict[start]
        #print("Dictionary: ",transition_key," ,value: ",transition[transition_key],"state key: ",state_key," State dict value: ",state_dict[state_key]," postag count: ",postag_dict[i])

    
    # For rest of the tags, i = NNP , J = CD
    for i in postag_dict:
        for j in postag_dict:
            #print("i: ",i," j: ",j)
            transition_key = "("+i+","+j+")"
            state_key = i+"to"+j
            if state_key not in state_dict:
                state_dict[state_key] = 0
            transition[transition_key] = state_dict[state_key]/postag_dict[i]
            #print("Dictionary: ",transition_key," read as ",j,"|",i," value: ",transition[transition_key]," state key: ",state_key," State dict value: ",state_dict[state_key]," postag count: ",postag_dict[i])

    #print("Post tag dict: ",postag_dict)        
    print("Size of Transition dict : ",len(transition))
    return transition

def get_transition_emission_product(transition,emission,transition_key,emission_key):
    T = 0
    E = 0
    if transition_key in transition:
        T = transition[transition_key]      
    if emission_key in emission:
        E = emission[emission_key]   
    return T * E 

def greedy_decoding_values(file_path,predicted_file_path,vocab,transition, emission):
    fileVar = open(file_path,"r+")
    outputFile = open(predicted_file_path,"w") 
    line = fileVar.readlines()
    file_index_counter = 1
    predicted_tag = "" 
    max_prob_start = -1
    max_prob = -1
    for i in range(0,len(line)):
        if (line[i].strip()):
            word = line[i].strip().split("\t")[1]

            if word not in vocab: 
                    word = UNKNOWN

            
            
            # ONLY for first state
            if i == 0:
                for tag in postag_dict:
                    # Calculate Transition prob with every state and emission with given word
                    transition_key = "(start"+","+tag+")"
                    emission_key = "("+tag+","+word+")"

                    prob = get_transition_emission_product(transition,emission,transition_key,emission_key)
                    
                    if (max_prob_start < prob):
                        max_prob_start = prob 
                        predicted_tag = tag
                #print("First tag we predicted is : ",predicted_tag)  
                
            else:
                for tag in postag_dict:
                    transition_key = "("+predicted_tag+","+tag+")"
                    emission_key = "("+tag+","+word+")"
                    prob = get_transition_emission_product(transition,emission,transition_key,emission_key)
                    if (max_prob < prob):
                        max_prob = prob
                        predicted_tag = tag

                #print("Next tag we predicted is : ",predicted_tag)  
                

            record = str(file_index_counter) + "\t"+ line[i].strip().split("\t")[1] + "\t" + predicted_tag + '\n'
            outputFile.write(record) 
            file_index_counter = file_index_counter +1    
            
                     

        else:
            file_index_counter = 1
            outputFile.write('\n') 

#############################################################

# Read the training data file
train_file = open(input_file_path, "r+")
line = train_file.readlines()

# Keep all word counts in a dictionary
dictionary = dict()
postag_dict = dict() # {'pos_tag': frequency in train doc}
state_dict = dict() # {'(state A, state B) : no of times this transition was found'}
obs_state_dict = dict() # {'(state, obs) : no of times this transition was found'}

print("Task 1 started...")

# Task1: Create Vocabulary
sorted_dict, unk_count = create_vocabulary(dictionary) # This vocab file has punctuations and special characters too
create_vocab_file(sorted_dict,unk_count)
print("Vocabulary size is: ",len(sorted_dict))
print("Vocabulary file is created. Task 1 complete.")

# Task2: Generate Transition and Emission probabilities
print("Task 2 started...")
postag_dict,state_dict = calculate_data_for_prob(line)
transition = calculate_transition_prob_new(postag_dict,state_dict)
emission = calculate_emission_prob_new(input_file_path,postag_dict,obs_state_dict,sorted_dict)
create_prob_json(transition,emission)
print("File hmm.json with Transition and Emission probabilities created. Task 2 complete.")

# Task3.1: Greedy algorithm for HMM
print("Task 3 started...")
correct_counter , actual_lines = greedy_decoding(dev_file_path, sorted_dict, transition, emission)
print("For Dev data, Correct predicted tags: ",correct_counter," ,Total no. of tags: ",actual_lines," ,Greedy HMM Accuracy: ",(correct_counter/actual_lines)*100)

# Task3.2 Run Greedy on Test data
print("Now Predicting parts-of-speech for test data using Greedy HMM.")
greedy_decoding_values(test_file_path,predicted_file_path,sorted_dict,transition, emission)
print("File greedy.out created. Task 3 complete.")



