# Importing all libraries
import re
import json

# Constants
input_file_path = "data/train"
dev_file_path = "data/dev"
test_file_path = "data/test"
output_fle_path = "hmm.json"
predicted_file_path = "viterbi.out"
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
    unk_entry = UNKNOWN+'\t'+"0"+'\t'+str(unk_count)+'\n'
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
    line.append('\n')
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
                
                state_key = start_tag+"to"+tag
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
                    next_state = end_tag    

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
        
    #print("Obs state dict: ",obs_state_dict)
    #print("Post tag counts: ",postag_dict)
    #print("State transitions: ",state_dict)
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
                # if word not in vocab:
                #     word = UNKNOWN

                emission_key= "("+i+","+word+")"
                state_key = i+"to"+word

                if state_key not in obs_state_dict:
                    obs_state_dict[state_key] = 0

                # An attempt at Laplace smoothing for Emission values so we better estimate the probabilities of unknown tags
                emission[emission_key] = (obs_state_dict[state_key]+1) /(postag_dict[i] + len(postag_dict) )  
                #print("Dictionary: ",emission_key," ,value: ",emission[emission_key],"state key: ",state_key," State dict value: ",obs_state_dict[state_key]," postag count: ",postag_dict[i])
                

  
    #print("Emission dictionary: ",len(emission))
    return emission


def calculate_transition_prob_new(postag_dict,state_dict):
    # We have to calculate values of transition using post-tag values
    transition = dict()

    #We have to handle start state separately
    for i in postag_dict:
        transition_key = "("+start_tag+","+i+")"
        state_key = start_tag+"to"+i
        if state_key not in state_dict:
            state_dict[state_key] = 0
        transition[transition_key] = state_dict[state_key]# not divding by /postag_dict[i] as we do not have postag_dict[start]
        #print("Dictionary: ",transition_key," ,value: ",transition[transition_key],"state key: ",state_key," State dict value: ",state_dict[state_key]," postag count: ",postag_dict[i])

    for i in postag_dict:
        transition_key = "("+i+","+end_tag+")"
        state_key = i+"to"+end_tag
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
    #print("New Transition dict : ",len(transition))
    return transition

def get_transition_emission_product(transition,emission,transition_key,emission_key):
    T = 0
    E = 0
    if transition_key in transition:
        T = transition[transition_key]      
    if emission_key in emission:
        E = emission[emission_key]   
    return T * E 

def get_transition(transition,transition_key):
    T = 0  
    if transition_key in transition:
        T = transition[transition_key]   
    return T    

def get_emission(emission,emission_key):
    E = 0 
    if emission_key in emission:
        E = emission[emission_key]   
    return E 




def get_data_for_viterbi(file_path,vocab,postag_dict,transition,emission,key):
    fileVar = open(file_path,"r+")
    line = fileVar.readlines()
    line.append('\n') # Needed so that the else block for skip line is executed at least once
    end_of_line = True
    sentence = []
    pos = 0
    word_lst = list()
    tag_lst = list()
    sentence_list = {}
    sentence_index = 0
    correct_tag_list = {}
    no_of_lines = 0


    # Just get the sentences, list of current tags, etc 
    for i in range(pos,len(line)):
        if(line[i].strip()):
            word_lst.append(line[i].strip().split("\t")[1])
            if(key == "dev"):
                tag_lst.append(line[i].strip().split("\t")[2])
            no_of_lines = no_of_lines + 1
                    
        else:
            sentence_list[sentence_index] = word_lst
            correct_tag_list[sentence_index] = tag_lst
            sentence_index = sentence_index + 1
            word_lst = []
            tag_lst = []
            pos = i

    return sentence_list,correct_tag_list, no_of_lines

def assign_POS_tag(word):
    
    tag = UNKNOWN
    gerund = re.compile(r'.*ing$')
    past_tense_verbs = re.compile(r'.*ed$')
    singular_present_verbs = re.compile(r'.*es$')
    modal_verbs = re.compile(r'.*ould$')
    possessive_nouns = re.compile(r'.*\'s$')
    plural_nouns = re.compile(r'.*s$')
    cardinal_numbers = re.compile(r'^-?[0-9]+(.[0-9]+)?$')
    articles_determinants = re.compile(r'(The|the|A|a|An|an)$')
    adjectives = re.compile(r'.*able$')
    nouns_formed_from_adjectives = re.compile(r'.*ness$')
    adverbs = re.compile(r'.*ly$')
    nouns = re.compile(r'.*')
    if (gerund.match(word) or past_tense_verbs.match(word) or singular_present_verbs.match(word) or modal_verbs.match(word)):
        tag = "VBZ"
    if (possessive_nouns.match(word) or plural_nouns.match(word) or nouns.match(word) or nouns_formed_from_adjectives.match(word)  ):
        tag = "NNP"
    if (adjectives.match(word)):
        tag = "JJ"
    if (cardinal_numbers.match(word)):
        tag = "CD"
    if (articles_determinants.match(word)):
        tag = "DT" 
    if (adverbs.match(word)):
        tag = "RB"

    return tag           


def viterbi(sentence,correct_tag_list,vocab,postag_dict,transition,emission):
    viterbi_tags={}
    viterbi_backpointer={}   
   

    # We create a new dictionary with start and end tags
    unique_pos_tags = postag_dict
    unique_pos_tags[start_tag] = 1
    unique_pos_tags[end_tag] = 1
    
    
    correct_value_counter = 0
 
    if (len(sentence) > 0): 
        word = sentence[0]

    
    for tag in unique_pos_tags:
        if tag==start_tag:
            continue
        viterbi_backpointer[tag]=start_tag
        transition_key = "("+start_tag+","+tag+")"
        if (word in vocab):
            emission_key = "("+tag+","+word+")"
            prob = get_transition_emission_product(transition,emission,transition_key,emission_key)
            viterbi_tags[tag]=prob   # current prob
        else:
            viterbi_tags[tag]= get_transition(transition,transition_key) * 1 
        
    
    viterbi_main=[]
    backpointer_main=[]
    viterbi_main.append(viterbi_tags)
    backpointer_main.append(viterbi_backpointer)
    correct_value_counter = unique_pos_tags[start_tag] + unique_pos_tags[end_tag] + 1
    if word in vocab:
        current_best=max(viterbi_tags.keys(),key=lambda tag: viterbi_tags[tag])
    else:
        # We assign a tag for it 
        current_best=assign_POS_tag(word)
        #print("We used custom function for prediction")
    #print("Word", "'" + word+ "'", "current best two-tag sequence:", viterbi_backpointer[current_best], current_best)
        

    mylist = list()
    mylist.append(current_best)

    for index in range(1,len(sentence)):
        #print(sentence_list[s])
        curr_viterbi={}
        curr_backpointer={}
        prev_viterbi=viterbi_main[-1]

        if (sentence[index] not in vocab):
            index = index + 1

        if (index < len(sentence)):
            for tag in unique_pos_tags:
                #print("TAGS::::: ",tag)
                if tag != start_tag:
                    if sentence[index] not in vocab:
                        prev_viterbi[assign_POS_tag(sentence[index])] = 1

                    emission_key = "("+tag+","+sentence[index]+")"
                    #print("EMission key: ",emission_key)
                    #print("prev viterbi keys: ",prev_viterbi.keys())
                    
                    prev_best = max(prev_viterbi.keys(),key=lambda prevtag: \
                    prev_viterbi[prevtag] * get_transition_emission_product(transition,emission, "("+prevtag+","+tag+")" ,emission_key ) )
                    
                    curr_viterbi[tag] = prev_viterbi[prev_best] * \
                    get_transition_emission_product(transition,emission, "("+prev_best+","+tag+")" , 
                        "("+tag+","+sentence[index]+")" )
                    
                    curr_backpointer[tag] = prev_best    
                        


            current_best = max(curr_viterbi.keys(), key=lambda tag: curr_viterbi[tag])
            #print(" For rest of the sentence ")
            #print("Word", "'" + sentence[index] + "'", "current best two-tag sequence:", curr_backpointer[current_best], current_best)
            viterbi_main.append(curr_viterbi)
            backpointer_main.append(curr_backpointer)
            mylist.append(current_best)
               


    #print("Correct list: ",correct_tag_list)
    #print("Predicted: ",mylist) 
    #print("predicted list: ",prev_best," length- ",len(prev_best))

    for e1,e2 in zip(correct_tag_list,mylist):
        #print("e1: ",e1," e2:", e2)
        if (e1 == e2):
            correct_value_counter = correct_value_counter + 1

    return correct_value_counter

def viterbi_decode(fileVar,sentence,vocab,postag_dict,transition,emission):
    viterbi_tags={}
    viterbi_backpointer={}   
    file_record = ""
    index_counter = 1

    # We create a new dictionary with start and end tags
    unique_pos_tags = postag_dict
    unique_pos_tags[start_tag] = 1
    unique_pos_tags[end_tag] = 1
    
 
    if (len(sentence) > 0): 
        word = sentence[0]

    
    for tag in unique_pos_tags:
        if tag==start_tag:
            continue
        viterbi_backpointer[tag]=start_tag
        transition_key = "("+start_tag+","+tag+")"
        if (word in vocab):
            emission_key = "("+tag+","+word+")"
            prob = get_transition_emission_product(transition,emission,transition_key,emission_key)
            viterbi_tags[tag]=prob   # current prob
        else:
            viterbi_tags[tag]= get_transition(transition,transition_key) * 1 
        

    viterbi_main=[]
    backpointer_main=[]
    viterbi_main.append(viterbi_tags)
    backpointer_main.append(viterbi_backpointer)
    if word in vocab:
        current_best=max(viterbi_tags.keys(),key=lambda tag: viterbi_tags[tag])
    else:
        # We assign a tag for it ourselves since it is unknown
        current_best=assign_POS_tag(word)
    
        

    mylist = list()
    mylist.append(current_best)
    file_record = "1"+"\t" + word + "\t" + current_best + '\n'
    fileVar.write(file_record)
    
    for index in range(1,len(sentence)):
        #print(sentence_list[s])
        curr_viterbi={}
        curr_backpointer={}
        prev_viterbi=viterbi_main[-1]

        if (sentence[index] not in vocab):
            index = index + 1

        if (index < len(sentence)):
            for tag in unique_pos_tags:
                #print("TAGS::::: ",tag)
                if tag != start_tag:
                    if sentence[index] not in vocab:
                        prev_viterbi[assign_POS_tag(sentence[index])] = 1

                    emission_key = "("+tag+","+sentence[index]+")"
                    
                    prev_best = max(prev_viterbi.keys(),key=lambda prevtag: \
                    prev_viterbi[prevtag] * get_transition_emission_product(transition,emission, "("+prevtag+","+tag+")" ,emission_key ) )
                    
                    curr_viterbi[tag] = prev_viterbi[prev_best] * \
                    get_transition_emission_product(transition,emission, "("+prev_best+","+tag+")" , 
                        "("+tag+","+sentence[index]+")" )
                    
                    curr_backpointer[tag] = prev_best    
                        


            current_best = max(curr_viterbi.keys(), key=lambda tag: curr_viterbi[tag])
            #print(" For rest of the sentence ")
            #print("Word", "'" + sentence[index] + "'", "current best two-tag sequence:", curr_backpointer[current_best], current_best)
            viterbi_main.append(curr_viterbi)
            backpointer_main.append(curr_backpointer)
            mylist.append(current_best)
            index_counter = index_counter + 1
            if (sentence[index-1] not in vocab):
                file_record = str(index-1) + "\t" + sentence[index-1] + "\t" + current_best + '\n'
            file_record = str(index) + "\t" + sentence[index] + "\t" + current_best + '\n'
            fileVar.write(file_record)
            
           
    




# Read the training data file
train_file = open(input_file_path, "r+")
line = train_file.readlines()

# Keep all word counts in a dictionary
dictionary = dict()
postag_dict = dict() # {'pos_tag': frequency in train doc}
state_dict = dict() # {'(state A, state B) : no of times this transition was found'}
obs_state_dict = dict() # {'(state, obs) : no of times this transition was found'}

print("Task 4.1 started...")
# Task1: Create Vocabulary
sorted_dict, unk_count = create_vocabulary(dictionary) # This vocab file has punctuations and special characters too
postag_dict,state_dict = calculate_data_for_prob(line)
postag_dict[UNKNOWN] = 1
transition = calculate_transition_prob_new(postag_dict,state_dict)
emission = calculate_emission_prob_new(input_file_path,postag_dict,obs_state_dict,sorted_dict)
sentence_list , correct_tag_list , no_of_lines = get_data_for_viterbi(dev_file_path,sorted_dict,postag_dict,transition,emission,"dev")
correctCount = 0
for s in sentence_list:
    #print("sentence from dev: ",sentence_list[s])
    ret_value = viterbi(sentence_list[s],correct_tag_list[s],sorted_dict,postag_dict,transition,emission)
    correctCount = correctCount + ret_value
    #if ret_value < len(sentence_list[s]):
        #print("Only ",ret_value," out of ",len(sentence_list[s])," correct. CHECK: ",sentence_list[s])
print("For Dev data, Correct predicted tags: ",correctCount," Total no. of tags: ",no_of_lines," ,Viterbi Accuracy:: ",(correctCount/no_of_lines)*100)    

# Task4.2 Run Viterbi on Test data
print("Now Predicting parts-of-speech for test data using Viteri.")
sentence_list , correct_tag_list , no_of_lines = get_data_for_viterbi(test_file_path,sorted_dict,postag_dict,transition,emission,"test")
fileVar = open(predicted_file_path,"w")
for s in sentence_list:
#     #print("sentence from test: ",sentence_list[s])
    viterbi_decode(fileVar,sentence_list[s],sorted_dict,postag_dict,transition,emission)
    fileVar.write('\n')
print("File viterbi.out created. Task 4.2 complete.")
