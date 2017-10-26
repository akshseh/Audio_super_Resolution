from pandas_ml import ConfusionMatrix

# transition and emission rate
t1 = 0.00001
e1 = t1

def applyViterbi(uniqTags, test_set, transitionList, emissionList, freqOfWords_mod):
    bigramProbList = []
    bigramTagsList = []

	with open('predict_out.txt', 'w') as f:
		for i, val in enumerate(test_set):
			hmmProbTagList = {}
			flag = 0
			# If unseen word in test set, make this word unknown
			if not val[0] in freqOfWords_mod:
				flag = 1
			# Get emission prob and transition prob
			for tag in uniqTags:
				if flag == 1:
					# the word is new (not seen in training set)
					if ('<UNK>', tag) in emissionList:
						emission = emissionList[('<UNK>', tag)]
				else:			
					emission = emissionList[(val[0], tag)]
				# Set transitionProb as 1 if end of sentence or initial start
				if (val[0] == ".") or (i == 0):	
					transition = 1
					bigramProb = 1
				else:
					transition = transitionList[(bigramTagsList[i-1], tag)]
					bigramProb = bigramProbList[i-1]
				# Generating the hmmProbability for the tags
				hmmProbTagList[tag] = emission * transition * bigramProb
			# Find the max probable tags
			hmmProb = max(hmmProbTagList.values())
			hmmTag = list(filter(lambda t: t[1]==hmmProb, hmmProbTagList.items()))[0][0]
			# Add to the tag list
			bigramTagsList.append(hmmTag)
			bigramProbList.append(hmmProb)
			
			f.write(test_set[i][0] + '\t' + str(hmmTag) + '\n')

			# Each word with '.', place a new line
			if test_set[i][0] == '.':
				f.write('\n')
	f.close()

def calcBigram(training_set_mod):
	# Create list of tag-tag bigrams and tag-words bigram
	tagtagBigram = dict()
	tagWordsBigram = dict()

	for i, val in enumerate(training_set_mod):
		if i == 0 or i == len(training_set_mod)-1:
			continue
		# For tag to tag bigrams
		setv = (training_set_mod[i][1], training_set_mod[i+1][1])
		if tagtagBigram.get(setv) is None:
			tagtagBigram[setv] = 1
		else:
			tagtagBigram[setv] += 1
		# For tag & word bigrams
		if tuple(val) not in tagWordsBigram:
			tagWordsBigram[tuple(val)] = 1
		else:
			tagWordsBigram[tuple(val)] += 1
	return tagtagBigram, tagWordsBigram

def countTagFrequency(training_set_mod):
	freqOfTags = dict()
	# Find frequency of each tags
	for i, val in enumerate(training_set_mod):
		# Add the count of tags to the frequency of tags
		if val[1] not in freqOfTags:
			freqOfTags[val[1]] = 1
		else:
			freqOfTags[val[1]] += 1
	return freqOfTags

def countFreqOfWords(training_set):
	freqOfWords = dict()
	# Calculate the frequency of words in training set
	for i, val in enumerate(training_set):
		# Check if the word already exist
		if val[0] not in freqOfWords:
			freqOfWords[val[0]] = 1
		else:
		# If already exists add 1 to it
			freqOfWords[val[0]] += 1
	return freqOfWords

def countUniqTags(training_set):
	# Count Unique tags from the training set
	uniqTags = set()
	for i, val in enumerate(training_set):
		# Add to the set uniqTags for each new tag
		if val[1] not in uniqTags:
			uniqTags.add(val[1])
	return uniqTags

def handlingUNK(training_set, freqOfWords):
	for i, val in enumerate(training_set):
		# If word occurs once, then replace with <UNK>
		if freqOfWords[val[0]] <= 1:
			val[0] = '<UNK>'
			freqOfWords[val[0]] += 1
	return training_set, freqOfWords

def hmm_train_tagger(freqOfWords_mod, tagtagBigram, tagWordsBigram, uniqTags, tagFrequencyList, training_setSize):
	# Create transition list
	transitionList = dict()
	for tags in tagtagBigram:
		if not tags in transitionList:
			# Create transition matrix for each pair of tags
			transitionList[tags] = float(t1 * 1.0 + float(tagtagBigram[tags]))/(t1 * 1.0 + float(tagFrequencyList[tags[1]])/float(training_setSize))
	for i in uniqTags:
		for j in uniqTags:
			if not (i,j) in transitionList:
				transitionList[(i,j)] = float(t1 * 1.0)/(((t1 * 1.0)  + float(tagFrequencyList[i])/float(training_setSize)))
	# Create emission list
	emissionList = dict()

	for wordTagPair in training_set:
		if not tuple(wordTagPair) in emissionList:
			emissionList[tuple(wordTagPair)] = float(e1 * 1.0 + (float(tagWordsBigram[tuple(wordTagPair)]))/(float((e1 * 1.0)  + (float(freqOfWords_mod[wordTagPair[0]])))))

	for i in uniqTags:
		for wordTagPair in training_set:
			if not (wordTagPair[0],i) in emissionList:
				emissionList[(wordTagPair[0], i)] = float(e1 * 1.0)/((e1 * 1.0) + (float(freqOfWords_mod[wordTagPair[0]])/float(len(freqOfWords_mod))))
	
	return transitionList,emissionList

def trainingHMM(training_set):
	# Count of words from training data
	freqOfWords = countFreqOfWords(training_set)
	# Extract unique tags from training data 
	uniqTags = countUniqTags(training_set)
	# Add a value of 0 for key '<UNK>'
	freqOfWords['<UNK>'] = 0
	training_set_mod, freqOfWords_mod = handlingUNK(training_set, freqOfWords)
	# Count tag frequency
	tagFrequencyList = countTagFrequency(training_set_mod)
	# Calculate bigram list
	tagtagBigram, tagWordsBigram = calcBigram(training_set_mod)
	# Calculate transition and emission probability
	transitionList,emissionList = hmm_train_tagger(freqOfWords_mod, tagtagBigram, tagWordsBigram, uniqTags, tagFrequencyList, len(training_set)-1)
	# Decoding and Apply viterbi
	applyViterbi(uniqTags, testing_set, transitionList, emissionList, freqOfWords_mod)
	# Evaluation Script
	ourPredict = [line.rstrip('\n') for line in open('predict_out.txt')]
	samplePredict = [line.rstrip('\n') for line in open('predict_out.txt')]
	# Our Predictions
	predictSet = []
	for eachPair in ourPredict:
		if eachPair:
			predictSet.append(eachPair.split()[1])
	# Sample Set
	sampleSet = []
	for eachPair in samplePredict:
		if eachPair:
			sampleSet.append(eachPair.split()[1])
	#confusion matrix
	cm = ConfusionMatrix(sampleSet, predictSet)
	print cm

#lines from training and testing data
linesTraining = [line.rstrip('\n') for line in open('Training_set.txt')]
linesTesting = [line.rstrip('\n') for line in open('Testing.txt')]

# Create Training Data and Testing Set from input file
training_set = list()
testing_set = list()

for eachPair in linesTraining:
	if eachPair:
		training_set.append(eachPair.split())
for eachPair in linesTesting:
	if len(eachPair) != 1:
		testing_set.append(eachPair.split())

trainingHMM(training_set)


