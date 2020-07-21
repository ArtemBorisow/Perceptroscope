#include "includes.h"

class neuralNetworkFF
{
public:
	//destructor
	~neuralNetworkFF()
	{
		for (int i = 0; i < layersCount; i++)
		{
			for (int n = 0; n < neuronsCountInEachLayer[i]; n++)
			{
				delete networkGrid[i][n];
			}
			delete[] networkGrid[i];
		}
		delete[] networkGrid;

		delete[] neuronsCountInEachLayer;

		for (int i = 0; i < setsCount; i++)
		{
			delete[] trainingSetArray[i].inputs;
			delete[] trainingSetArray[i].outputs;
		}
		delete[] trainingSetArray;
	}

	//inner "neuron" class
	class neuron
	{
	public: //constructor & destructor sapce
		neuron(int synapsesAmount)
		{
			this->synapsesAmount = synapsesAmount;
			inputWeights = new float[synapsesAmount];
			for (int i = 0; i < synapsesAmount; i++) inputWeights[i] = rand() % 21 - 10; //initializing weights with random number in range of [-10;10]
		}
		~neuron()
		{
			delete[] inputWeights;
		}

	public:
		float output; //output value
		float errorDelta; //error delta on these neuron while back propagation learning
		float* inputWeights; //array of input values on these neuron
		int synapsesAmount; //amount of input values and their weights (each input value and his weight is ONE synaps)
		void compute(float* outputs) //compute own output value method
		{
			calculateNonNormalizedNeuronValue(outputs);
			activate();
		}

	private:
		float nonNormalizedNeuronValue; //argument for activation function
		void calculateNonNormalizedNeuronValue(float* outputs) //inner method for value that represents sum of all synapses values
		{
			nonNormalizedNeuronValue = 0;
			for (int i = 0; i < synapsesAmount; i++) nonNormalizedNeuronValue += outputs[i] * inputWeights[i];
		}
		void activate() //inner method for normalizing nonNormalizedNeuronValue via "activation" function
		{
			output = 1 / (1 + exp(-1 * nonNormalizedNeuronValue));
		}
	};

	//fields
	struct trainingSet
	{
		float* inputs;
		float* outputs;
	}* trainingSetArray; //data set for neural network supervised training
	int setsCount; //amount of sets in trainingSetArray
	neuron*** networkGrid; //grid made out of neurons (to acces them by layer index and index of neuron in layer)
	int layersCount; //amount of layers
	int amountOfInputNeurons; //amount of inputs for whole neural network
	int* neuronsCountInEachLayer; //amount of neurons in each layer (in each layer except input layer)

	//update methods
	void updateNetwork(std::string structure)
	{
		//calculating amount of layers and neurons in each layer
		layersCount = 0;
		for (int i = 0; i < structure.length(); i++) if (structure[i] == '/') layersCount++;
		neuronsCountInEachLayer = new int[layersCount];
		int layerIndex = -1;
		amountOfInputNeurons = 0;
		for (int i = 0; i < structure.length(); i++)
		{
			if (structure[i] == '/')
			{
				layerIndex++;
				neuronsCountInEachLayer[layerIndex] = 0;
			}
			else if (layerIndex == -1) amountOfInputNeurons++;
			else neuronsCountInEachLayer[layerIndex]++;
		}

		//creating neuron objects and making neural network grid
		networkGrid = new neuron**[layersCount];
		for (layerIndex = 0; layerIndex < layersCount; layerIndex++) networkGrid[layerIndex] = new neuron*[neuronsCountInEachLayer[layerIndex]];
		for (layerIndex = 0; layerIndex < layersCount; layerIndex++)
			for (int neuronIndex = 0; neuronIndex < neuronsCountInEachLayer[layerIndex]; neuronIndex++)
			{
				if (layerIndex == 0) networkGrid[layerIndex][neuronIndex] = new neuron(amountOfInputNeurons);
				else networkGrid[layerIndex][neuronIndex] = new neuron(neuronsCountInEachLayer[layerIndex - 1]);
			}
	}
	void updateNetwork(float* weights)
	{
		int mainIterator = 1;

		for (int layerIndex = 0; layerIndex < layersCount; layerIndex++)
			for (int neuronIndex = 0; neuronIndex < neuronsCountInEachLayer[layerIndex]; neuronIndex++)
			{
				for (int i = 0; i < networkGrid[layerIndex][neuronIndex]->synapsesAmount; i++)
				{
					networkGrid[layerIndex][neuronIndex]->inputWeights[i] = weights[mainIterator];
					mainIterator++;
				}
			}
	}
	void updateNetwork(std::string structure, float* weights)
	{
		updateNetwork(structure);
		updateNetwork(weights);
	}
	void updateNetwork(std::string structure, std::string fileName)
	{
		updateNetwork(structure);
		setAllweights_fromFile(fileName);
	}
	void setAllweights_fromFile(std::string fileName)
	{
		std::string stringFromFile = "";
		FILE* pFile = fopen(fileName.c_str(), "r");
		int weightsValuesCounter = 0;
		while (!feof(pFile))
		{
			stringFromFile += fgetc(pFile);
			if (stringFromFile[stringFromFile.length() - 1] == ';') weightsValuesCounter++;
		}
		fclose(pFile);

		float* allWeightsInFloatArray_formatedForInstallMethod = new float[weightsValuesCounter + 1];
		int mainIterator = 1;
		std::string numBuffer = "";
		for (int i = 0; i <= stringFromFile.length(); i++)
		{
			if (stringFromFile[i] == ';')
			{
				allWeightsInFloatArray_formatedForInstallMethod[mainIterator] = atof(numBuffer.c_str());
				mainIterator++;
				numBuffer = "";
				continue;
			}

			numBuffer += stringFromFile[i];
		}

		updateNetwork(allWeightsInFloatArray_formatedForInstallMethod);
		delete[] allWeightsInFloatArray_formatedForInstallMethod;
	}

	//get content methods
	float* getAllWeights()
	{
		float* allWeightsValues = new float[1]; //resulting array
		allWeightsValues[0] = 0; //these index ([0]) represents amount of synapses in whole neural network
		int mainIterator = 1;

		for (int layerIndex = 0; layerIndex < layersCount; layerIndex++)
			for (int neuronIndex = 0; neuronIndex < neuronsCountInEachLayer[layerIndex]; neuronIndex++)
			{
				allWeightsValues[0] += networkGrid[layerIndex][neuronIndex]->synapsesAmount;
				float* temp = allWeightsValues;
				allWeightsValues = new float[temp[0] + 1];
				for (int i = 0; i < mainIterator; i++) allWeightsValues[i] = temp[i];
				delete[] temp;
				for (int i = 0; i < networkGrid[layerIndex][neuronIndex]->synapsesAmount; i++)
				{
					///std::cout << "neuron in " << layerIndex << " layer, with index " << neuronIndex << " in it, his weight[" << i << "] = " << networkGrid[layerIndex][neuronIndex]->inputWeights[i] << std::endl;
					allWeightsValues[mainIterator] = networkGrid[layerIndex][neuronIndex]->inputWeights[i];
					mainIterator++;
				}
			}

		return allWeightsValues;
	}
	float** getAllSynapsesData() //in additional information stored for each synapse is available: weight value, index in neuron, neuron index in layer, layer index
	{
		float** allSynapsesData = new float*[1]; //resulting array of arrays (in each final array index represents: 0=weight_value; 1=index_in_neuron; 2=neuron_index_in_layer; 3=layer_index)
		allSynapsesData[0] = new float[4];
		allSynapsesData[0][0] = 0; //these index ([0][0]) represents amount of synapses in whole neural network
		int mainIterator = 1;

		for (int layerIndex = 0; layerIndex < layersCount; layerIndex++)
			for (int neuronIndex = 0; neuronIndex < neuronsCountInEachLayer[layerIndex]; neuronIndex++)
			{
				allSynapsesData[0][0] += networkGrid[layerIndex][neuronIndex]->synapsesAmount;

				float** temp = allSynapsesData;
				allSynapsesData = new float*[temp[0][0] + 1];
				for (int i = 0; i < temp[0][0] + 1; i++) allSynapsesData[i] = new float[4];
				for (int i = 0; i < mainIterator; i++)
					for (int n = 0; n < 4; n++) allSynapsesData[i][n] = temp[i][n];
				for (int i = 0; i < mainIterator; i++) delete[] temp[i];
				delete[] temp;

				for (int i = 0; i < networkGrid[layerIndex][neuronIndex]->synapsesAmount; i++)
				{
					///std::cout << "neuron in " << layerIndex << " layer, with index " << neuronIndex << " in it, his weight[" << i << "] = " << networkGrid[layerIndex][neuronIndex]->inputWeights[i] << std::endl;
					allSynapsesData[mainIterator][0] = networkGrid[layerIndex][neuronIndex]->inputWeights[i];
					allSynapsesData[mainIterator][1] = i;
					allSynapsesData[mainIterator][2] = neuronIndex;
					allSynapsesData[mainIterator][3] = layerIndex;
					mainIterator++;
				}
			}

		return allSynapsesData;
	}
	std::string getAllWeights_inFormatedString()
	{
		float* weightsInFloatArray = getAllWeights();
		std::string outputString = "";

		for (int i = 1; i <= weightsInFloatArray[0]; i++)
		{
			outputString += std::to_string(weightsInFloatArray[i]);
			outputString += ";";
		}

		delete[] weightsInFloatArray;
		return outputString;
	}

	//"using" method
	float* call(float* inputs, bool deleteInputs) //main call method for computing all network (if (arg2 = true) pass heap mem only)
	{
		float* outputs = new float[neuronsCountInEachLayer[layersCount - 1]];
		float* local_inputs = new float[amountOfInputNeurons];
		if (deleteInputs)
		{
			delete[] local_inputs;
			local_inputs = inputs;
		}
		else for (int i = 0; i < amountOfInputNeurons; i++) local_inputs[i] = inputs[i];

		for (int layerIndex = 0; layerIndex < layersCount; layerIndex++)
			for (int neuronIndex = 0; neuronIndex < neuronsCountInEachLayer[layerIndex]; neuronIndex++)
			{
				if (layerIndex != 0)
				{
					delete[] local_inputs;
					local_inputs = new float[neuronsCountInEachLayer[layerIndex - 1]]; //setting inputs for current layer
					for (int i = 0; i < neuronsCountInEachLayer[layerIndex - 1]; i++) local_inputs[i] = networkGrid[layerIndex - 1][i]->output;
				}
				networkGrid[layerIndex][neuronIndex]->compute(local_inputs);

				if (layerIndex == layersCount - 1) outputs[neuronIndex] = networkGrid[layerIndex][neuronIndex]->output;
			}
		delete[] local_inputs;
		return outputs;
	}

	//changing content methods
	void randomizeWeights()
	{
		for (int layerIndex = 0; layerIndex < layersCount; layerIndex++)
			for (int neuronIndex = 0; neuronIndex < neuronsCountInEachLayer[layerIndex]; neuronIndex++)
			{
				for (int i = 0; i < networkGrid[layerIndex][neuronIndex]->synapsesAmount; i++)
				networkGrid[layerIndex][neuronIndex]->inputWeights[i] = rand() % 21 - 10; //initializing weights with random number in range of [-10;10]
			}
	}
	void createTrainingSetArray(std::string ts_string) //argument "ts_string" is formated via local format protocol, value example: "0,0:0/0,1:0/1,0:0/1,1:1;" represents data set for "AND" logic function
	{
		setsCount = 1;
		for (int i = 0; i < ts_string.length(); i++) if (ts_string[i] == '/') setsCount++;
		trainingSetArray = new trainingSet[setsCount];
		for (int i = 0; i < setsCount; i++)
		{
			trainingSetArray[i].inputs = new float[0];
			trainingSetArray[i].outputs = new float[0];
		}
		std::string numBuffer = "";
		bool isInput = true;
		int tsIndex = 0; //training set index
		int localArrayIndex = 0; //index in inputs/outputs array of training set
		float* temp; //temporary buffer for inputs/outputs array
		for (int i = 0; i < ts_string.length(); i++)
		{
			///std::cout << ts_string[i] << ": ";
			if (ts_string[i] == ',' || ts_string[i] == ':' || ts_string[i] == '/' || ts_string[i] == ';')
			{
				///std::cout << "nonNumSection; ";
				localArrayIndex++;
				if (isInput)
				{
					temp = trainingSetArray[tsIndex].inputs;
					trainingSetArray[tsIndex].inputs = new float[localArrayIndex];
					for (int n = 0; n < localArrayIndex - 1; n++) trainingSetArray[tsIndex].inputs[n] = temp[n];
					trainingSetArray[tsIndex].inputs[localArrayIndex - 1] = atof(numBuffer.c_str());
					///std::cout << "trainingSetArray[" << tsIndex << "].inputs[" << localArrayIndex - 1 << "]: " << trainingSetArray[tsIndex].inputs[localArrayIndex - 1] << "; ";
				}
				else
				{
					temp = trainingSetArray[tsIndex].outputs;
					trainingSetArray[tsIndex].outputs = new float[localArrayIndex];
					for (int n = 0; n < localArrayIndex - 1; n++) trainingSetArray[tsIndex].outputs[n] = temp[n];
					trainingSetArray[tsIndex].outputs[localArrayIndex - 1] = atof(numBuffer.c_str());
					///std::cout << "trainingSetArray[" << tsIndex << "].outputs[" << localArrayIndex - 1 << "]: " << trainingSetArray[tsIndex].outputs[localArrayIndex - 1] << "; ";
				}
				delete[] temp;
				numBuffer = "";
				if (ts_string[i] != ',') localArrayIndex = 0;
				if (ts_string[i] == ':') isInput = false;
				else if (ts_string[i] == '/')
				{
					isInput = true;
					tsIndex++;
				}
			}
			else numBuffer += ts_string[i];
			///std::cout << std::endl;
		}

		//freeing memory (outdated)
		/*std::cout << "s" << std::endl;
		std::vector<float*>::iterator it;
		std::sort(vectorOfAdressesThatNeedToBeDeleted.begin(), vectorOfAdressesThatNeedToBeDeleted.end());
		it = std::unique(vectorOfAdressesThatNeedToBeDeleted.begin(), vectorOfAdressesThatNeedToBeDeleted.end());
		vectorOfAdressesThatNeedToBeDeleted.resize(std::distance(vectorOfAdressesThatNeedToBeDeleted.begin(), it));
		while (!vectorOfAdressesThatNeedToBeDeleted.empty())
		{
			delete[] vectorOfAdressesThatNeedToBeDeleted.back();
			vectorOfAdressesThatNeedToBeDeleted.pop_back();
		}*/
	}
	float learn(int epochsCount, float LR, bool logEpochInfo) //LR - learning rate
	{
		int epochsCounter = 0;
		float errorOnSetParticleParticle = 0;
		float errorOnSetParticle = 0;
		float errorOnSet = 0;
		float averageError = 0;
		float* actual_output = new float[0];

		for (int epochsIterator = 0; epochsIterator < epochsCount; epochsIterator++)
		{
			for (int setIterator = 0; setIterator < setsCount; setIterator++)
			{
				//setting errorDelta for all neurons
				delete[] actual_output;
				actual_output = call(trainingSetArray[setIterator].inputs, false);
				for (int layerIndex = layersCount - 1; layerIndex >= 0; layerIndex--)
					for (int neuronIndex = 0; neuronIndex < neuronsCountInEachLayer[layerIndex]; neuronIndex++)
					{
						if (layerIndex == layersCount - 1) //making error delta for each output neuron
						{
							for (int i = 0; i < neuronsCountInEachLayer[layerIndex]; i++)
							{
								networkGrid[layerIndex][neuronIndex]->errorDelta = trainingSetArray[setIterator].outputs[i] - actual_output[i];
								errorOnSetParticleParticle += networkGrid[layerIndex][neuronIndex]->errorDelta;
							}
							errorOnSetParticle += errorOnSetParticleParticle / neuronsCountInEachLayer[layerIndex];
							errorOnSetParticleParticle = 0;
						}
						else //making error delta for each of the remaining neurons
						{
							float local_errorDelta = 0;
							for (int i = 0; i < neuronsCountInEachLayer[layerIndex + 1]; i++) local_errorDelta += networkGrid[layerIndex + 1][i]->errorDelta * networkGrid[layerIndex + 1][i]->inputWeights[neuronIndex];
							networkGrid[layerIndex][neuronIndex]->errorDelta = local_errorDelta;
							errorOnSetParticle += errorOnSetParticleParticle / neuronsCountInEachLayer[layerIndex];
							errorOnSetParticleParticle = 0;
						}
					}
				errorOnSet += errorOnSetParticle / layersCount;
				errorOnSetParticle = 0;

				//correcting weights of all neurons
				for (int layerIndex = 0; layerIndex < layersCount; layerIndex++)
					for (int neuronIndex = 0; neuronIndex < neuronsCountInEachLayer[layerIndex]; neuronIndex++)
					{
						if (layerIndex == 0) //correcting weights of each input+1 layer neuron
							for (int i = 0; i < networkGrid[layerIndex][neuronIndex]->synapsesAmount; i++)
								networkGrid[layerIndex][neuronIndex]->inputWeights[i] += LR * networkGrid[layerIndex][neuronIndex]->errorDelta * trainingSetArray[setIterator].inputs[i] * (1 - networkGrid[layerIndex][neuronIndex]->output) * networkGrid[layerIndex][neuronIndex]->output;
						else //correcting weights of each of the remaining neurons
						{
							for (int i = 0; i < networkGrid[layerIndex][neuronIndex]->synapsesAmount; i++)
								networkGrid[layerIndex][neuronIndex]->inputWeights[i] += LR * networkGrid[layerIndex][neuronIndex]->errorDelta * networkGrid[layerIndex - 1][i]->output * (1 - networkGrid[layerIndex][neuronIndex]->output) * networkGrid[layerIndex][neuronIndex]->output;
						}
					}
			}
			epochsCounter++;
			averageError = errorOnSet / setsCount;
			errorOnSet = 0;
			if (logEpochInfo) std::cout << "Epoch index: " << epochsCounter << ", average error = " << abs(averageError) << std::endl;
		}
		delete[] actual_output;

		if (epochsCount == 1) return abs(averageError);
		else return 0;
	}
	void mutate(float MR) //MR - mutating rate
	{
		float* weights = getAllWeights();
		for (int i = 1; i < weights[0]; i++) weights[i] += MR * (rand() % 201 - 100);
		updateNetwork(weights);
		delete[] weights;
	}
};