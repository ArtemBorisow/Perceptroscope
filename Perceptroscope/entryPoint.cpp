#include "includes.h"

void makeTrainingAndRecording(neuralNetworkFF& neuralNetworkObject, int AoLCI, float LR, bool logEpochInfo, std::string globalIterationIndex)
{
	for (int i = 0; i < AoLCI; i++)
	{
		//training NN for one iteration to collect data about learning process
		float iterationNNError = neuralNetworkObject.learn(1, LR, logEpochInfo);

		//building new string query for database and adding it to dataBuffer
		rest_util::actual_dataPerRequestBufferSize++; //informing rest_util that data is bigger now by 1 Influx querie
		std::string newQuery = "";
		std::string newTimeStamp_for_newQuery = " " + std::to_string(((int64_t)i * (int64_t)1000000000) + (int64_t)946684800000000000); //constants are for special date in RFC3339 (2000.01.01)
		newQuery = rest_util::DBMeasurementName + ",networkIndex=" + globalIterationIndex + " NNError=" + std::to_string(iterationNNError); //recording average training set error of neural network
		newQuery += newTimeStamp_for_newQuery; //adding timestamp
		rest_util::dataBuffer += newQuery + "\n"; //adding query to data buffer
		float** synapsesData = neuralNetworkObject.getAllSynapsesData();
		for (int index = 1; index <= synapsesData[0][0]; index++)
		{
			newQuery = "";
			newQuery += rest_util::DBMeasurementName + ",networkIndex=" + globalIterationIndex + ",weightIndex=" + std::to_string(index - 1) + ",indexInNeuron=" + std::to_string((int)synapsesData[index][1]) + ",neuronIndexInLayer=" + std::to_string((int)synapsesData[index][2]) + ",layerIndex=" + std::to_string((int)synapsesData[index][3]); //adding tags values
			newQuery += " weight=" + std::to_string(synapsesData[index][0]); //adding weight field value
			newQuery += " " + newTimeStamp_for_newQuery; //adding timestamp
			rest_util::dataBuffer += newQuery + "\n"; //adding query to data buffer
		}
		for (int n = 1; n <= synapsesData[0][0]; n++) delete[] synapsesData[n];
		delete[] synapsesData[0];
		delete[] synapsesData;

		//trying to call for dataSend func
		if (rest_util::actual_dataPerRequestBufferSize > rest_util::dataPerRequestBufferSize) rest_util::sendData_call();
	}
}

int main()
{
	//main objects declar + init
	neuralNetworkFF neuralNetworkObject; //creating neural network object
	//back propagation learning parameters
	int AoLCI; //amount of learning cycle iterations
	float LR; //learning rate
	bool logEpochInfo; //log basic information for each learning iteration
	rest_util::actual_dataPerRequestBufferSize = 0; //init
	rest_util::isReadyToSend = true; //init

	//configurations loadout
	if (config_util::apply__nn_conf(neuralNetworkObject, AoLCI, LR, logEpochInfo)) return 1;
	if (config_util::apply__db_conf()) return 1;

	//preparations
	rest_util::measermentDropQuery();

	//neural network learning/trainig process & database updating (multiple times)
	for (int i = 0; i < rest_util::NNAmount; i++)
	{
		makeTrainingAndRecording(neuralNetworkObject, AoLCI, LR, logEpochInfo, std::to_string(i));
		neuralNetworkObject.randomizeWeights();
		if (rest_util::sendDataFunc_mainThreadPtr->joinable())
		{
			rest_util::sendDataFunc_mainThreadPtr->join();
			rest_util::sendData_call();
		}
	}

	//terminating operations
	if (rest_util::sendDataFunc_mainThreadPtr->joinable()) rest_util::sendDataFunc_mainThreadPtr->join();
	return 0;
}