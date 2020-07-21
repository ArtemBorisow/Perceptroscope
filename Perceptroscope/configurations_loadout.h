#include "includes.h"

namespace config_util
{
	int apply__nn_conf(neuralNetworkFF& neuralNetworkObject, int& AoLCI, float& LR, bool& logEpochInfo)
	{
		int returnCode = 0; //used to handle exceptions

		try
		{
			YAML::Node configRootNode = YAML::LoadFile("nn_conf.yml");

			//initializing "amount of neural networks" macro
			rest_util::NNAmount = configRootNode["NNAmount"].as<int>();

			//applying structure config
			std::string networkStructure = "";
			//making neuralNetworkFF standart structure in string
			for (int i = 0; i < configRootNode["structure"]["inputLayer"].as<int>(); i++) networkStructure += '#';
			networkStructure += '/';
			YAML::Node hiddenLayers_structureNode_arrayNode = configRootNode["structure"]["hiddenLayers"];
			for (std::size_t index = 0; index < hiddenLayers_structureNode_arrayNode.size(); index++)
			{
				for (int i = 0; i < hiddenLayers_structureNode_arrayNode[index].as<int>(); i++) networkStructure += '#';
				networkStructure += '/';
			}
			for (int i = 0; i < configRootNode["structure"]["outputLayer"].as<int>(); i++) networkStructure += '#';
			//updating network via new structure located in string
			neuralNetworkObject.updateNetwork(networkStructure);

			//applying back propagation learning params config
			AoLCI = configRootNode["BP_learningParams"]["AoLCI"].as<int>();
			LR = configRootNode["BP_learningParams"]["learningRate"].as<float>();
			logEpochInfo = configRootNode["BP_learningParams"]["logEpochInfo"].as<bool>();
			//applying training set array config
			std::string trainingSetArray = "";
			YAML::Node trainingSetArray_BP_learningParamsNode_arrayNode = configRootNode["BP_learningParams"]["trainingSetArray"];
			for (std::size_t index = 0; index < trainingSetArray_BP_learningParamsNode_arrayNode.size(); index++)
			{
				for (std::size_t i = 0; i < trainingSetArray_BP_learningParamsNode_arrayNode[index]["inputs"].size(); i++)
				{
					trainingSetArray += trainingSetArray_BP_learningParamsNode_arrayNode[index]["inputs"][i].as<char>();
					trainingSetArray += ',';
				}
				trainingSetArray[trainingSetArray.length() - 1] = ':';
				for (std::size_t i = 0; i < trainingSetArray_BP_learningParamsNode_arrayNode[index]["outputs"].size(); i++)
				{
					trainingSetArray += trainingSetArray_BP_learningParamsNode_arrayNode[index]["outputs"][i].as<char>();
					trainingSetArray += ',';
				}
				trainingSetArray[trainingSetArray.length() - 1] = '/';
			}
			trainingSetArray[trainingSetArray.length() - 1] = ';';
			//creating new trainig set array for network via new trainingSetArray located in string
			neuralNetworkObject.createTrainingSetArray(trainingSetArray);
		}
		catch (const std::exception &e)
		{
			std::cout << "INTERNAL ERROR: in \"nn_conf\" LOADING BLOCK: " << e.what() << std::endl;
			returnCode = 1;
		}

		return returnCode;
	}
	int apply__db_conf()
	{
		int returnCode = 0; //used to handle exceptions

		try
		{
			YAML::Node configRootNode = YAML::LoadFile("db_conf.yml");

			//applying config
			rest_util::DBName = configRootNode["name"].as<std::string>();
			rest_util::DBMeasurementName = configRootNode["measurementName"].as<std::string>();
			rest_util::DBEndpoint = configRootNode["endpoint"].as<std::string>();
			rest_util::dataPerRequestBufferSize = configRootNode["dataPerRequestBufferSize"].as<int>();
		}
		catch (const std::exception &e)
		{
			std::cout << "INTERNAL ERROR: in \"db_conf\" LOADING BLOCK: " << e.what() << std::endl;
			returnCode = 1;
		}

		return returnCode;
	}
}