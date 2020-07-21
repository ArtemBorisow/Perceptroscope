#include "includes.h"

namespace rest_util
{
	std::thread* sendDataFunc_mainThreadPtr;
	int NNAmount; //amount of neural networks involved
	std::string DBName; //database name in Influx service module
	std::string DBMeasurementName; //measurement name in desired database
	std::string DBEndpoint; //endpoint of Influx service with port
	int dataPerRequestBufferSize; //size of dataBuffer (measures in Influx queries)
	int actual_dataPerRequestBufferSize; //actual value of dataPerRequestBufferSize in runtime
	std::string dataBuffer; //buffer for storing dataToSend, for requests count minimizing
	bool isReadyToSend; //send function lockdown flag

	void sendData(std::string data)
	{
		//lockdown activation
		isReadyToSend = false;

		//request to database
		int statusCode; //status code of the response
		web::http::client::http_client httpClient(utility::conversions::to_string_t(DBEndpoint)); //setting 'endpoint'
		web::http::uri_builder uri(U("/write")); //uri method
		uri.append_query(U("db"), utility::conversions::to_string_t(DBName)); //adding parameter 'databaseName = nameOfDesiredDatabase' (Influx rule)
		//start of the request sequence
		pplx::task<void> requestTask =
			httpClient.request(web::http::methods::POST, uri.to_string(), utility::conversions::to_string_t(data.c_str()))
		.then
			([&statusCode](web::http::http_response response) mutable {
			statusCode = (int)(response.status_code());
			response.content_ready().wait();
			return response.extract_utf8string();})
		.then
			([&statusCode](std::string body) mutable {
				if (statusCode != 204)
				std::cout << "INTERNAL ERROR: in REST_SENDER block, given code: " << statusCode << "; With followed body: " << body.c_str() << std::endl;
		});

		//waiting for request full complete
		try
		{
			requestTask.wait();
		}
		catch (const std::exception &e)
		{
			std::cout << "INTERNAL ERROR: in REST_SENDER block: " << e.what() << std::endl;
		}

		//lockdown deactivation
		isReadyToSend = true;
	}
	void sendData_call()
	{
		if (isReadyToSend)
		{
			sendDataFunc_mainThreadPtr = new std::thread(sendData, dataBuffer);
			dataBuffer = "";
		}
	}
	void measermentDropQuery()
	{
		//request to database
		int statusCode; //status code of the response
		web::http::client::http_client httpClient(utility::conversions::to_string_t(DBEndpoint)); //setting 'endpoint'
		web::http::uri_builder uri(U("/query")); //uri method
		uri.append_query(U("db"), utility::conversions::to_string_t(DBName)); //adding parameter 'databaseName = nameOfDesiredDatabase' (Influx rule)
		uri.append_query(U("q"), utility::conversions::to_string_t("drop series from \"" + DBMeasurementName + "\"\n"));
		//start of the request sequence
		pplx::task<void> requestTask =
			httpClient.request(web::http::methods::POST, uri.to_string())
			.then
			([&statusCode](web::http::http_response response) mutable {
			statusCode = (int)(response.status_code());
			response.content_ready().wait();
			return response.extract_utf8string(); })
			.then
			([&statusCode](std::string body) mutable {
				if (statusCode != 200)
					std::cout << "INTERNAL ERROR: in REST_PREPARATION block, given code: " << statusCode << "; With followed body: " << body.c_str() << std::endl;
			});

			//waiting for request full complete
			try
			{
				requestTask.wait();
			}
			catch (const std::exception &e)
			{
				std::cout << "INTERNAL ERROR: in REST_PREPARATION block: " << e.what() << std::endl;
			}
	}
}