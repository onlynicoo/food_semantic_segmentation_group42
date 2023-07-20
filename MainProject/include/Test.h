#pragma once
#include <vector>
#include <string>
#include "../include/Tray.h"

class Test {

	private:
		//Vector containg all the Tray (before,after) objects
		std::vector<Tray> trayVector;

	public:
		//constructor
		Test(std::vector<Tray>);

		//It takes in input "Food Dataset Leftover". Using trayVector, it starts the test
		void testTheSystem(const std::string&);
};