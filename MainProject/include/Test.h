// Author: Carmine Graniello

#pragma once
#include <vector>
#include <string>
#include "../include/Tray.h"

// Handles the testing of the system over a set of trays
class Test {

	private:
		
        // Contains all the Tray (before, after) objects
		std::vector<Tray> trayVector;

	public:
		
        // Constructor
		Test(std::vector<Tray>);

		// Tests the Trays taking in input the path to "Food_leftover_dataset"
		void testTheSystem(const std::string&);
};