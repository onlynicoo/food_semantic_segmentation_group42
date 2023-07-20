#pragma once
#include <vector>
#include <string>
#include "../include/Tray.h"

class Test {

	private:
		std::vector<Tray> trayVector;
	public:
		Test(std::vector<Tray>);
		void testTheSystem(const std::string&);
};