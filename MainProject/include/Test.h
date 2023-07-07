#pragma once
#include <vector>
#include "../include/Tray.h"
#include <string>

class Test {

	private:
		std::vector<Tray> trayVector;
	public:
		Test(std::vector<Tray>);
		void test_the_system(const std::string&);
};