#pragma once
#include <vector>
#include <string>
#include "../include/Tray.h"

class Test {

	private:
		std::vector<Tray> trayVector;
	public:
		Test(std::vector<Tray>);
		void test_the_system(const std::string&);
		void test_the_system_randomly(const std::string&);
};