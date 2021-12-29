#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Dense>


class Perceptron {
public:
	Perceptron(int number_of_inputs);
	int number_of_inputs;
	double bias = 1.0;
	double lr = 0.1;
	Eigen::VectorXd weights;
	Eigen::VectorXd inputs;
	void set_weights(Eigen::VectorXd w_init);
	void set_inputs(Eigen::VectorXd d_in);
	void init_weights();
	double step_function(double x);
	double output();
	double loss(double output, double* label);
	void delta_rule(double loss);
};