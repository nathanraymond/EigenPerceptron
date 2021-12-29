// Perceptron.cpp : This file contains the 'main' function. Program execution begins and ends there.
#include <algorithm>
#include <vector>
#include <iostream>
#include <random>
#include <numeric>
#include <cmath>
#include <time.h>
#include <Eigen/Dense>
#include "EigenPerceptron.h"

Perceptron::Perceptron(int x_inputs) {
    this->bias = bias; // Save bias value
    this->number_of_inputs = x_inputs;
}
void Perceptron::init_weights() {
    weights.resize((number_of_inputs + 1));
    weights.setRandom();
}
void Perceptron::set_weights(Eigen::VectorXd w_init) {
    weights.resize((number_of_inputs + 1));
    weights << w_init;
}
void Perceptron::delta_rule(double loss) {
    Eigen::VectorXd delta_w = inputs * lr * loss;
    set_weights((weights + delta_w));
}

void Perceptron::set_inputs(Eigen::VectorXd d_in) {
    inputs.resize((d_in.size() + 1));
    inputs << d_in, bias; // Add bias term to the end of the inputs vector
}

double Perceptron::output() {
    double weighted_sum = inputs.dot(weights); // Dot product of Inputs⋅Weights vectors
    return step_function(weighted_sum);
}

double Perceptron::loss(double output, double* label) {
    double l = *label;
    return std::pow((l - output), 1);
}

double Perceptron::step_function(double weighted_sum) {
    // weighted_sum is a 1D tensor
    if (weighted_sum < 0) {
        return 0;
    }

    else {
        return 1;
    }
    //return 1.0/(1.0 + std::exp(-weighted_sum));
}

int main()
{
    Eigen::MatrixXd batch{
      {0, 0, 0},
      {0, 1, 1},
      {1, 0, 1},
      {1, 1, 1}
    };

    Perceptron* p = new Perceptron(2);

    p->init_weights();
    for (int epoch = 0; epoch < 10; epoch++) {
        std::cout << "Epoch: " << epoch << std::endl;
        std::cout << "\n" << std::endl;
        for (int i = 0; i < batch.rows(); i++) {
            auto data = batch.row(i).head(2);
            auto labels = batch.row(i).tail(1);
            p->set_inputs(data);

            std::cout << "Output: " << p->output() << std::endl;
            std::cout << "Label: " << labels << std::endl;
            std::cout << "Loss: " << p->loss((p->output()), labels.data()) << std::endl;
            std::cout << "\n" << std::endl;

            p->delta_rule(p->loss((p->output()), labels.data()));
        }
        std::cout << "\n" << std::endl;
        std::cout << "\n" << std::endl;

    }

}