/*
 * Copyright 2014-2015, Max Planck Society.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

/* Created by Edgar Klenske <edgar.klenske@tuebingen.mpg.de>
 *
 * This file is a playground to try and plot things.
 *
 */


#include "../tools/math_tools.h"
#include "gaussian_process.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <string>

int main(int argc, char **argv)
{

  ///// Play with the spectrum /////
  double Tmax = 1024;
  int res = 1024;

  double p = 512;
  double s2 = 200;
  double sigma = 100;

  Eigen::VectorXd X = Eigen::VectorXd::LinSpaced(res,0,Tmax);
  Eigen::VectorXd Y = s2*(2*M_PI*X/p).array().sin();

  int N_fft = 4096;

  Eigen::ArrayXd W = math_tools::hamming_window(Y.rows());

  Eigen::VectorXd Yw = Y.array()*W;

  clock_t begin = std::clock();
  std::pair<Eigen::VectorXd, Eigen::VectorXd> result = math_tools::compute_spectrum(Yw, N_fft);
  clock_t end = std::clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  std::cout << "time for FFT: " << elapsed_secs << " s." << std::endl;


  Eigen::VectorXd amplitudes = result.first;
  Eigen::VectorXd frequencies = result.second;

  double dt = Tmax/res;
  frequencies /= dt;

  Eigen::VectorXd periods = 1/frequencies.array();

  assert(amplitudes.size() == frequencies.size());

  Eigen::VectorXd::Index maxIndex;
  amplitudes.maxCoeff(&maxIndex);

  std::cout << maxIndex << std::endl;
  std::cout << periods(maxIndex) << std::endl;

  std::ofstream outfile;
  outfile.open("spectrum_data.csv", std::ios_base::out);
  if(outfile.is_open()) {
    outfile << "frequency, amplitude\n";
    for( int i = 0; i < amplitudes.size(); ++i) {
      outfile << std::setw(8) << frequencies[i] << "," << std::setw(8) << amplitudes[i] << "\n";
    }
  } else {
    std::cout << "unable to write to file" << std::endl;
  }
  outfile.close();

  ///// Play with the GP /////

  // first, read data from file
  double buff[static_cast<int>(1e6)];
  int rows=0;
  int cols=0;

  // Read numbers from file into buffer.
  std::ifstream infile;
  infile.open("gear_error.csv");

  while (! infile.eof())
  {
    std::string line;
    std::getline(infile, line);

    int temp_cols = 0;
    std::stringstream stream(line);
    std::string token;

    while(std::getline(stream, token, ',')) {
      std::stringstream temp(token);
      temp >> buff[cols*rows+temp_cols++];
    }

    if (temp_cols == 0)
      continue;

    if (cols == 0)
      cols = temp_cols;

    rows++;
  }
  infile.close();
  rows--;

  // Populate matrix with numbers.
  Eigen::MatrixXd data(rows,cols);
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
      data(i,j) = buff[ cols*i+j ];

  // prepare the data from this matrix
  Eigen::VectorXd time = data.col(1);
  Eigen::VectorXd ra_raw_dist = data.col(5);
  Eigen::VectorXd control = data.col(7);
  Eigen::VectorXd meas(time.rows());

  meas(0) = control(0);
  for (int i = 1; i<time.rows(); ++i) {
    meas(i) = meas(i-1) + control(i);
  }
  meas += ra_raw_dist;

  Eigen::VectorXd hyper_parameters(7);
  hyper_parameters << 10, 1, 0.5, 500, 10, 500, 1;

  covariance_functions::PeriodicSquareExponential2 covariance_function(hyper_parameters.array().log());
  GP gp(1e0, covariance_function);

  gp.enableExplicitTrend();
  gp.infer(time, meas);

  int M = 512; // number of prediction points
  Eigen::VectorXd locations = Eigen::VectorXd::LinSpaced(M, 0, time.maxCoeff() + 1500);
  std::pair<Eigen::VectorXd, Eigen::MatrixXd> predictions = gp.predict(locations);

  Eigen::VectorXd means = predictions.first;
  Eigen::VectorXd stds = predictions.second.diagonal();

  outfile.open("measurement_data.csv", std::ios_base::out);
  if(outfile.is_open()) {
    outfile << "location, output\n";
    for( int i = 0; i < time.size(); ++i) {
      outfile << std::setw(8) << time[i] << "," << std::setw(8) << meas[i] << "\n";
    }
  } else {
    std::cout << "unable to write to file" << std::endl;
  }
  outfile.close();

  outfile.open("gp_data.csv", std::ios_base::out);
  if(outfile.is_open()) {
    outfile << "location, mean, std\n";
    for( int i = 0; i < locations.size(); ++i) {
      outfile << std::setw(8) << locations[i] << "," << std::setw(8) << means[i] << "," << std::setw(8) << stds[i] << "\n";
    }
  } else {
    std::cout << "unable to write to file" << std::endl;
  }
  outfile.close();

}

