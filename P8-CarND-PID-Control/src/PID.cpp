#include "PID.h"

/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
  /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   */
  Kp = Kp_;
  Ki = Ki_;
  Kd = Kd_;
  
  p_error = 0.0;
  i_error = 0.0;
  d_error = 0.0;
  prev_cte = 0.0;

}

void PID::UpdateError(double cte) {
  /**
   * TODO: Update PID errors based on cte.
   */
  //P error
  p_error = cte;
  
  //D error
  d_error = cte - prev_cte;
  prev_cte = cte;
  
  //I error
  i_error = i_error + cte;

}

double PID::TotalError() {
  /**
   * TODO: Calculate and return the total error
   */
  double pid_value;
  pid_value = -p_error * Kp - d_error * Kd - i_error * Ki;
  //pid_value = -p_error * Kp - d_error * Kd;
  
  return pid_value;  // TODO: Add your total error calc here!
}