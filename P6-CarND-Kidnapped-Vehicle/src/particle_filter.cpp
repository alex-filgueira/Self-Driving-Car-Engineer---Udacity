/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>


#include "helper_functions.h"

using std::string;
using std::vector;

using namespace std;
static default_random_engine generator;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  if(is_initialized){
    return;
  }
  
  num_particles = 200;  // TODO: Set the number of particles
  
   /* @param x Initial x position [m] (simulated estimate from GPS)
   * @param y Initial y position [m]
   * @param theta Initial orientation [rad]
   * @param std[] Array of dimension 3 [standard deviation of x [m], 
   *   standard deviation of y [m], standard deviation of yaw [rad]]
   */
  
  
  //Normal distributions
  normal_distribution<double> ndist_x(x, std[0]);
  normal_distribution<double> ndist_y(y, std[1]);
  normal_distribution<double> ndist_theta(theta, std[2]);
  
  for(unsigned int i=0;i<num_particles;i++){
    Particle p;
    p.id = i;
    p.x = ndist_x(generator);
    p.y = ndist_y(generator);
    p.theta = ndist_theta(generator);
    p.weight = 1.0; //Checkear
    particles.push_back(p);
    
  }
  
  is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  /**
   * prediction Predicts the state for the next time step
   *   using the process model.
   * @param delta_t Time between time step t and t+1 in measurements [s]
   * @param std_pos[] Array of dimension 3 [standard deviation of x [m], 
   *   standard deviation of y [m], standard deviation of yaw [rad]]
   * @param velocity Velocity of car from t to t+1 [m/s]
   * @param yaw_rate Yaw rate of car from t to t+1 [rad/s]
   */
  
  for(unsigned int i=0;i<num_particles;i++){
    if( fabs(yaw_rate) > 0.00001){
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta ));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
    else{
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
      // Theta will stay the same due to no yaw_rate
    }
    
    // Add noise to the particles
    //Normal distributions
    normal_distribution<double> ndist_x(0, std_pos[0]);
    normal_distribution<double> ndist_y(0, std_pos[1]);
    normal_distribution<double> ndist_theta(0, std_pos[2]);
	
    particles[i].x += ndist_x(generator);
    particles[i].y += ndist_y(generator);
    particles[i].theta += ndist_theta(generator);
      
  }
}


void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  /**
   * dataAssociation Finds which observations correspond to which landmarks 
   *   (likely by using a nearest-neighbors data association).
   * @param predicted Vector of predicted landmark observations
   * @param observations Vector of landmark observations
   */
  
  
  for(unsigned int i=0;i<observations.size();i++){
    //double d_min = dist(predicted[i].x,predicted[i].y,observations[0].x,observations[0].y);
	double d_min = numeric_limits<double>::max();
    observations[i].id = predicted[0].id; //upadte id
	for(unsigned int j=0;j<predicted.size();j++){
		//double d = dist(predicted[i].x,predicted[i].y,observations[j].x,observations[j].y);
		double xDist = observations[i].x - predicted[j].x;
		double yDist = observations[i].y - predicted[j].y;
		double d = xDist * xDist + yDist * yDist;
		if(d < d_min){
          d_min = d;
          observations[i].id = predicted[j].id; //upadte id
		  //cout << "d_min: "<<d_min <<" i: " <<i<<" predicted[j].id: " <<predicted[j].id<<endl;
        }
  	}
  }
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  /**
   * updateWeights Updates the weights for each particle based on the likelihood
   *   of the observed measurements. 
   * @param sensor_range Range [m] of sensor
   * @param std_landmark[] Array of dimension 2
   *   [Landmark measurement uncertainty [x [m], y [m]]]
   * @param observations Vector of landmark observations
   * @param map Map class containing map landmarks
   */
  

  //Select nearest landmarks
  for(unsigned int i=0;i<num_particles;i++){
	  
	vector<LandmarkObs> predict;
    for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
	  //considers a rectangular region is computationally more faster)
      //if(dist(particles[i].x,particles[i].y,map_landmarks.landmark_list[j].x_f,map_landmarks.landmark_list[j].y_f) < sensor_range){
        //if distance between particle and landmark is less than sensor_range
		if(fabs(particles[i].x - map_landmarks.landmark_list[j].x_f) <= sensor_range && fabs(particles[i].y - map_landmarks.landmark_list[j].y_f) <= sensor_range){
        predict.push_back(LandmarkObs{ map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f });
      }
    }
  
    //Transform coordinates vehicle to map
    vector<LandmarkObs> t_os;
    for(unsigned int m=0; m< observations.size(); m++  ){
      double tx = cos(particles[i].theta) * observations[m].x - sin(particles[i].theta) * observations[m].y + particles[i].x;
      double ty = sin(particles[i].theta) * observations[m].x + cos(particles[i].theta) * observations[m].y + particles[i].y;
      t_os.push_back(LandmarkObs{ observations[m].id, tx, ty });
    }
    
    dataAssociation(predict, t_os);
	particles[i].weight = 1.0;
    
    for(unsigned int n=0;n<t_os.size();n++){
		double pred_x, pred_y;
      
      //x,y coordinates of the prediction associated with the current observation
      for(unsigned int l=0;l<predict.size();l++){
        if(predict[l].id == t_os[n].id){
			pred_x = predict[l].x;
			pred_y = predict[l].y;
        }
      }
      //Weight for this observation with multivariate Gaussian
      particles[i].weight *=  ( 1/(2*M_PI*std_landmark[0]*std_landmark[1])) * exp( -( pow(pred_x-t_os[n].x,2)/(2*pow(std_landmark[0], 2)) + (pow(pred_y-t_os[n].y,2)/(2*pow(std_landmark[1], 2))) ) );
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  vector<double> weights;
  for(unsigned int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  double maxWeight = *max_element(weights.begin(), weights.end()); 

  uniform_real_distribution<double> dist_D(0.0, maxWeight);
  uniform_int_distribution<int> dist_I(0, num_particles - 1);
  int index = dist_I(generator);
  double beta = 0.0;
  vector<Particle> resampledParticles;
  for(unsigned int i = 0; i < num_particles; i++) {
    beta += dist_D(generator) * 2.0;
    while(beta > particles[index].weight) {
      beta -= particles[index].weight;
      index = (index + 1) % num_particles;
    }
    resampledParticles.push_back(particles[index]);
  }
  //update values
  particles = resampledParticles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}