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
using std::normal_distribution;
using std::uniform_int_distribution;
using std::uniform_real_distribution;
using std::default_random_engine;
using std::random_device;
using std::mt19937;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 1000;  // TODO: Set the number of particles

  random_device rd; // non-deteministic random number as the seed
  mt19937 mt(rd()); // mt random number generator

  // creates a normal (Gaussian) distribution for x, y, theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for(int i = 0; i < num_particles; i++)
  {
    Particle p;

    p.id = i;
    p.weight = 1.0;

    //Sample from these normal distributions
    p.x = dist_x(mt);
    p.y = dist_y(mt);
    p.theta = dist_theta(mt);

    particles.push_back(p);

    weights.push_back(1.0);
  }
  
  std::cout << "init particles size: " << particles.size() << std::endl; 

  std::cout << "init weights size: " << weights.size() << std::endl;

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
  random_device rd; // non-deteministic random number as the seed
  mt19937 mt(rd()); // mt random number generator

  for(u_int i = 0; i < particles.size(); i++)
  {
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;
    double post_x = 0.0;
    double post_y = 0.0;
    double post_theta = 0.0;

    // predict the position of each particle
    // avoid division by zero
    if (fabs(yaw_rate) > 0.001) {
        post_x = x + velocity/yaw_rate * ( sin(theta + yaw_rate*delta_t) - sin(theta));
        post_y = y + velocity/yaw_rate * ( cos(theta) - cos(theta+yaw_rate*delta_t) );
    } else {
        post_x = x + velocity*delta_t*cos(theta);
        post_y = y + velocity*delta_t*sin(theta);
    }

    post_theta = theta + yaw_rate*delta_t;

    // add noise
    // creates a normal (Gaussian) distribution for x, y, theta with updated value
    normal_distribution<double> dist_x(post_x, std_pos[0]);
    normal_distribution<double> dist_y(post_y, std_pos[1]);
    normal_distribution<double> dist_theta(post_theta, std_pos[2]);

    particles[i].x = dist_x(mt);
    particles[i].y = dist_y(mt);
    particles[i].theta = dist_theta(mt);
  }
}

LandmarkObs ParticleFilter::transform_obs(const Particle &p, const LandmarkObs &obs) {
  // Transform the x and y coordinates
  double x_map, y_map;

  x_map = p.x + (cos(p.theta) * obs.x) - (sin(p.theta) * obs.y);
  y_map = p.y + (sin(p.theta) * obs.x) + (cos(p.theta) * obs.y);
    
  // Create new Position to hold transformed observation
  LandmarkObs transformed_obs;
  transformed_obs.id = obs.id;
  transformed_obs.x = x_map;
  transformed_obs.y = y_map;
    
  return transformed_obs;
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
  
  //for all the observations, associate its id with the nearest predicted landmark id.
  for(u_int i = 0; i < observations.size(); i++)
  {
    int closest_landmark = 0;
    double min_dist = 999999.0;
    double curr_dist;
    // Iterate through all landmarks to check which is closest
    for (u_int j = 0; j < predicted.size(); ++j) 
    {
      // Calculate Euclidean distance
      curr_dist = sqrt(pow(predicted[i].x - observations[j].x, 2)
                     + pow(predicted[i].y - observations[j].y, 2));
      // Compare to min_dist and update if closest
      if (curr_dist < min_dist) 
      {
        min_dist = curr_dist;
        closest_landmark = j;
      }
    }

    observations[i].id = closest_landmark;
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

  vector<LandmarkObs> observations_pred;
  vector<LandmarkObs> observations_trans;
  double weight_all = 0.0;
  
  //for each particle, do the following:
  for(u_int i = 0; i < particles.size(); i++)
  {
    double probs = 1.0;

    //get the predited measurement which is map landmarks within sensor range.
    for(u_int k = 0; k < map_landmarks.landmark_list.size(); k++)
    {
      double distance = 0.0;
      LandmarkObs obs;
      obs.id = map_landmarks.landmark_list[i].id_i;
      obs.x  = map_landmarks.landmark_list[i].x_f;
      obs.y  = map_landmarks.landmark_list[i].y_f;

      distance = dist(particles[i].x, particles[i].y, obs.x, obs.y);

      if (distance < sensor_range) 
      {
        observations_pred.push_back(obs);
      }
    }
    
    //perform the transform of the observation measurements
    for(u_int j = 0; j < observations.size(); j++)
    {
      LandmarkObs ob_trans;
      ob_trans = transform_obs(particles[i], observations[j]);
      observations_trans.push_back(ob_trans);
    }

    //associate the transformed observation measurement with the landmarks
    dataAssociation(observations_pred, observations_trans);

    for(u_int j = 0; j < observations_trans.size(); j++)
    {
      particles[i].associations.push_back(observations_trans[j].id);
      particles[i].sense_x.push_back(observations_trans[j].x);
      particles[i].sense_y.push_back(observations_trans[j].y);
    }

    
    //calculate the probability of each observation with multi Guassian pdf.
    for(u_int j = 0; j < particles[i].associations.size(); j++)
    {
      double mu_x;
      double mu_y;
      double prob;

      for(u_int k = 0; k < observations_pred.size(); k++)
      {
        if (observations_pred[k].id == particles[i].associations[j]) 
        {
          mu_x = observations_pred[k].x;
          mu_y = observations_pred[k].y;
          break;
        }
      }

      prob = multiv_prob(std_landmark[0], std_landmark[1], 
                         particles[i].sense_x[j], particles[i].sense_x[j],
                         mu_x, mu_y);

      //multiply each probability to get the weight
      probs *= prob;
        
    }

    weights[i] = probs;

    //all weights, for normalization.
    weight_all += weights[i];
  }

  //normalized the weight.
  for(u_int i = 0; i < weights.size(); i++)
  {
    weights[i] /= weight_all;
    particles[i].weight = weights[i];
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  random_device rd; // non-deteministic random number as the seed
  mt19937 mt(rd()); // mt random number generator

  uniform_int_distribution<int> dist_index(1,1000);
  uniform_real_distribution<double> dist_beta(0.0, 1.0);

  vector<Particle> post_particles;

  int index = dist_index(mt);
  double beta = 0.0;
  double max_weight = 0.0;
  for(u_int i = 0; i < particles.size(); i++)
  {
    if(max_weight < weights[i])
    {
      max_weight = weights[i];
    }
  }
  
  for(u_int i = 0; i < particles.size(); i++)
  {
    beta += dist_beta(mt) * 2.0 * max_weight;

    while(beta > weights[index])
    {
      beta -= weights[index];
      index = (index + 1) % particles.size();
    }

    post_particles.push_back(particles[index]);
  }
  
  particles = post_particles;
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