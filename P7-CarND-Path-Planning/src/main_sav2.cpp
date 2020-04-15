#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"
//Added--
#include "spline.h" //Cubic Spline interpolation in C++
#include <math.h> //Set of functions to compute common mathematical operations and transformations
#include <chrono> //The elements in this header deal with time
#include <thread>//Class to represent individual threads of execution.


// for convenience
using nlohmann::json;
using std::string;
using std::vector;

//Added--
using namespace std;
using json = nlohmann::json;


int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  //double max_s = 6945.554;

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

 double speed_ref = 0.0;
  int next_car_road_lane = 1;

  h.onMessage([&speed_ref, &next_car_road_lane, &map_waypoints_x,&map_waypoints_y,&map_waypoints_s,
               &map_waypoints_dx,&map_waypoints_dy]
              (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
               uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
          // Main car's localization Data
          double car_x = j[1]["x"];
          double car_y = j[1]["y"];
          double car_s = j[1]["s"];
          double car_d = j[1]["d"];
          double car_yaw = j[1]["yaw"];
          double car_speed = j[1]["speed"];
          
          //Calculate My road lane
          //cout << "car_d: " <<car_d<<endl;
          int car_road_lane = -1;
           if(car_d>=0 && car_d <4){car_road_lane = 0;}
           if(car_d>=4 && car_d <8){car_road_lane = 1;}
           if(car_d>=8 && car_d <12){car_road_lane = 2;}
          
          //Calculate My next S
          //double car_nextS =  car_s + car_speed * 0.02;
          //double car_theta = atan(car_y/car_x);
          vector<double> car_nextSD;
          car_nextSD =  getFrenet(car_x, car_y, car_yaw,  map_waypoints_x, map_waypoints_y);
          double car_nextS = car_nextSD[0];
          //cout << "car_nextS: " <<car_nextSD[0]<< " car_nextD: " <<car_nextSD[1]<<endl;
 
          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values 
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];

          // Sensor Fusion Data, a list of all other cars on the same side 
          //   of the road.
          auto sensor_fusion = j[1]["sensor_fusion"];

          json msgJson;
          
             if (previous_path_x.size() > 0) {
              car_nextS = end_path_s;
            }
			cout<< "My-> speed: "<<car_speed<< " s: "<<car_s<<" car_nextS: "<<car_nextS<< " road_lane: "<<car_road_lane << endl;

          //Search Others cars
          //The data format for each car is: [ id, x, y, vx, vy, s, d]
          bool flag_front = false;
          bool flag_right = false;
          bool flag_left = false;
          float d_security = 25.0;
          
          double dToFront = 1000;
          double speed_front = 0;
          double dToLeft = 1000;
          double dToRight = 1000;
          
          //Recorre toda la lista de coches vecinos
          for(int i=0; i < sensor_fusion.size();i++){
            int id = sensor_fusion[i][0];
            //Find Speed
            double vx = sensor_fusion[i][3];
            double vy = sensor_fusion[i][4];
            double speed = sqrt(vx*vx+vy*vy);
            //Find S
            double s =  sensor_fusion[i][5];
            //Find road line
           	double d = sensor_fusion[i][6];
            int road_lane = -1; //if is in the other hand or out the road
            if(d>=0 && d <4){road_lane = 0;}
            if(d>=4 && d <8){road_lane = 1;}
            if(d>=8 && d <12){road_lane = 2;}
            
          //Calculate next position
          double x = sensor_fusion[i][1];
          double y = sensor_fusion[i][2];
          double nextx = x + vx * 0.02;
          double nexty = y + vy * 0.02;
          double theta = atan(y/x);
            
          double nextS;
          nextS = s + ((double)previous_path_x.size()*0.02*speed);
            
          //cout<< "neighbor-> ID: "<<id<<" speed: "<<speed<< " s: "<<s<< " nextSs: "<<nextS << " road_lane: "<<road_lane <<endl;
            
            //Adapt d_security in function from speed
            //d_security = car_speed * 30 / 50;
            //if(d_security < 5){d_security = 20;}
            //cout << "d_security: "<<d_security<<endl;
            
            //check nerarness for the neighbors
            if(road_lane>=0){
              if((road_lane == car_road_lane) && (nextS > car_nextS+10)){
                //in the same lane -> in front
                if((nextS - car_nextS) < dToFront){//For select the in front but more near.
                  dToFront = nextS - car_nextS;
                  speed_front = speed;
                }
                if((nextS - car_nextS)<d_security){
                  flag_front = true;
                  cout << "flag_front: "<<flag_front<<" dToFront: "<<dToFront <<endl; 
                  cout<< "neighbor-> ID: "<<id<<" speed: "<<speed<< " s: "<<s<< " nextSs: "<<nextS << " road_lane: "<<road_lane <<endl;
                }
              }
              
              if(road_lane == (car_road_lane-1)){
                //in the left lane
                if(nextS > car_nextS+10){//In front
                  if((nextS - car_nextS) < dToLeft){//For select the in front but more near.
                    dToLeft = nextS - car_nextS;
                  }
                  if((nextS - car_nextS)<(d_security*1.8)){
                    flag_left = true;
                    cout << "flag_left: "<<flag_left<<" dToLeft: "<<dToLeft <<endl; 
                    cout<< "neighbor-> ID: "<<id<<" speed: "<<speed<< " s: "<<s<< " nextSs: "<<nextS << " road_lane: "<<road_lane <<endl;
                  }
                }
                else{//Behind
                  if(speed <= car_speed){//low
                    if(fabs(car_nextS-nextS)<(d_security/2)){
                      flag_left = true;
                      cout << "flag_left: "<<flag_left <<endl;
                      cout<< "neighbor-> ID: "<<id<<" speed: "<<speed<< " s: "<<s<< " nextSs: "<<nextS << " road_lane: "<<road_lane <<endl;
                    }
                  }
                  else{//quick
                    if(fabs(car_nextS-nextS)<d_security){
                      flag_left = true;
                      cout << "flag_left: "<<flag_left<<endl;
                      cout<< "neighbor-> ID: "<<id<<" speed: "<<speed<< " s: "<<s<< " nextSs: "<<nextS << " road_lane: "<<road_lane <<endl;
                    }
                  }
                }
              }
              
              if(road_lane == (car_road_lane+1)){
                //in the right lane
                if(nextS > car_nextS+10){//in front
                  if((nextS - car_nextS) < dToRight){//For select the in front but more near.
                    dToRight = nextS - car_nextS;
                  }
                  if(nextS - car_nextS<(d_security*1.8)){
                    flag_right = true;
                    cout << "flag_right: "<<flag_right<<" dToRight: "<<dToRight <<endl;
                    cout<< "neighbor-> ID: "<<id<<" speed: "<<speed<< " s: "<<s<< " nextSs: "<<nextS << " road_lane: "<<road_lane <<endl;
                  }
                }
                else{//Behind
                  if(speed <= car_speed){//low
                    if(fabs(car_nextS-nextS)<d_security/2){
                      flag_right = true;
                      cout << "flag_right: "<<flag_right<<endl;
                      cout<< "neighbor-> ID: "<<id<<" speed: "<<speed<< " s: "<<s<< " nextSs: "<<nextS << " road_lane: "<<road_lane <<endl;
                    }
                  }
                  else{//quick
                    if(fabs(car_nextS-nextS)<d_security){
                      flag_right = true;
                      cout << "flag_right: "<<flag_right<<endl;
                      cout<< "neighbor-> ID: "<<id<<" speed: "<<speed<< " s: "<<s<< " nextSs: "<<nextS << " road_lane: "<<road_lane <<endl;
                    }
                  }
                }
              }
            }
          }//end for neighbors
          
          //state machine
          //3 lanes:0,1,2
          //I try go ever in the center lane = 1
          float max_speed = 49.8;
          float max_acc = .244;
          double speed_dif = 0;
          
         next_car_road_lane = car_road_lane;
          if(flag_front){//car in front
           	 if(!flag_right && !flag_left && next_car_road_lane == 1){
                if(dToRight >= dToLeft){
                    next_car_road_lane ++;
                    cout << "Change to line->: "<<next_car_road_lane<<endl;
                }
                else{
                    next_car_road_lane --;
                    cout << "Change to line<-: "<<next_car_road_lane<<endl;
                }
           	 }
            //flag_front but not in middle + 2 frees
           	 else{
                if(!flag_right && next_car_road_lane < 2 ){//right is free and not is in the last lane
                  next_car_road_lane ++;
                  cout << "Change to line->: "<<next_car_road_lane<<endl;
                }
                else if(!flag_left && next_car_road_lane > 0 ){//left is free and not is in the first lane
                  next_car_road_lane --;
                  cout << "Change to line<-: "<<next_car_road_lane<<endl;
                }
                else{ //STOP!
                  if(dToFront < 5){
                    //Emergenci STOP
                    speed_dif = speed_dif - max_acc * 4;
                    cout << "STOP- EMERGENCY! ->speed_dif: "<<speed_dif<<endl;
                  }
                  else{
                    if(speed_ref > speed_front){//Our car is more quick than the in front
                      speed_dif = speed_dif - max_acc;
                      cout << "STOP ->speed_dif: "<<speed_dif<<endl;
                    }
                    else{
                      speed_dif = speed_dif + max_acc;//gas
                    }
                  }
                }
           	 }
          }
		  
          else{//not car in front
            if(next_car_road_lane == 1){
              if(speed_ref < max_speed){
                //push gas
                speed_dif = speed_dif + max_acc;
                cout << "push gas ->speed_dif: "<<speed_dif<<endl;
              }
            }
            else{
              if(next_car_road_lane <1 && !flag_right){//not are in the last lane and the right is free
                next_car_road_lane ++;
                cout << "Change to line->: "<<next_car_road_lane<<endl;
              }
              else if(next_car_road_lane >1 && !flag_left){//not are in the last lane and the right is free
                next_car_road_lane --;
                cout << "Change to line<-: "<<next_car_road_lane<<endl;
              }
              else{
                if(speed_ref < max_speed){
                  //push gas
                  speed_dif = speed_dif + max_acc;
                  cout << "push gas ->speed_dif: "<<speed_dif<<endl;
                }
              }
            }
          }
          if(next_car_road_lane > 2){next_car_road_lane = 2;}
          if(next_car_road_lane < 0){next_car_road_lane = 0;}
          
          //END state machine
          
          //trajectory
          //I use 5 points for the set_points function (library spline)
          vector<double> pts_x;
          vector<double> pts_y;
          double ref_x = car_x;
          double ref_y = car_y;
         double ref_yaw = deg2rad(car_yaw);
          if(previous_path_x.size() >=2){
            //take the las 2 points
            ref_x = previous_path_x[previous_path_x.size() - 1];
            ref_y = previous_path_y[previous_path_x.size() - 1];

            double ref_x_prev = previous_path_x[previous_path_x.size() - 2];
            double ref_y_prev = previous_path_y[previous_path_x.size() - 2];
            ref_yaw = atan2(ref_y-ref_y_prev, ref_x-ref_x_prev);
            
            pts_x.push_back(ref_x_prev);
            pts_x.push_back(ref_x);

            pts_y.push_back(ref_y_prev);
            pts_y.push_back(ref_y);
          }
          else{
            //I have not 2 points -> use actual and invent the last
            pts_x.push_back(car_x - cos(car_yaw));
            pts_y.push_back(car_y - sin(car_yaw));
            
            pts_x.push_back(car_x);
            pts_y.push_back(car_y);
          }
          
          //Calculate the next 3 points
		  //values 38,68,98 are selected for have a good curve answer.(less values -> more quick)
          vector<double> next_pts1 = getXY(car_nextS+30, 2+4*next_car_road_lane,map_waypoints_s,map_waypoints_x,map_waypoints_y);
          vector<double> next_pts2 = getXY(car_nextS+60, 2+4*next_car_road_lane,map_waypoints_s,map_waypoints_x,map_waypoints_y);
          vector<double> next_pts3 = getXY(car_nextS+90, 2+4*next_car_road_lane,map_waypoints_s,map_waypoints_x,map_waypoints_y);
          
          pts_x.push_back(next_pts1[0]);
          pts_y.push_back(next_pts1[1]);
          pts_x.push_back(next_pts2[0]);
          pts_y.push_back(next_pts2[1]);
          pts_x.push_back(next_pts3[0]);
          pts_y.push_back(next_pts3[1]);
          
          
          //mod-----

          //Transform to local car coords.
          for ( int i = 0; i < pts_x.size(); i++ ) {
            double t_x = pts_x[i] - ref_x;
            double t_y = pts_y[i] - ref_y;

            pts_x[i] = t_x * cos(0 - ref_yaw) - t_y * sin(0 - ref_yaw);
            pts_y[i] = t_x * sin(0 - ref_yaw) + t_y * cos(0 - ref_yaw);
          }

          // Create the spline.
          tk::spline s;
          s.set_points(pts_x, pts_y);

          vector<double> next_x_vals;
          vector<double> next_y_vals;
          for ( int i = 0; i < previous_path_x.size(); i++ ) {
            next_x_vals.push_back(previous_path_x[i]);
            next_y_vals.push_back(previous_path_y[i]);
          }

          // Calculate distance y position on 30 m ahead.
          double target_dist = sqrt(30.0*30.0 + s(30.0)*s(30.0));

          double x_add_on = 0;

          for( int i = 1; i < 50 - previous_path_x.size(); i++ ) {
            speed_ref += speed_dif;
            if ( speed_ref > max_speed ) {
              speed_ref = max_speed;
            } else if ( speed_ref < max_acc ) {
              speed_ref = max_acc;
            }
            
            //cout << "speed_ref: " << speed_ref<<endl;
            
            double x_point = x_add_on + 30.0/ (target_dist/(0.02*speed_ref/2.24));
            double y_point = s(x_point);

            x_add_on = x_point;

            double x_ref = x_point;
            double y_ref = y_point;

            x_point = x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw);
            y_point = x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw);

            x_point += ref_x;
            y_point += ref_y;

            next_x_vals.push_back(x_point);
            next_y_vals.push_back(y_point);
          }

		  //send values
          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}