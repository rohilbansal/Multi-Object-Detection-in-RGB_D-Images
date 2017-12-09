#include <iostream>
#include <stdio.h>
#include <vector>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <Sift.h>
#include "CImg.h"

using namespace std;
using namespace cv;
using namespace cimg_library;

RNG rng(12345);
int localizedCount = 1;

struct GravityEstimation{

  vector<int> angleThresh;
  vector<int> iterations;
  vector<int> initialAxis;

} gravityEst;

// Calculate distance between points.
double calculateDistance(double x, double y, double x1, double y1){
  return sqrt((pow((x - x1), 2)) + (pow((y - y1), 2)));
}

// Combine clusters
void combineClusters(int desc1, int desc2, vector< vector<SiftDescriptor> > &descArray){
  for(int i=0; i<descArray[desc2].size(); i++){
    descArray[desc1].push_back(descArray[desc2][i]);
  }
  descArray.erase(descArray.begin()+ desc2);
}

// Agglomerative clustering
void aggClustering(vector<SiftDescriptor> siftDescriptors, Mat inputImage, int classLabel){
  vector< vector<SiftDescriptor> > myArray(siftDescriptors.size());
  //cout<<myArray.size()<<endl;
  //cout<<myArray[1].size()<<endl;
  for(int i=0; i<siftDescriptors.size(); i++){
    myArray[i].push_back(siftDescriptors[i]);
  }
  double distance  = 0;
  while(myArray.size() > 5){
    double minVal = 10000;
    int desc1 = -1;
    int desc2 = -1;
    for(int i=0; i<myArray.size(); i++){
      double row1 = 0;
      double col1 = 0;
      if(myArray[i].size() > 1){
        for(int k=0; k<myArray[i].size(); k++){
          row1 += myArray[i][k].row;
          col1 += myArray[i][k].col;
        }
        row1 = row1/myArray[i].size();
        col1 = col1/myArray[i].size();
      }
      else{
        row1 = myArray[i][0].row;
        col1 = myArray[i][0].col;
      }
      for(int j=0; j<myArray.size(); j++){
        double row2 = 0;
        double col2 = 0;
        if(myArray[j].size() > 1){
          for(int k=0; k<myArray[j].size(); k++){
            row2 += myArray[j][k].row;
            col2 += myArray[j][k].col;
          }
          row2 = row2/myArray[j].size();
          col2 = col2/myArray[j].size();
        }
        else{
          row2 = myArray[j][0].row;
          col2 = myArray[j][0].col;
        }
        distance = calculateDistance(row1, col1, row2, col2);
        if(i != j &&  distance < minVal){
          minVal = distance;
          desc1 = i;
          desc2 = j;
        }
      }
    }
    combineClusters(desc1, desc2, myArray);
  }
  int count = 0;
  while(myArray.size() > 0){
    int max = 0;
    int index = 0;
    for(int i=0; i<myArray.size(); i++){
      if(max < myArray[i].size()){
        max = myArray[i].size();
        index = i;
      }
    }
    double minRow = 10000;
    double minCol = 10000;
    double maxRow = 0;
    double maxCol = 0;
    for(int i=0; i<myArray[index].size(); i++){
      if(minRow > myArray[index][i].row){
        minRow = myArray[index][i].row;
      }
      if(minCol > myArray[index][i].col){
        minCol = myArray[index][i].col;
      }
      if(maxRow < myArray[index][i].row){
        maxRow = myArray[index][i].row;
      }
      if(maxCol < myArray[index][i].col){
        maxCol = myArray[index][i].col;
      }
    }
    Point topLeft = Point(minCol, minRow);
    Point bottomRight = Point(maxCol, maxRow);
    //const unsigned char color[] = {rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255)};
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    const unsigned int boundary= 1;
   double rectWidth = maxCol - minCol;
   double rectHeight = maxRow - minRow;
   //Mat src;//Source image load here
   Rect R(minCol, minRow, rectWidth, rectHeight); //Create a rect
   Mat ROI = inputImage(R); //Crop the region of interest using above rect
   string sift_localize = "LocalizedImages/";
   sift_localize += "sift_localize_";
   sift_localize += to_string(localizedCount);
   sift_localize += ".jpg";
   rectangle(inputImage, topLeft, bottomRight, color);
   imwrite(sift_localize, ROI);
   localizedCount++;
   myArray.erase(myArray.begin()+index);
  }
    string testFileName = "GenerateImages/";
    testFileName += to_string(classLabel);
    testFileName += "_sift_2.png";
    imwrite(testFileName, inputImage);
    //inputImage.get_normalize(0, 255).save("sift_2.png");
}

// Calculate Sift decsriptors
vector<SiftDescriptor> calculateSift(CImg<double> inputImage){
CImg<double> grayImage = inputImage.get_RGBtoHSI().get_channel(2);
vector<SiftDescriptor> descriptors = Sift::compute_sift(grayImage);
for(int i=0; i<descriptors.size(); i++)
  {
    for(int j=0; j<5; j++)
      for(int k=0; k<5; k++)
        if(j==2 || k==2)
      for(int p=0; p<3; p++)
        inputImage(descriptors[i].col+k-1, descriptors[i].row+j-1, 0, p)=0;
  }
  string fileName = "sift_";
  //fileName += to_string(count);
  fileName += ".png";
  inputImage.get_normalize(0, 255).save(fileName.c_str());
  return descriptors;
}

/*
vector<vector<double> > getCameraParam(string color){

  vector<vector<double> > C(3, vector<double>(3));

  if(color == "color"){
    double fx_rgb = 5.1885790117450188e+02;
    double fy_rgb = 5.1946961112127485e+02;
    double cx_rgb = 3.2558244941119034e+02;
    double cy_rgb = 2.5373616633400465e+02;

    C[0][0] = fx_rgb;
    C[0][1] = 0;
    C[0][2] = cx_rgb;
    C[1][0] = 0;
    C[1][1] = fy_rgb;
    C[1][2] = cy_rgb;
    C[2][0] = 0;
    C[2][1] = 0;
    C[2][2] = 1;

  }
  else if(color == "depth"){

    double fx_d = 5.8262448167737955e+02;
    double fy_d = 5.8269103270988637e+02;
    double cx_d = 3.1304475870804731e+02;
    double cy_d = 2.3844389626620386e+02;

    C[0][0] = fx_d;
    C[0][1] = 0;
    C[0][2] = cx_d;
    C[1][0] = 0;
    C[1][1] = fy_d;
    C[1][2] = cy_d;
    C[2][0] = 0;
    C[2][1] = 0;
    C[2][2] = 1;

  }
  return C;
}

void cropCamera(vector<vector<double> > &C){
  C[0][2] = C[0][2] - 40;
  C[1][2] = C[1][2] - 45;
}

void getPointCloud(Mat z, vector<vector<double> > C, int s, vector<vector<double> > &x3, vector<vector<double> > &y3,
                  vector<vector<double> > &z3){
  int H = z.rows;
  int W = z.cols;
  int gr = 1;
  vector<vector<double> > xx(H, vector<double>(W));
  vector<vector<double> >yy(H, vector<double>(W));
  for(int i=0; i<H; i++){
    for(int j=0;j<W;j++){
      xx[i][j] = j+1;
    }
  }

  for(int i=0; i<H; i++){
    for(int j=0; j<W; j++){
      yy[i][j] = i+1;
    }
  }

  vector<vector<double> > temp_xx(H, vector<double>(W));
  for(int i=0; i<H; i++){
    for(int j=0;j<W;j++){
      temp_xx[i][j] = xx[i][j] - C[0][2];
    }
  }
  vector<vector<double> > temp_xx2(z.rows, vector<double>(z.cols));
  for(int i=0; i<z.rows; i++){
    for(int j=0;j<z.cols;j++){
      temp_xx2[i][j] = z.at<double>(i, j) / C[0][0];
    }
  }

  //vector<vector<double> > x3(H, vector<double>(W));
  for(int i=0; i<z.rows; i++){
    for(int j=0;j<z.cols;j++){
      x3[i][j] = temp_xx[i][j] * temp_xx2[i][j];
    }
  }

  vector<vector<double> > temp_yy(H, vector<double>(W));
  for(int i=0; i<H; i++){
    for(int j=0;j<W;j++){
      temp_yy[i][j] = yy[i][j] - C[1][2];
    }
  }
  vector<vector<double> > temp_yy2(z.rows, vector<double>(z.cols));
  for(int i=0; i<z.rows; i++){
    for(int j=0;j<z.cols;j++){
      temp_yy2[i][j] = z.at<double>(i, j) / C[1][1];
    }
  }

  //vector<vector<double> > y3(H, vector<double>(W));
  for(int i=0; i<z.rows; i++){
    for(int j=0;j<z.cols;j++){
      y3[i][j] = temp_yy[i][j] * temp_yy2[i][j];
    }
  }

  //vector<vector<double> > z3(H, vector<double>(W));

  for(int i=0; i<z.rows; i++){
    for(int j=0;j<z.cols;j++){
      z3[i][j] = z.at<double>(i, j);
    }
  }
}

void saveHHA(Mat depthImage, vector<vector<double> > C){

  Mat missingMask = Mat(depthImage.rows, depthImage.cols, CV_32F);

  for(int i=0; i<depthImage.rows; i++){
    for(int j=0; j<depthImage.cols; j++){
      if(depthImage.at<double>(i, j) != 0){
        missingMask.at<double>(i, j) = 0;
      }
      else{
        missingMask.at<double>(i, j) = 1;
      }
    }
  }

  gravityEst.angleThresh.push_back(45);
  gravityEst.angleThresh.push_back(15);

  gravityEst.iterations.push_back(5);
  gravityEst.iterations.push_back(5);

  gravityEst.initialAxis.push_back(0);
  gravityEst.initialAxis.push_back(1);
  gravityEst.initialAxis.push_back(0);

  Mat depthCM = Mat(depthImage.rows, depthImage.cols, CV_32F);

  for(int i=0; i<depthImage.rows; i++){
    for(int j=0; j<depthImage.cols; j++){
      depthCM.at<double>(i, j) = depthImage.at<double>(i, j)*100;
    }
  }

  vector<vector<double> > X(depthImage.rows, vector<double>(depthImage.cols));
  vector<vector<double> > Y(depthImage.rows, vector<double>(depthImage.cols));
  vector<vector<double> > Z(depthImage.rows, vector<double>(depthImage.cols));

  getPointCloud(depthCM, C, 1, X, Y, Z);

  for(int i=0; i<depthImage.rows; i++){
    for(int j=0;j<depthImage.cols;j++){
      if(missingMask.at<double>(i,j) == 1){
        X[i][j] = NAN;
        Y[i][j] = NAN;
        Z[i][j] = NAN;
      }
    }
  }

  Mat one_Z = Mat(depthImage.rows, depthImage.cols, CV_32F);
  Mat X_Z = Mat(depthImage.rows, depthImage.cols, CV_32F);
  Mat Y_Z = Mat(depthImage.rows, depthImage.cols, CV_32F);
  Mat X_ZZ = Mat(depthImage.rows, depthImage.cols, CV_32F);
  Mat Y_ZZ = Mat(depthImage.rows, depthImage.cols, CV_32F);

  for(int i=0; i<depthImage.rows; i++){
    for(int j=0; j<depthImage.cols;j++){
      one_Z.at<double>(i, j) = 1/(depthImage.at<double>(i, j));
      X_Z.at<double>(i, j) = X[i][j]/(depthImage.at<double>(i, j));
      Y_Z.at<double>(i, j) = Y[i][j]/(depthImage.at<double>(i, j));
      X_ZZ.at<double>(i, j) = X[i][j]/((depthImage.at<double>(i, j)) * (depthImage.at<double>(i, j)));
      Y_ZZ.at<double>(i, j) = Y[i][j]/((depthImage.at<double>(i, j)) * (depthImage.at<double>(i, j)));
    }
  }

  Mat one = Mat(depthImage.rows, depthImage.cols, CV_32F);

  for(int i=0; i<depthImage.rows; i++){
    for(int j=0; j<depthImage.cols; j++){
      if(depthImage.at<double>(i, j) != NAN){
        one.at<double>(i, j) = 1;
      }
      else{
        one.at<double>(i, j) = NAN;
      }
    }
  }

}
*/
int main(){

  Mat inputImage = imread("rgb.ppm", CV_LOAD_IMAGE_COLOR);
  //Mat depthIm = imread("depth.png", CV_LOAD_IMAGE_ANYDEPTH);
  Mat depthIm = imread("depth_2.png", CV_LOAD_IMAGE_ANYDEPTH);

  CImg<double> imageCImg("rgb.ppm");

  cout<<inputImage.at<double>(10, 10)<<endl;
  cout<<depthIm.at<double>(10, 10)<<endl;

/*
  cout<<inputImage_2.rows<<endl;
  cout<<inputImage_2.cols<<endl;
  cout<<inputImage_2.depth()<<endl;
  */

  depthIm.convertTo(depthIm, CV_64FC1);

  //Mat normals(depthIm.size(), CV_32FC3);
  Mat normals(depthIm.size(), CV_64FC3);

/*
  for(int i=0; i<depthIm.rows;i++){
  	for(int j=0; j<depthIm.cols;j++){
  		float dzdx = (depthIm.at<float>(i+1, j) - depthIm.at<float>(i-1, j)) / 2.0;
  		float dzdy = (depthIm.at<float>(i, j+1) - depthIm.at<float>(i, j-1)) / 2.0;

      //Vec3f d(-dzdx, -dzdy, 1.0f);
      //Vec3f n = normalize(d);

      Vec3f d;
      d[0] = -dzdx;
      d[1] = -dzdy;
      d[2] = 1.0;

      Vec3f n = normalize(d);
      normals.at<Vec3f>(i, j)[0] = n[0];
      normals.at<Vec3f>(i, j)[1] = n[1];
      normals.at<Vec3f>(i, j)[2] = n[2];

      cout<<"Rohil "<<n[0]<<endl;
      cout<<"Bansal "<<n[1]<<endl;
      cout<<"Ishita "<<n[2]<<endl;

      //cout<<normals<<endl;

  	}
  }
  */

  for(int i=1; i<depthIm.cols-1; i++){
    for(int j=1; j<depthIm.rows-1; j++){
      Vec3d t(i, j-1, depthIm.at<double>(j-1, i));
      Vec3d l(i-1, j, depthIm.at<double>(j, i-1));
      Vec3d c(i, j, depthIm.at<double>(j, i));
      Vec3d d = (l-c).cross(t-c);
      Vec3d n = normalize(d);
      normals.at<Vec3d>(j, i) = n;
    }
  }

/*
  namedWindow( "Display window", WINDOW_AUTOSIZE );
  imshow("Display window",normals);
  waitKey(0);
  */
  Mat singleChannel = Mat(depthIm.rows, depthIm.cols, CV_32F);

  for(int i=0; i<depthIm.rows;i++){
    for(int j=0; j<depthIm.cols;j++){
      singleChannel.at<float>(i, j) = depthIm.at<float>(i, j);
    }
  }

  vector<SiftDescriptor> siftDescriptors = calculateSift(imageCImg);
  aggClustering(siftDescriptors, inputImage, 2);


  Mat cov, mu;

  //calcCovarMatrix(singleChannel, cov, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS);

/*  cov = cov / (depthIm.rows-1);

  cout<<"COV:"<<endl;
  cout<<cov<<endl;

  cout<<"MU"<<endl;
  cout<<mu<<endl;
  */


  //vector<vector<double> > C = getCameraParam("color");
  //cropCamera(C);
  //saveHHA(inputImage_2, C);

  //getPointCloud(inputImage_2, C, 1);
  return 0;

}
