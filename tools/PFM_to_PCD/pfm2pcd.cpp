#include <opencv2/opencv.hpp>
#include <libconfig.h++>
#include <iostream>
#include <thread>

// pcl headers
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>

using namespace libconfig;
using namespace std;
using namespace cv;
using namespace pcl;

//Function to read Qmatrix
int readQMatrix(std::string path, Mat *Qmatrix)
{
    Config cfg;
    try{
        cfg.readFile(path.c_str());
    }
    catch (const FileIOException &fioex)
    {
        std::cerr << "I/O error while reading file." << std::endl;
        return (0);
    }
    catch (const ParseException &pex)
    {
        std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine() << " - " << pex.getError() << std::endl;
        return (0);
    }
    try
    {
        string name = cfg.lookup("name");
        stringstream ss(name);
        vector<string> result;
        cout.precision(10);
        while (ss.good())
        {
            string val;
            getline(ss, val, ',');
            result.push_back(val);
        }
        *Qmatrix = Mat::zeros(4, 4, CV_64F);
        int cnt = 0;
        for (int row = 0; row < 4; row++)
            for (int col = 0; col < 4; col++)
            {
                Qmatrix->at<double>(row, col) = std::stod(result[cnt]);
                cnt++;
            }
    }
    catch (const SettingNotFoundException &nfex)
    {
        cerr << "No 'name' setting in configuration file." << endl;
        return 0;
    }
    return 1;
}
// Read the Colour Camera Matrix
int readColorMatrix(string path,Mat *colorMatrix)
{
    Config cfg;
    try{
        cfg.readFile(path.c_str());
    }
    catch(const FileIOException &fioex)
    {
        std::cerr << "I/O error while reading file." << std::endl;
        return 0;
    }
    catch(const ParseException &pex)
    {
        std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
            << " - " << pex.getError() << std::endl;
        return 0;
    }

    try
    {
        string name = cfg.lookup("Color_camera_Matrix");
        stringstream ss(name);
        vector<string> result;
        cout.precision(10);
        while (ss.good ()) {
            string val;
            getline (ss, val, ',');
            result.push_back(val);
        }
        *colorMatrix = Mat::zeros(2, 2, CV_64F);
        int cnt=0;
        for ( int row=0; row<2; row++)
            for (int col=0; col<2; col++) {
                colorMatrix->at<double>(row, col) = std::stod(result[cnt]);
                cnt++;
            }
    }

    catch(const SettingNotFoundException &nfex)
    {
        cerr << "No 'Color_camera_Matrix' setting in configuration file." << endl;
        return 0;
    }
    return 1;
}

// Read the Transformation matrix
int readcolorCameraRotationTranslation(string path, Mat *depthColorMatrix)
{
    Config cfg;
    try{
        cfg.readFile(path.c_str());
    }
    catch(const FileIOException &fioex)
    {
        std::cerr << "I/O error while reading file." << std::endl;
        return(0);
    }
    catch(const ParseException &pex)
    {
        std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
            << " - " << pex.getError() << std::endl;
        return(0);
    }
    try
    {
        string RotationTranslation = cfg.lookup("Rotation_Translation_Matrix");
        stringstream ss(RotationTranslation);
        vector<string> result;
        cout.precision(10);
        while (ss.good ()) {
            string val;
            getline (ss, val, ',');
            result.push_back(val);
        }
        *depthColorMatrix = Mat::zeros(4, 4, CV_64F);
        int cnt=0;
        for ( int row=0; row<4; row++)
        {
            for (int col=0; col<4; col++) {
                depthColorMatrix->at<double>(row, col) = std::stod(result[cnt]);
                cnt++;
            }
        }
    }
    catch(const SettingNotFoundException &nfex)
    {
        cerr << "No 'colorCameraRotationTranslation' setting in configuration file." << endl;
        return 0;
    }
    return 1;
}

int main(int argc, char* argv[])
{   
    if(strcmp(argv[1],"--help") == 0) 
    {
        cout << "PFMtoPCD DESCRIPTION - Used to convert a PFM file to Point Cloud or to a rgb point cloud\n\nMandatory Arguments:\n1. -n for Binary Cloud and -c for RGB Cloud\n" << endl;
        cout << "Parameters for Binary Cloud (-n/-normal):\n1. Path of PFM file\n2. Path of Qmatrix file" << endl;
        cout << "Parameters for RGB Cloud (-c/-colour):\n1. Path of PFM file\n2. Path of Qmatrix file\n3. Path of RGB Image\n" << endl;
        cout << "Along with this the path of the destination PCD file could also be added. If no destination file is provided the point cloud will be saved with a default name." << endl;
        return 0;
    }

    string pfmPath;
    string matrixPath;
    string pcdPath;
    string colorImagePath;

//  For binary point cloud creation
    if((strcmp(argv[1],"-n") == 0) || (strcmp(argv[1],"-normal") == 0)){
        if(argc < 4){
            cout << "Minimum required parameters are not met. Kindly use --help for further details..." << endl;
            return 0;
        }
        else if(argc == 4 || argc == 5 ){
            pfmPath = argv[2];
            matrixPath = argv[3];
            if(argc == 5)
                pcdPath = argv[4];          // If pcd path is not given then pcd will be saved in the current directory.
            else
                pcdPath = "./pointCloud.pcd";
        }
        else{
            cout << "More than maximum number of arguments required. Kindly use --help for further details" << endl;
            return 0;
        }
    }
//  For RGB Cloud creation
    else if((strcmp(argv[1],"-c") == 0) || (strcmp(argv[1],"-colour") == 0)){
        if(argc < 5){
            cout << "Minimum required parameters are not set. Kindly use --help for further details..." << endl;
            return 0;
        }
        if(argc == 5 || argc == 6 ){
            pfmPath = argv[2];
            matrixPath = argv[3];
            colorImagePath = argv[4];
            if(argc == 6)
                pcdPath = argv[5];          // If pcd path is not given then pcd will be saved in the current directory.
            else
                pcdPath = "./RGBpointCloud.pcd";
        }
        else{
            cout << "More than maximum number of arguments required. Kindly use --help for further details" << endl;
            return 0;
        }
    }
    else{
        cout << "kindly choose an option: -n for normal cloud; -c for colour cloud conversion" << endl;
        return 0;
    }

//  Read the pfm and change to 64 bit image
    Mat depthImage;
    Mat pfmImage = imread(pfmPath, IMREAD_UNCHANGED);
    pfmImage.convertTo(depthImage, CV_64F);
    
//  Read the Qmatrix to obtain the focal length and principal points
    Mat Qmatrix;
    int Qmatrix_read = readQMatrix(matrixPath, &Qmatrix);
    if(Qmatrix_read == 0){
        cout << "Kindly check the Qmatrix and try again... " << endl;
        return 0;
    }
    double focal_length = Qmatrix.at<double>(2,3);
    double cx = Qmatrix.at<double>(0,3);
    double cy = Qmatrix.at<double>(1,3);
    
//  Construction of 3D Point Cloud
    PointCloud<PointXYZ>::Ptr pointCloud(new PointCloud<PointXYZ>);
    for (int j = 0; j < depthImage.size().height; j++)
    {
        for (int i = 0; i < depthImage.size().width; i++)
        {
            PointXYZ point;
            point.z = depthImage.at<double>(Point(i, j))/1000;
            point.x = ((i + cx) / focal_length) * point.z;
            point.y = ((j + cy) / focal_length) * point.z;
            pointCloud->points.push_back(point);
        }
    }
    pointCloud->width = depthImage.size().width;
    pointCloud->height = depthImage.size().height;

    if((strcmp(argv[1],"-c") == 0) || strcmp(argv[1],"-colour") == 0){
//  Read the Colour camera matrix
        Mat colour2DImage = imread(colorImagePath);
        Mat colorMatrix, depthColorMatrix;
        Eigen::Matrix4f transformationMatrix = Eigen::Matrix4f::Identity();
        int colorMatrix_read = readColorMatrix(matrixPath, &colorMatrix);
        int depthColorMatrix_read = readcolorCameraRotationTranslation(matrixPath, &depthColorMatrix);
        if(colorMatrix_read == 0 || depthColorMatrix_read == 0)
        {
            cout << "Kindly check the colour camera and transformation matrix and try again..." << endl;
            return 0;
        }
        transformationMatrix(0,3) = depthColorMatrix.at<double>(0, 3);
        transformationMatrix(1,3) = depthColorMatrix.at<double>(1, 3);
        transformationMatrix(2,3) = depthColorMatrix.at<double>(2, 3);
        transformationMatrix(0,0) = depthColorMatrix.at<double>(0, 0);
        transformationMatrix(0,1) = depthColorMatrix.at<double>(0, 1);
        transformationMatrix(0,2) = depthColorMatrix.at<double>(0, 2);
        transformationMatrix(1,0) = depthColorMatrix.at<double>(1, 0);
        transformationMatrix(1,1) = depthColorMatrix.at<double>(1, 1);
        transformationMatrix(1,2) = depthColorMatrix.at<double>(1, 2);
        transformationMatrix(2,0) = depthColorMatrix.at<double>(2, 0);
        transformationMatrix(2,1) = depthColorMatrix.at<double>(2, 1);
        transformationMatrix(2,2) = depthColorMatrix.at<double>(2, 2);

//  Transform the cloud to colour camera's view
        PointCloud<PointXYZ>::Ptr colourCameraCloud(new PointCloud<PointXYZ>);
        transformPointCloud (*pointCloud, *colourCameraCloud, transformationMatrix);
        
//  Add colour value to the cloud from the rgb Image
        PointCloud<pcl::PointXYZRGB>::Ptr rgbCloud(new pcl::PointCloud<PointXYZRGB>);
        for (size_t i = 0; i < colourCameraCloud->points.size(); i++)
        {
            double x,y;
            int x_actual,y_actual;

            if(colourCameraCloud->points[i].z != 0 )
            {
                x = ((colourCameraCloud->points[i].x * colorMatrix.at<double>(0, 0)) / colourCameraCloud->points[i].z) - colorMatrix.at<double>(1, 0);
                y = ((colourCameraCloud->points[i].y * colorMatrix.at<double>(0, 1)) / colourCameraCloud->points[i].z) - colorMatrix.at<double>(1, 1);

                if(!isnan(x) && !isnan(y))
                {
                    x_actual = abs(int(round(x)));
                    y_actual = abs(int(round(y)));
                    PointXYZRGB current_point;
                    current_point.x = colourCameraCloud->points[i].x;
                    current_point.y = colourCameraCloud->points[i].y;
                    current_point.z = colourCameraCloud->points[i].z;
                    current_point.r = colour2DImage.at<Vec3b>(y_actual,x_actual)[2];
                    current_point.g = colour2DImage.at<Vec3b>(y_actual,x_actual)[1];
                    current_point.b = colour2DImage.at<Vec3b>(y_actual,x_actual)[0];
                    rgbCloud->points.push_back(current_point);
                }
            }
        }
        rgbCloud ->points.resize(colourCameraCloud->points.size());
        rgbCloud->width = rgbCloud->points.size();
        rgbCloud->height = 1;
        io::savePCDFileASCII (pcdPath, *rgbCloud);
    }
    else
        io::savePCDFileASCII (pcdPath, *pointCloud);
    return 0;
}