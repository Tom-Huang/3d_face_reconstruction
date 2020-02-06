#ifndef RGBDDATALOADER_H
#define RGBDDATALOADER_H

#endif  // RGBDDATALOADER_H

#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>

using namespace std;
class RGBDDataLoader {
 public:
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
  RGBDDataLoader() {}

  ~RGBDDataLoader() {}

  bool loadPCDData(const string filename) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_local(
        new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>("../data/cloud_1.pcd",
                                               *RGBDDataLoader::cloud) ==
        -1)  //* load the file
    {
      PCL_ERROR("Couldn't read the pcd file \n");
      return false;
    }
  }

 private:
};
