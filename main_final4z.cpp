#include <fstream>
#include <iostream>

#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "BaselModelReader.h"
#include "Eigen.h"
#include "ICPOptimizer.h"
#include "PointCloud.h"
#include "ProcrustesAligner.h"
#include "SimpleMesh.h"
#include "VirtualSensor.h"
#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"

//// TODO PROJECT: include opencv
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/surface_matching.hpp"
#include "opencv2/surface_matching/ppf_helpers.hpp"

#include "dlib/gui_widgets.h"
#include "dlib/image_io.h"
#include "dlib/image_processing.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing/render_face_detections.h"

#define USE_POINT_TO_PLANE 1

using namespace dlib;
using namespace cv;

#define RUN_PROCRUSTES 1
#define RUN_SHAPE_ICP 0
#define RUN_SEQUENCE_ICP 0

#define PRIN_COMP_NUM 10

// bool WriteMesh(Vertex* vertices, unsigned int width, unsigned int height,
//               const std::string& filename) {
//  float edgeThreshold = 0.01f;  // 1cm

//  // TODO 2: use the OFF file format to save the vertices grid
//  // (http://www.geomview.org/docs/html/OFF.html)
//  // - have a look at the "off_sample.off" file to see how to store the
//  vertices
//  // and triangles
//  // - for debugging we recommend to first only write out the vertices (set
//  the
//  // number of faces to zero)
//  // - for simplicity write every vertex to file, even if it is not valid
//  // (position.x() == MINF) (note that all vertices in the off file have to be
//  // valid, thus, if a point is not valid write out a dummy point like
//  (0,0,0))
//  // - use a simple triangulation exploiting the grid structure (neighboring
//  // vertices build a triangle, two triangles per grid cell)
//  // - you can use an arbitrary triangulation of the cells, but make sure that
//  // the triangles are consistently oriented
//  // - only write triangles with valid vertices and an edge length smaller
//  then
//  // edgeThreshold

//  // TODO: Get number of vertices
//  unsigned int nVertices = width * height;

//  // TODO: Get number of faces
//  // This number is not correct, later we will rewrite this line
//  unsigned nFaces = 0;
//  unsigned int* faces = new unsigned int[3 * 2 * nVertices];
//  unsigned int face_ind = 0;
//  for (unsigned int i = 0; i < height - 1; i++) {
//    for (unsigned int j = 0; j < width - 1; j++) {
//      // Vertices in one grid
//      Vector4f upperleft = vertices[i * width + j].position;
//      Vector4f lowerleft = vertices[(i + 1) * width + j].position;
//      Vector4f upperright = vertices[i * width + j + 1].position;
//      Vector4f lowerright = vertices[(i + 1) * width + j + 1].position;

//      // Edges in one grid (two faces)
//      Vector4f left = lowerleft - upperleft;
//      Vector4f up = upperleft - upperright;
//      Vector4f diagonal = upperright - lowerleft;
//      Vector4f right = upperright - lowerright;
//      Vector4f down = lowerright - lowerleft;

//      // Measure the length of edges to decide whether to save
//      if (upperleft[0] != MINF && upperright[0] != MINF &&
//          lowerleft[0] != MINF && left.norm() < edgeThreshold &&
//          up.norm() < edgeThreshold && diagonal.norm() < edgeThreshold) {
//        // std::cout << i << std::endl;
//        // std::cout << i * width + j << std::endl;
//        faces[face_ind++] = i * width + j;
//        faces[face_ind++] = (i + 1) * width + j;
//        faces[face_ind++] = i * width + j + 1;
//        nFaces += 1;
//      }
//      if (lowerright[0] != MINF && upperright[0] != MINF &&
//          lowerleft[0] != MINF && right.norm() < edgeThreshold &&
//          down.norm() < edgeThreshold && diagonal.norm() < edgeThreshold) {
//        faces[face_ind++] = (i + 1) * width + j;
//        faces[face_ind++] = (i + 1) * width + j + 1;
//        faces[face_ind++] = i * width + j + 1;
//        nFaces += 1;
//      }
//    }
//  }

//  // Write off file
//  std::ofstream outFile(filename);
//  if (!outFile.is_open()) return false;

//  // write header
//  outFile << "COFF" << std::endl;
//  outFile << nVertices << " " << nFaces << " 0" << std::endl;

//  // TODO: save vertices
//  for (int i = 0; i < nVertices; i++) {
//    if (vertices[i].position[0] != MINF) {
//      outFile << vertices[i].position[0] << " " << vertices[i].position[1]
//              << " " << vertices[i].position[2] << " "
//              << (int)vertices[i].color[0] << " " << (int)vertices[i].color[1]
//              << " " << (int)vertices[i].color[2] << " "
//              << (int)vertices[i].color[3] << std::endl;
//    } else {
//      outFile << "0 0 0 " << (int)vertices[i].color[0] << " "
//              << (int)vertices[i].color[1] << " " << (int)vertices[i].color[2]
//              << " " << (int)vertices[i].color[3] << std::endl;
//    }
//  }

//  // TODO: save faces
//  for (int i = 0; i < nFaces; i++) {
//    outFile << "3 " << faces[i * 3] << " " << faces[i * 3 + 1] << " "
//            << faces[i * 3 + 2] << std::endl;
//  }
//  delete faces;

//  // close file
//  outFile.close();

//  std::ifstream inFile(filename);
//  std::stringstream line;
//  std::string line_str;
//  line << nVertices << " " << nFaces << " 0" << std::endl;
//  line_str = line.str();

//  return true;
//}

using namespace std;
bool WriteMesh(Vertex* vertices, unsigned int width, unsigned int height,
               const std::string& filename) {
  float edgeThreshold = 0.01f;  // 1cm

  // TODO 2: use the OFF file format to save the vertices grid
  // (http://www.geomview.org/docs/html/OFF.html)
  // - have a look at the "off_sample.off" file to see how to store the vertices
  // and triangles
  // - for debugging we recommend to first only write out the vertices (set the
  // number of faces to zero)
  // - for simplicity write every vertex to file, even if it is not valid
  // (position.x() == MINF) (note that all vertices in the off file have to be
  // valid, thus, if a point is not valid write out a dummy point like (0,0,0))
  // - use a simple triangulation exploiting the grid structure (neighboring
  // vertices build a triangle, two triangles per grid cell)
  // - you can use an arbitrary triangulation of the cells, but make sure that
  // the triangles are consistently oriented
  // - only write triangles with valid vertices and an edge length smaller then
  // edgeThreshold

  // TODO: Get number of vertices
  unsigned int nVertices = width * height;

  // TODO: Get number of faces
  // This number is not correct, later we will rewrite this line
  unsigned nFaces = 0;
  unsigned int* faces = new unsigned int[3 * 2 * nVertices];
  unsigned int face_ind = 0;
  for (unsigned int i = 0; i < height - 1; i++) {
    for (unsigned int j = 0; j < width - 1; j++) {
      // Vertices in one grid
      Vector4f upperleft = vertices[i * width + j].position;
      Vector4f lowerleft = vertices[(i + 1) * width + j].position;
      Vector4f upperright = vertices[i * width + j + 1].position;
      Vector4f lowerright = vertices[(i + 1) * width + j + 1].position;

      // Edges in one grid (two faces)
      Vector4f left = lowerleft - upperleft;
      Vector4f up = upperleft - upperright;
      Vector4f diagonal = upperright - lowerleft;
      Vector4f right = upperright - lowerright;
      Vector4f down = lowerright - lowerleft;

      // Measure the length of edges to decide whether to save
      if (upperleft[0] != MINF && upperright[0] != MINF &&
          lowerleft[0] != MINF && left.norm() < edgeThreshold &&
          up.norm() < edgeThreshold && diagonal.norm() < edgeThreshold) {
        // std::cout << i << std::endl;
        // std::cout << i * width + j << std::endl;
        faces[face_ind++] = i * width + j;
        faces[face_ind++] = (i + 1) * width + j;
        faces[face_ind++] = i * width + j + 1;
        nFaces += 1;
      }
      if (lowerright[0] != MINF && upperright[0] != MINF &&
          lowerleft[0] != MINF && right.norm() < edgeThreshold &&
          down.norm() < edgeThreshold && diagonal.norm() < edgeThreshold) {
        faces[face_ind++] = (i + 1) * width + j;
        faces[face_ind++] = (i + 1) * width + j + 1;
        faces[face_ind++] = i * width + j + 1;
        nFaces += 1;
      }
    }
  }

  // Write off file
  std::ofstream outFile(filename);
  if (!outFile.is_open()) return false;

  // write header
  outFile << "COFF" << std::endl;
  outFile << nVertices << " " << nFaces << " 0" << std::endl;

  // TODO: save vertices
  for (int i = 0; i < nVertices; i++) {
    if (vertices[i].position[0] != MINF) {
      outFile << vertices[i].position[0] << " " << vertices[i].position[1]
              << " " << vertices[i].position[2] << " "
              << (int)vertices[i].color[0] << " " << (int)vertices[i].color[1]
              << " " << (int)vertices[i].color[2] << " "
              << (int)vertices[i].color[3] << std::endl;
    } else {
      outFile << "0 0 0 " << (int)vertices[i].color[0] << " "
              << (int)vertices[i].color[1] << " " << (int)vertices[i].color[2]
              << " " << (int)vertices[i].color[3] << std::endl;
    }
  }

  // TODO: save faces
  for (int i = 0; i < nFaces; i++) {
    outFile << "3 " << faces[i * 3] << " " << faces[i * 3 + 1] << " "
            << faces[i * 3 + 2] << std::endl;
  }
  delete faces;

  // close file
  outFile.close();

  std::ifstream inFile(filename);
  std::stringstream line;
  std::string line_str;
  line << nVertices << " " << nFaces << " 0" << std::endl;
  line_str = line.str();

  return true;
}

// TODO PROJECT HUANG: get the transformation matrix and scale factor from
// source to target
int getProcrustesTransformation(const double& initial_scale,
                                Matrix4d& estimatedPose,
                                double& scale_source2target) {
  // Load the source and target mesh.
  // const std::string filenameSource = PROJECT_DIR +
  // std::string("/data/bs.off");
  const std::string filenameTarget =
      PROJECT_DIR + std::string("/data/landmarks.off");

  SimpleMesh targetMesh;
  if (!targetMesh.loadMesh(filenameTarget)) {
    std::cout << "Mesh file wasn't read successfully at location: "
              << filenameTarget << std::endl;
    return -1;
  }

  std::cout << "target mesh vertices number: "
            << targetMesh.getVertices().size() << std::endl;

  // Fill in the matched points: sourcePoints[i] is matched with
  // targetPoints[i].
  std::vector<Vector3d> sourcePoints;

  sourcePoints.push_back(Vector3d(-745.985f, -74873.7f, 102048.0f) *
                         initial_scale);
  sourcePoints.push_back(Vector3d(-18835.9f, 49239.7f, 105833.0f) *
                         initial_scale);
  sourcePoints.push_back(Vector3d(17075.8f, 49594.4f, 106127.0f) *
                         initial_scale);
  sourcePoints.push_back(Vector3d(48.2984f, 3569.18f, 131652.0f) *
                         initial_scale);
  sourcePoints.push_back(Vector3d(47637.9f, 32988.7f, 83889.6f) *
                         initial_scale);
  sourcePoints.push_back(Vector3d(-26253.5f, -33360.6f, 98470.2f) *
                         initial_scale);

  std::vector<Vector3d> targetPoints;

  std::vector<Vertex> landmarks = targetMesh.getVertices();
  std::cout << landmarks.size() << std::endl;
  for (const auto ver : landmarks) {
    // std::cout << "target points: " << ver.position.block(0, 0, 3, 1)
    // << std::endl;
    targetPoints.push_back(ver.position.block(0, 0, 3, 1).cast<double>());
  }

  // Estimate the pose from source to target mesh with Procrustes alignment.
  ProcrustesAligner aligner;

  estimatedPose =
      aligner.estimatePose(sourcePoints, targetPoints, scale_source2target);

  return 0;
}
// TODO PROJECT HUANG: align basel model with depth map, output transformed
// points' matrix
void alignBaselwithDepth(const Eigen::Matrix3Xd& sourcePointsMatrix,
                         const Eigen::Matrix4d& transformation,
                         const double& scale_source2target,
                         Eigen::Matrix3Xd& transformed_sourcePointsMatrix) {
  Eigen::Matrix3Xd scaled_sourcePointsMatrix(3, sourcePointsMatrix.cols());

  // std::cout << "scaled first points: " << scaled_sourcePointsMatrix.col(0)
  // << std::endl;

  Eigen::Matrix3d R = transformation.block(0, 0, 3, 3);
  Eigen::Vector3d t = transformation.block(0, 3, 3, 1);

  transformed_sourcePointsMatrix =
      (scale_source2target * R * sourcePointsMatrix).colwise() + t;

  //  std::cout << "scaled matrix cols num: " <<
  //  scaled_sourcePointsMatrix.cols()
  //            << std::endl;
  //  std::cout << "transformed matrix cols num: "
  //            << transformed_sourcePointsMatrix.cols() << std::endl;
  //  std::cout << "one example: " << transformed_sourcePointsMatrix.col(0)
  //            << std::endl;
}
// TODO PROJECT HUANG: ICP with opencv

using namespace std;

void Procrustes(Eigen::Matrix4d& T_pro, double& scale_bs2gt) {
  // Load the source and target mesh.
  const std::string filenameSource = PROJECT_DIR + std::string("/data/bs.off");
  const std::string filenameTarget =
      PROJECT_DIR + std::string("/data/mesh_entire_face.off");

  SimpleMesh sourceMesh;
  if (!sourceMesh.loadMesh(filenameSource)) {
    std::cout << "Mesh file wasn't read successfully at location: "
              << filenameSource << std::endl;
    return;
  }

  SimpleMesh targetMesh;
  if (!targetMesh.loadMesh(filenameTarget)) {
    std::cout << "Mesh file wasn't read successfully at location: "
              << filenameTarget << std::endl;
    return;
  }

  // scale basel model because the model coordinate values are too big
  SimpleMesh scaled_sourceMesh;
  // store basel model vertices in 3XN matrix
  std::vector<Vertex> sourcePoints = sourceMesh.getVertices();
  std::vector<Triangle> sourceTriangles = sourceMesh.getTriangles();
  std::vector<Vertex> scaled_sourcePoints;
  Matrix3Xd sourcePointsMatrix(3, sourcePoints.size());
  Matrix3Xd scaled_sourcePointsMatrix(3, sourcePoints.size());

  for (int i = 0; i < sourcePoints.size(); i++) {
    sourcePointsMatrix.col(i) =
        sourcePoints[i].position.block(0, 0, 3, 1).cast<double>();
  }

  // get the transformation from basel to depth map using manually selected
  // landmarks on basel model and detected corresponding landmarks on depth
  // map
  Matrix4d estimatePose;
  double scale_source2target;
  double initial_scale = 0.0001f;
  {
    //      scaled_sourcePointsMatrix = sourcePointsMatrix * initial_scale;

    //      for (int i = 0; i < scaled_sourcePointsMatrix.cols(); i++) {
    //          Vertex ver;
    //          ver.position.block(0, 0, 3, 1) =
    //              scaled_sourcePointsMatrix.col(i).cast<float>();
    //          ver.position(3) = float(1);
    //          scaled_sourceMesh.addVertex(ver);
    //      }
    //      scaled_sourceMesh.writeMesh(PROJECT_DIR +
    //      std::string("/data/bs_scaled.off"));
  }
  getProcrustesTransformation(initial_scale, estimatePose, scale_source2target);
  scale_source2target *= initial_scale;

  // transform basel model to align with depth
  Matrix3Xd transformed_sourcePointsMatrix(3, sourcePointsMatrix.cols());
  alignBaselwithDepth(sourcePointsMatrix, estimatePose, scale_source2target,
                      transformed_sourcePointsMatrix);

  {
    //  // store transformed key points selected on basel model for debugging
    //  std::vector<Vector3d> LandmarksPoints;
    //  LandmarksPoints.push_back(Vector3d(-745.985f, -74873.7f, 102048.0f) *
    //                            0.0001f);
    //  LandmarksPoints.push_back(Vector3d(-18835.9f, 49239.7f, 105833.0f) *
    //                            initial_scale);
    //  LandmarksPoints.push_back(Vector3d(17075.8f, 49594.4f, 106127.0f) *
    //                            initial_scale);
    //  LandmarksPoints.push_back(Vector3d(48.2984f, 3569.18f, 131652.0f) *
    //                            initial_scale);
    //  LandmarksPoints.push_back(Vector3d(47637.9f, 32988.7f, 83889.6f) *
    //                            initial_scale);
    //  LandmarksPoints.push_back(Vector3d(-26253.5f, -33360.6f, 98470.2f) *
    //  0.0001f); SimpleMesh landmarksMesh; Eigen::Matrix3d R =
    //  estimatePose.block(0, 0, 3, 3); Eigen::Vector3d t =
    //  estimatePose.block(0, 3, 3, 1); for (int i = 0; i <
    //  LandmarksPoints.size(); i++) {
    //    Vertex ver;
    //    ver.position.block(0, 0, 3, 1) =
    //        (scale_source2target * R * LandmarksPoints[i] + t).cast<float>();
    //    ver.position(3) = float(1);
    //    landmarksMesh.addVertex(ver);
    //  }
    //  landmarksMesh.writeMesh(PROJECT_DIR +
    //                          std::string("/data/bs_tf_landmarks.off"));
  }

  SimpleMesh transformed_sourcePointsMesh;
  transformed_sourcePointsMesh.clear();
  std::cout << "source points matrix columns num: " << sourcePointsMatrix.cols()
            << std::endl;
  std::cout << "transformed source points matrix columns num: "
            << transformed_sourcePointsMatrix.cols() << std::endl;

  std::cout << scale_source2target << std::endl;

  std::cout << estimatePose << std::endl;

  for (int i = 0; i < transformed_sourcePointsMatrix.cols(); i++) {
    Vertex ver;
    Vector3f vec = transformed_sourcePointsMatrix.col(i).cast<float>();
    ver.position.block(0, 0, 3, 1) = vec;
    ver.position(3) = float(1);
    ver.color = sourcePoints[i].color;
    transformed_sourcePointsMesh.addVertex(ver);
  }

  for (int i = 0; i < sourceTriangles.size(); i++) {
    transformed_sourcePointsMesh.addFace(sourceTriangles[i].idx0,
                                         sourceTriangles[i].idx1,
                                         sourceTriangles[i].idx2);
  }

  if (!transformed_sourcePointsMesh.writeMesh(
          PROJECT_DIR + std::string("/data/transformed_bs.off"))) {
    std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
    return;
  }

  T_pro = estimatePose;
  cout << "T_pro: " << endl << T_pro << endl;
  scale_bs2gt = scale_source2target;
}

void getICPTransformation(Eigen::Matrix4d& T_icp) {
  const std::string filenameSource =
      PROJECT_DIR + std::string("/data/transformed_bs.off");
  const std::string filenameTarget =
      PROJECT_DIR + std::string("/data/entire_face.off");
  const std::string filenameMask(PROJECT_DIR +
                                 std::string("/data/face_seg_mask.txt"));

  int mask[53490];
  ifstream f(filenameMask, std::ios::in | std::ios::binary);
  if (!f.is_open()) {
    cout << "load matrix file failed!" << endl;
    return;
  }
  double data;
  for (int r = 0; r < 53490; r++) {
    f >> data;
    std::cout << data << std::endl;
    mask[r] = int(data);
  }
  f.close();

  SimpleMesh sourceMesh;
  if (!sourceMesh.loadMesh(filenameSource)) {
    std::cout << "Mesh file wasn't read successfully at location: "
              << filenameSource << std::endl;
  }

  SimpleMesh targetMesh;
  if (!targetMesh.loadMesh(filenameTarget)) {
    std::cout << "Mesh file wasn't read successfully at location: "
              << filenameTarget << std::endl;
  }

  SimpleMesh filtersourceMesh;
  SimpleMesh sourceMesh_seg0;
  SimpleMesh sourceMesh_seg1;
  SimpleMesh sourceMesh_seg2;
  SimpleMesh sourceMesh_seg3;
  SimpleMesh sourceMesh_other;
  std::vector<Vertex> sourceVertices = sourceMesh.getVertices();
  if (sourceVertices.size() != 53490) {
    std::cout << "Mask size does not correspond to vertices number!"
              << std::endl;
    return;
  } else {
    for (int i = 0; i < sourceVertices.size(); i++) {
      Vertex ver;
      ver = sourceVertices[i];
      if (mask[i] == 0) {
        sourceMesh_seg0.addVertex(ver);
      }
      if (mask[i] == 1) {
        sourceMesh_seg1.addVertex(ver);
      }
      if (mask[i] == 2) {
        sourceMesh_seg2.addVertex(ver);
      }
      if (mask[i] == 3) {
        sourceMesh_seg3.addVertex(ver);
      }
      if (i > 42926) {
        sourceMesh_other.addVertex(ver);
      }
      if (mask[i] < 3) {
        filtersourceMesh.addVertex(ver);
      }
    }
  }

  if (!sourceMesh_seg0.writeMesh(PROJECT_DIR +
                                 std::string("/data/face_seg0.off"))) {
    std::cout << "write face seg mesh failed!" << std::endl;
  }

  if (!sourceMesh_seg1.writeMesh(PROJECT_DIR +
                                 std::string("/data/face_seg1.off"))) {
    std::cout << "write face seg mesh failed!" << std::endl;
  }
  if (!sourceMesh_seg2.writeMesh(PROJECT_DIR +
                                 std::string("/data/face_seg2.off"))) {
    std::cout << "write face seg mesh failed!" << std::endl;
  }
  if (!sourceMesh_seg3.writeMesh(PROJECT_DIR +
                                 std::string("/data/face_seg3.off"))) {
    std::cout << "write face seg mesh failed!" << std::endl;
  }
  if (!sourceMesh_other.writeMesh(PROJECT_DIR +
                                  std::string("/data/face_other.off"))) {
    std::cout << "write face seg mesh failed!" << std::endl;
  }

  std::cout << "write face segmentation files success!" << std::endl;

  // Estimate the pose from source to target mesh with ICP optimization.
  ICPOptimizer optimizer;
  optimizer.setMatchingMaxDistance(0.0003f);
  if (USE_POINT_TO_PLANE) {
    optimizer.usePointToPlaneConstraints(true);
    optimizer.setNbOfIterations(10);
  } else {
    optimizer.usePointToPlaneConstraints(false);
    optimizer.setNbOfIterations(20);
  }

  PointCloud source{filtersourceMesh};  // sourceMesh
  PointCloud target{targetMesh};

  Matrix4f estimatedPose = optimizer.estimatePose(source, target);
  std::cout << estimatedPose << std::endl;

  // Visualize the resulting joined mesh. We add triangulated spheres for point
  // matches.
  SimpleMesh resultingMesh = SimpleMesh::joinMeshes(
      filtersourceMesh, targetMesh, estimatedPose);  // sourceMesh
  resultingMesh.writeMesh(PROJECT_DIR + std::string("/data/result_6.off"));
  std::cout << "Resulting mesh written." << std::endl;

  T_icp = estimatedPose.cast<double>();
}

struct RegularizerCostFunctor {
  RegularizerCostFunctor(int para_num_, double weight_)
      : para_num(para_num_), weight(weight_) {}

  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    for (int i = 0; i < para_num; i++) {
      residual[i] = T(weight) * x[i];
    }
    return true;
  }

 private:
  const int& para_num;
  const double& weight;
};
struct RGBDCostFunction {
  RGBDCostFunction(const MatrixXd& neutral_xyz_, const MatrixXd& neutral_rgb_,
                   const MatrixXd& pc_xyz_, const MatrixXd& pc_rgb_,
                   const MatrixXd& shapeEV_, const MatrixXd& texEV_,
                   const MatrixXd& depth_map_, const MatrixXd& r_map_,
                   const MatrixXd& g_map_, const MatrixXd& b_map_,
                   const MatrixXd& P_)
      : neutral_xyz(neutral_xyz_),
        neutral_rgb(neutral_rgb_),
        pc_xyz(pc_xyz_),
        pc_rgb(pc_rgb_),
        shapeEV(shapeEV_),
        texEV(texEV_),
        depth_map(depth_map_),
        r_map(r_map_),
        g_map(g_map_),
        b_map(b_map_),
        P(P_) {}

  template <typename T>
  bool operator()(const T* const alpha, const T* const beta,
                  T* residual) const {
    //    std::cout << "enter operator!" << std::endl;
    Eigen::Map<Matrix<T, PRIN_COMP_NUM, 1> const> const alpha_v(alpha);
    Eigen::Map<Matrix<T, PRIN_COMP_NUM, 1> const> const beta_v(beta);

    //    Matrix<T, PRIN_COMP_NUM, 1> shapeEV_cast = (shapeEV).cast<T>();
    //    Matrix<T, PRIN_COMP_NUM, 1> texEV_cast =
    //    texEV.block(0,0,PRIN_COMP_NUM,1);

    //    std::cout << "finished mapping array to matrix" << std::endl;

    // cout << "xyz operator in " << neutral_xyz(0,0) << ", " <<
    // neutral_xyz(1,0) << ", " << neutral_xyz(2,0) << endl;
    // cout << "rgb operator in " << neutral_rgb(0,0) << ", " <<
    // neutral_rgb(1,0) << ", " << neutral_rgb(2,0) << endl;
    //
    //    cout << "alpha shape " << alpha_v.rows() << " ," << alpha_v.cols() <<
    //    endl; cout << "shapeEV shape " << shapeEV_cast.rows() << " ," <<
    //    shapeEV_cast.cols() << endl; cout << "shapeEV cast" << shapeEV_cast <<
    //    endl;
    //
    //    cout << "neutral face shape" << neutral_xyz.cast<T>().rows() << " ,"
    //    << neutral_xyz.cast<T>().cols() << endl; cout << " neutral face " <<
    //    neutral_xyz.cast<T>() << endl;

    // decompose the calculation
    //    Matrix<T, 199, 1> temp = shapeEV.cwiseProduct(alpha_v);
    //      cout << "EV * alpha" << temp << endl;
    //      Matrix<T, 3, 1> temp_1 = pc_xyz.cast<T>() * temp;
    //      cout << "deformation " << temp_1 << endl;
    // cout << "EV * alpha element wise " << shapeEV.cwiseProduct(alpha_v) <<
    // endl; cout << "PC " << pc_xyz.cast<T>() << endl; cout << "displacement
    // after PC " << (pc_xyz.cast<T>() * (shapeEV.cwiseProduct(alpha_v)))(0,0)
    // << endl;

    Matrix<T, 3, 1> outputs_xyz =
        neutral_xyz.cast<T>() +
        pc_xyz.cast<T>() * (shapeEV.cwiseProduct(alpha_v));
    Matrix<T, 3, 1> outputs_rgb =
        neutral_rgb.cast<T>() + pc_rgb.cast<T>() * (texEV.cwiseProduct(beta_v));

    // cout << " outputs xyz " << outputs_xyz << " " << outputs_xyz.rows() << "
    // ," << outputs_xyz.cols() << endl;

    Matrix<T, 4, 1> outputs_homo(T(outputs_xyz.x()), T(outputs_xyz.y()),
                                 T(outputs_xyz.z()), T(1));
    // std::cout << "outputs homo shape !!" << outputs_homo.rows() << ", " <<
    // outputs_homo.cols() << endl; std::cout << "P shape!! " <<
    // P.cast<T>().rows() << " ," << P.cast<T>().cols() << endl;
    Matrix<T, 3, 1> outputs_cam = P.cast<T>() * outputs_homo;

    //    cout << "outputs_cam" << outputs_cam(0,0) << ", " <<  outputs_cam(1,0)
    //    << ", " << outputs_cam(2,0) << endl;

    // std::cout << "finished pixel position calculation!" << std::endl;
    T pixel_x = outputs_cam(0, 0) / outputs_cam(2, 0);
    T pixel_y = outputs_cam(1, 0) / outputs_cam(2, 0);

    //    cout << "r_map: " << endl << r_map << endl;
    //    for (int r = 0; r < r_map.rows(); r++) {
    //      for (int c = 0; c < r_map.cols(); c++) {
    //        cout << "r_map: " << r << ", " << c << ": " << r_map(r, c) <<
    //        endl;
    //      }
    //    }
    //    cout << "pixel_x: " << pixel_x << endl;
    //    cout << "pixel_y: " << pixel_y << endl;

    if (pixel_x < T(r_map.rows()) && pixel_x > T(0) &&
        pixel_y < T(r_map.cols()) && pixel_y > T(0)) {
      //    cout << "image pixel " <<  pixel_x << ", " << pixel_y << endl;

      T outputs_depth = outputs_cam(2, 0);
      // 346,560~746,960
      // create grid and interpolate target value
      //////////////////////////////////////////////////////////////////
      // create table look up for depth map
      ceres::Grid2D<double, 1> depth_grid(&depth_map(0, 0), 0, depth_map.rows(),
                                          0, depth_map.cols());
      ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> depth_interpolator(
          depth_grid);
      // get target_depth from table, please douoble check order of y and x
      T target_depth;
      depth_interpolator.Evaluate(pixel_y, pixel_x, &target_depth);
      // std::cout << "finished grip interpolation for depth1" << std::endl;
      /////////////////////////////////
      // create table look up for r map
      ceres::Grid2D<double, 1> r_grid(&r_map(0, 0), 0, r_map.rows(), 0,
                                      r_map.cols());
      ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> r_interpolator(
          r_grid);

      // get target r from table, please double check order of y and x
      T target_r;
      r_interpolator.Evaluate(pixel_y, pixel_x, &target_r);
      T target_test;
      r_interpolator.Evaluate(T(673), T(636), &target_test);
      // cout << "target_r_test: " << target_test << endl;
      /////////////////////////////////
      // create table look up for g map
      ceres::Grid2D<double, 1> g_grid(&g_map(0, 0), 0, g_map.rows(), 0,
                                      g_map.cols());
      ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> g_interpolator(
          g_grid);
      // get target g from table, please double check order of y and x
      T target_g;
      g_interpolator.Evaluate(pixel_y, pixel_x, &target_g);

      /////////////////////////////////
      // create table look up for b map
      ceres::Grid2D<double, 1> b_grid(&b_map(0, 0), 0, b_map.rows(), 0,
                                      b_map.cols());
      ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> b_interpolator(
          b_grid);
      // get target b from table, please double check order of y and x
      T target_b;
      b_interpolator.Evaluate(pixel_y, pixel_x, &target_b);
      //////////////////////////////////////////////////////////////////

      Matrix<T, 3, 1> target_rgb(target_r, target_g, target_b);

      Matrix<T, 3, 1> res_rgb = (target_rgb - outputs_rgb) / T(255.0);

      // cout << "model color: " << endl << outputs_rgb << endl;
      // cout << "target rgb: " << endl << target_rgb << endl;
      if (target_depth < T(1e-30)) {
        residual[0] = T(0);
        residual[1] = T(0);
      } else {
        // cout << "depth not zero!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;

        T res_depth = (target_depth - outputs_depth) * T(1.0);
        //        cout << "res depth: " << res_depth << ", res_rgb" << endl
        //             << res_rgb << endl;
        // std::cout << "finished residual calculation." << std::endl;
        residual[0] = T(res_rgb.norm());
        residual[1] = T(res_depth);
        // cout << "target_rgb" << target_rgb << endl;
        // cout << "outputs_rgb" << outputs_rgb << endl;
        // cout << P << endl;
      }
    } else {
      residual[0] = T(0);
      residual[1] = T(0);
    }

    return true;
  }

 private:
  const MatrixXd neutral_xyz;
  const MatrixXd neutral_rgb;
  const MatrixXd pc_xyz;
  const MatrixXd pc_rgb;
  const MatrixXd shapeEV;
  const MatrixXd texEV;
  const MatrixXd& depth_map;
  const MatrixXd& r_map;
  const MatrixXd& g_map;
  const MatrixXd& b_map;
  const MatrixXd P;
};
struct RGBDCostFunction_v2 {
  RGBDCostFunction_v2(const MatrixXd& neutral_xyz_,
                      const MatrixXd& neutral_rgb_, const MatrixXd& pc_xyz_,
                      const MatrixXd& pc_rgb_, const MatrixXd& shapeEV_,
                      const MatrixXd& texEV_, const MatrixXd& depth_map_,
                      const MatrixXd& r_map_, const MatrixXd& g_map_,
                      const MatrixXd& b_map_, const MatrixXd& P_)
      : neutral_xyz(neutral_xyz_),
        neutral_rgb(neutral_rgb_),
        pc_xyz(pc_xyz_),
        pc_rgb(pc_rgb_),
        shapeEV(shapeEV_),
        texEV(texEV_),
        depth_map(depth_map_),
        r_map(r_map_),
        g_map(g_map_),
        b_map(b_map_),
        P(P_) {}

  template <typename T>
  bool operator()(const T* const alpha, const T* const beta,
                  const T* const sT_bs2gt, T* residual) const {
    //    std::cout << "enter operator!" << std::endl;
    Eigen::Map<Matrix<T, PRIN_COMP_NUM, 1> const> const alpha_v(alpha);
    Eigen::Map<Matrix<T, PRIN_COMP_NUM, 1> const> const beta_v(beta);
    T pose_c[6] = {sT_bs2gt[0], sT_bs2gt[1], sT_bs2gt[2],
                   sT_bs2gt[3], sT_bs2gt[4], sT_bs2gt[5]};
    PoseIncrement<T> pose_increment(pose_c);

    //    Matrix<T, PRIN_COMP_NUM, 1> shapeEV_cast = (shapeEV).cast<T>();
    //    Matrix<T, PRIN_COMP_NUM, 1> texEV_cast =
    //    texEV.block(0,0,PRIN_COMP_NUM,1);

    //    std::cout << "finished mapping array to matrix" << std::endl;

    // cout << "xyz operator in " << neutral_xyz(0,0) << ", " <<
    // neutral_xyz(1,0) << ", " << neutral_xyz(2,0) << endl;
    // cout << "rgb operator in " << neutral_rgb(0,0) << ", " <<
    // neutral_rgb(1,0) << ", " << neutral_rgb(2,0) << endl;
    //
    //    cout << "alpha shape " << alpha_v.rows() << " ," << alpha_v.cols() <<
    //    endl; cout << "shapeEV shape " << shapeEV_cast.rows() << " ," <<
    //    shapeEV_cast.cols() << endl; cout << "shapeEV cast" << shapeEV_cast <<
    //    endl;
    //
    //    cout << "neutral face shape" << neutral_xyz.cast<T>().rows() << " ,"
    //    << neutral_xyz.cast<T>().cols() << endl; cout << " neutral face " <<
    //    neutral_xyz.cast<T>() << endl;

    // decompose the calculation
    //    Matrix<T, 199, 1> temp = shapeEV.cwiseProduct(alpha_v);
    //      cout << "EV * alpha" << temp << endl;
    //      Matrix<T, 3, 1> temp_1 = pc_xyz.cast<T>() * temp;
    //      cout << "deformation " << temp_1 << endl;
    // cout << "EV * alpha element wise " << shapeEV.cwiseProduct(alpha_v) <<
    // endl; cout << "PC " << pc_xyz.cast<T>() << endl; cout << "displacement
    // after PC " << (pc_xyz.cast<T>() * (shapeEV.cwiseProduct(alpha_v)))(0,0)
    // << endl;

    Matrix<T, 3, 1> outputs_xyz_tmp =
        neutral_xyz.cast<T>() +
        pc_xyz.cast<T>() * (shapeEV.cwiseProduct(alpha_v));
    Matrix<T, 3, 1> outputs_rgb =
        neutral_rgb.cast<T>() + pc_rgb.cast<T>() * (texEV.cwiseProduct(beta_v));
    Matrix<T, 3, 1> outputs_xyz;
    T inputs_xyz_ptr[3] = {outputs_xyz_tmp(0, 0), outputs_xyz_tmp(1, 0),
                           outputs_xyz_tmp(2, 0)};
    T outputs_xyz_ptr[3];
    pose_increment.apply(inputs_xyz_ptr, outputs_xyz_ptr);
    outputs_xyz = Matrix<T, 3, 1>(outputs_xyz_ptr[0], outputs_xyz_ptr[1],
                                  outputs_xyz_ptr[2]);
    // cout << "inputs_xyz: " << endl << outputs_xyz_tmp << endl;
    // cout << "outputs_xyz: " << endl << outputs_xyz << endl;
    // cout << " outputs xyz " << outputs_xyz << " " << outputs_xyz.rows() << "
    // ," << outputs_xyz.cols() << endl;

    Matrix<T, 4, 1> outputs_homo(T(outputs_xyz.x()), T(outputs_xyz.y()),
                                 T(outputs_xyz.z()), T(1));
    // std::cout << "outputs homo shape !!" << outputs_homo.rows() << ", " <<
    // outputs_homo.cols() << endl; std::cout << "P shape!! " <<
    // P.cast<T>().rows() << " ," << P.cast<T>().cols() << endl;
    Matrix<T, 3, 1> outputs_cam = P.cast<T>() * outputs_homo;

    //    cout << "outputs_cam" << outputs_cam(0,0) << ", " <<  outputs_cam(1,0)
    //    << ", " << outputs_cam(2,0) << endl;

    // std::cout << "finished pixel position calculation!" << std::endl;
    T pixel_x = outputs_cam(0, 0) / outputs_cam(2, 0);
    T pixel_y = outputs_cam(1, 0) / outputs_cam(2, 0);

    if (pixel_x < T(r_map.rows()) && pixel_x > T(0) &&
        pixel_y < T(r_map.cols()) && pixel_y > T(0)) {
      //    cout << "image pixel " <<  pixel_x << ", " << pixel_y << endl;

      T outputs_depth = outputs_cam(2, 0);

      // create grid and interpolate target value
      //////////////////////////////////////////////////////////////////
      // create table look up for depth map
      ceres::Grid2D<double, 1> depth_grid(&depth_map(0, 0), 0, depth_map.rows(),
                                          0, depth_map.cols());
      ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> depth_interpolator(
          depth_grid);
      // get target_depth from table, please douoble check order of y and x
      T target_depth;
      depth_interpolator.Evaluate(pixel_y, pixel_x, &target_depth);
      // std::cout << "finished grip interpolation for depth1" << std::endl;
      /////////////////////////////////
      // create table look up for r map
      ceres::Grid2D<double, 1> r_grid(&r_map(0, 0), 0, r_map.rows(), 0,
                                      r_map.cols());
      ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> r_interpolator(
          r_grid);
      // get target r from table, please double check order of y and x
      T target_r;
      r_interpolator.Evaluate(pixel_y, pixel_x, &target_r);

      /////////////////////////////////
      // create table look up for g map
      ceres::Grid2D<double, 1> g_grid(&g_map(0, 0), 0, g_map.rows(), 0,
                                      g_map.cols());
      ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> g_interpolator(
          g_grid);
      // get target g from table, please double check order of y and x
      T target_g;
      g_interpolator.Evaluate(pixel_y, pixel_x, &target_g);

      /////////////////////////////////
      // create table look up for b map
      ceres::Grid2D<double, 1> b_grid(&b_map(0, 0), 0, b_map.rows(), 0,
                                      b_map.cols());
      ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> b_interpolator(
          b_grid);
      // get target b from table, please double check order of y and x
      T target_b;
      b_interpolator.Evaluate(pixel_y, pixel_x, &target_b);
      //////////////////////////////////////////////////////////////////

      Matrix<T, 3, 1> target_rgb(target_r, target_g, target_b);

      Matrix<T, 3, 1> res_rgb = target_rgb - outputs_rgb;

      // cout << "model color: " << endl << outputs_rgb << endl;
      // cout << "target rgb: " << endl << target_rgb << endl;
      if (target_depth < T(1e-30)) {
        residual[0] = T(0);
        residual[1] = T(0);
      } else {
        // cout << "depth not zero!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
        T res_depth = target_depth - outputs_depth;
        // cout << "res_depth: " << endl << res_depth << endl;
        // std::cout << "finished residual calculation." << std::endl;
        residual[0] = T(res_rgb.norm());
        residual[1] = T(res_depth);
        // cout << "target_rgb" << target_rgb << endl;
        // cout << "outputs_rgb" << outputs_rgb << endl;
        // cout << P << endl;
      }
    } else {
      residual[0] = T(0);
      residual[1] = T(0);
    }

    return true;
  }

 private:
  const MatrixXd neutral_xyz;
  const MatrixXd neutral_rgb;
  const MatrixXd pc_xyz;
  const MatrixXd pc_rgb;
  const MatrixXd shapeEV;
  const MatrixXd texEV;
  const MatrixXd& depth_map;
  const MatrixXd& r_map;
  const MatrixXd& g_map;
  const MatrixXd& b_map;
  const MatrixXd P;
};

struct RGBDCostFunction_v3 {
  RGBDCostFunction_v3(const MatrixXd& neutral_xyz_,
                      const MatrixXd& neutral_rgb_, const MatrixXd& pc_xyz_,
                      const MatrixXd& pc_rgb_, const MatrixXd& shapeEV_,
                      const MatrixXd& texEV_, const MatrixXd& depth_map_,
                      const cv::Mat& rgb_map_, const MatrixXd& P_)
      : neutral_xyz(neutral_xyz_),
        neutral_rgb(neutral_rgb_),
        pc_xyz(pc_xyz_),
        pc_rgb(pc_rgb_),
        shapeEV(shapeEV_),
        texEV(texEV_),
        depth_map(depth_map_),
        rgb_map(rgb_map_),
        P(P_) {}

  template <typename T>
  bool operator()(const T* const alpha, const T* const beta,
                  T* residual) const {
    //    std::cout << "enter operator!" << std::endl;
    Eigen::Map<Matrix<T, PRIN_COMP_NUM, 1> const> const alpha_v(alpha);
    Eigen::Map<Matrix<T, PRIN_COMP_NUM, 1> const> const beta_v(beta);

    //    Matrix<T, PRIN_COMP_NUM, 1> shapeEV_cast = (shapeEV).cast<T>();
    //    Matrix<T, PRIN_COMP_NUM, 1> texEV_cast =
    //    texEV.block(0,0,PRIN_COMP_NUM,1);

    //    std::cout << "finished mapping array to matrix" << std::endl;

    // cout << "xyz operator in " << neutral_xyz(0,0) << ", " <<
    // neutral_xyz(1,0) << ", " << neutral_xyz(2,0) << endl;
    // cout << "rgb operator in " << neutral_rgb(0,0) << ", " <<
    // neutral_rgb(1,0) << ", " << neutral_rgb(2,0) << endl;
    //
    //    cout << "alpha shape " << alpha_v.rows() << " ," << alpha_v.cols() <<
    //    endl; cout << "shapeEV shape " << shapeEV_cast.rows() << " ," <<
    //    shapeEV_cast.cols() << endl; cout << "shapeEV cast" << shapeEV_cast <<
    //    endl;
    //
    //    cout << "neutral face shape" << neutral_xyz.cast<T>().rows() << " ,"
    //    << neutral_xyz.cast<T>().cols() << endl; cout << " neutral face " <<
    //    neutral_xyz.cast<T>() << endl;

    // decompose the calculation
    //    Matrix<T, 199, 1> temp = shapeEV.cwiseProduct(alpha_v);
    //      cout << "EV * alpha" << temp << endl;
    //      Matrix<T, 3, 1> temp_1 = pc_xyz.cast<T>() * temp;
    //      cout << "deformation " << temp_1 << endl;
    // cout << "EV * alpha element wise " << shapeEV.cwiseProduct(alpha_v) <<
    // endl; cout << "PC " << pc_xyz.cast<T>() << endl; cout << "displacement
    // after PC " << (pc_xyz.cast<T>() * (shapeEV.cwiseProduct(alpha_v)))(0,0)
    // << endl;

    Matrix<T, 3, 1> outputs_xyz =
        neutral_xyz.cast<T>() +
        pc_xyz.cast<T>() * (shapeEV.cwiseProduct(alpha_v));
    Matrix<T, 3, 1> outputs_rgb =
        neutral_rgb.cast<T>() + pc_rgb.cast<T>() * (texEV.cwiseProduct(beta_v));

    // cout << " outputs xyz " << outputs_xyz << " " << outputs_xyz.rows() << "
    // ," << outputs_xyz.cols() << endl;

    Matrix<T, 4, 1> outputs_homo(T(outputs_xyz.x()), T(outputs_xyz.y()),
                                 T(outputs_xyz.z()), T(1));
    // std::cout << "outputs homo shape !!" << outputs_homo.rows() << ", " <<
    // outputs_homo.cols() << endl; std::cout << "P shape!! " <<
    // P.cast<T>().rows() << " ," << P.cast<T>().cols() << endl;
    Matrix<T, 3, 1> outputs_cam = P.cast<T>() * outputs_homo;

    //    cout << "outputs_cam" << outputs_cam(0,0) << ", " <<  outputs_cam(1,0)
    //    << ", " << outputs_cam(2,0) << endl;

    // std::cout << "finished pixel position calculation!" << std::endl;
    T pixel_x = outputs_cam(0, 0) / outputs_cam(2, 0);
    T pixel_y = outputs_cam(1, 0) / outputs_cam(2, 0);

    //    cout << "r_map: " << endl << r_map << endl;
    //    for (int r = 0; r < r_map.rows(); r++) {
    //      for (int c = 0; c < r_map.cols(); c++) {
    //        cout << "r_map: " << r << ", " << c << ": " << r_map(r, c) <<
    //        endl;
    //      }
    //    }
    //    cout << "pixel_x: " << pixel_x << endl;
    //    cout << "pixel_y: " << pixel_y << endl;

    if (pixel_x < T(rgb_map.rows) && pixel_x > T(0) &&
        pixel_y < T(rgb_map.cols) && pixel_y > T(0)) {
      //    cout << "image pixel " <<  pixel_x << ", " << pixel_y << endl;

      T outputs_depth = outputs_cam(2, 0);
      // 346,560~746,960
      // create grid and interpolate target value
      //////////////////////////////////////////////////////////////////
      // create table look up for depth map
      ceres::Grid2D<double, 1> depth_grid(&depth_map(0, 0), 0, depth_map.rows(),
                                          0, depth_map.cols());
      ceres::BiCubicInterpolator<ceres::Grid2D<double, 1>> depth_interpolator(
          depth_grid);
      // get target_depth from table, please douoble check order of y and x
      T target_depth;
      depth_interpolator.Evaluate(pixel_y, pixel_x, &target_depth);
      // std::cout << "finished grip interpolation for depth1" << std::endl;
      /////////////////////////////////
      // create table look up for r map
      ceres::Grid2D<uchar, 3> rgb_grid(rgb_map.ptr(), 0, rgb_map.rows, 0,
                                       rgb_map.cols);
      ceres::BiCubicInterpolator<ceres::Grid2D<uchar, 3>> rgb_interpolator(
          rgb_grid);

      // get target r from table, please double check order of y and x
      T target_rgb[3];
      rgb_interpolator.Evaluate(pixel_y, pixel_x, &target_rgb[0]);
      //      T target_test;
      //      r_interpolator.Evaluate(T(673), T(636), &target_test);
      // cout << "target_r_test: " << target_test << endl;

      Matrix<T, 3, 1> res_rgb;
      res_rgb(0, 0) = (target_rgb[0] - outputs_rgb(0, 0)) / T(255.0);
      res_rgb(1, 0) = (target_rgb[1] - outputs_rgb(1, 0)) / T(255.0);
      res_rgb(2, 0) = (target_rgb[2] - outputs_rgb(2, 0)) / T(255.0);

      // cout << "model color: " << endl << outputs_rgb << endl;
      // cout << "target rgb: " << endl << target_rgb << endl;
      if (target_depth < T(1e-30)) {
        residual[0] = T(0);
        residual[1] = T(0);
      } else {
        // cout << "depth not zero!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;

        T res_depth = (target_depth - outputs_depth) * T(1.0);
        //        cout << "res depth: " << res_depth << ", res_rgb" << endl
        //             << res_rgb << endl;
        // std::cout << "finished residual calculation." << std::endl;
        residual[0] = T(res_rgb.norm());
        residual[1] = T(res_depth);
        // cout << "target_rgb" << target_rgb << endl;
        // cout << "outputs_rgb" << outputs_rgb << endl;
        // cout << P << endl;
      }
    } else {
      residual[0] = T(0);
      residual[1] = T(0);
    }

    return true;
  }

 private:
  const MatrixXd neutral_xyz;
  const MatrixXd neutral_rgb;
  const MatrixXd pc_xyz;
  const MatrixXd pc_rgb;
  const MatrixXd shapeEV;
  const MatrixXd texEV;
  const MatrixXd& depth_map;
  const cv::Mat& rgb_map;
  const MatrixXd P;
};

bool generateFace(Eigen::MatrixXd shapeMU, Eigen::MatrixXd shapePC,
                  Eigen::MatrixXd shapeEV, Eigen::MatrixXd texMU,
                  Eigen::MatrixXd texPC, Eigen::MatrixXd texEV,
                  Eigen::MatrixXd P, double* alpha, double* beta,
                  Eigen::MatrixXd& tl) {
  //    cout << "mmp " << endl;

  Vertex* vertices = new Vertex[53490];
  // Triangle *triangles = new Triangle[106466];
  SimpleMesh finale;

  for (int i = 0; i < 53490; i++) {
    double neutral_x = shapeMU(i * 3);
    double neutral_y = shapeMU(i * 3 + 1);
    double neutral_z = shapeMU(i * 3 + 2);

    double neutral_r = texMU(i * 3);
    double neutral_g = texMU(i * 3 + 1);
    double neutral_b = texMU(i * 3 + 2);

    Vector3d neutral_xyz(neutral_x, neutral_y, neutral_z);
    Vector3d neutral_rgb(neutral_r, neutral_g, neutral_b);

    VectorXd pc_x = shapePC.block(i * 3, 0, 1, PRIN_COMP_NUM);
    VectorXd pc_y = shapePC.block(i * 3 + 1, 0, 1, PRIN_COMP_NUM);
    VectorXd pc_z = shapePC.block(i * 3 + 2, 0, 1, PRIN_COMP_NUM);

    VectorXd pc_r = texPC.block(i * 3, 0, 1, PRIN_COMP_NUM);
    VectorXd pc_g = texPC.block(i * 3 + 1, 0, 1, PRIN_COMP_NUM);
    VectorXd pc_b = texPC.block(i * 3 + 2, 0, 1, PRIN_COMP_NUM);

    MatrixXd pc_xyz = MatrixXd::Zero(3, PRIN_COMP_NUM);
    MatrixXd pc_rgb = MatrixXd::Zero(3, PRIN_COMP_NUM);

    pc_xyz.block(0, 0, 1, PRIN_COMP_NUM) = pc_x;
    pc_xyz.block(1, 0, 1, PRIN_COMP_NUM) = pc_y;
    pc_xyz.block(2, 0, 1, PRIN_COMP_NUM) = pc_z;

    pc_rgb.block(0, 0, 1, PRIN_COMP_NUM) = pc_r;
    pc_rgb.block(1, 0, 1, PRIN_COMP_NUM) = pc_g;
    pc_rgb.block(2, 0, 1, PRIN_COMP_NUM) = pc_b;

    Eigen::Map<Matrix<double, PRIN_COMP_NUM, 1> const> const alpha_v(alpha);
    Eigen::Map<Matrix<double, PRIN_COMP_NUM, 1> const> const beta_v(beta);

    Vector3d outputs_xyz = neutral_xyz + pc_xyz * shapeEV.cwiseProduct(alpha_v);
    Vector3d outputs_rgb =
        neutral_rgb;  //+ pc_rgb * texEV.cwiseProduct(beta_v);

    //        Vector4d outputs_homo(outputs_xyz.x(), outputs_xyz.y(),
    //        outputs_xyz.z(), double(1)); outputs_xyz = P * outputs_homo;

    // cout << "camera space " << outputs_xyz.x() << ", " << outputs_xyz.y() <<
    // ", " << outputs_xyz.z() << endl;

    vertices[i].position =
        Vector4f(outputs_xyz.x(), outputs_xyz.y(), outputs_xyz.z(), float(1));
    vertices[i].color = Vector4uc(
        (unsigned char)(outputs_rgb.x()), (unsigned char)(outputs_rgb.y()),
        (unsigned char)(outputs_rgb.z()), (unsigned char)(1));

    finale.addVertex(vertices[i]);
  }
  if (!finale.writeMesh(PROJECT_DIR + std::string("/finale.off"))) {
    std::cout << "face generating failed!!!!" << std::endl;
  }
  return true;
}
bool generateFace_v2(Eigen::MatrixXd shapeMU, Eigen::MatrixXd shapePC,
                     Eigen::MatrixXd shapeEV, Eigen::MatrixXd texMU,
                     Eigen::MatrixXd texPC, Eigen::MatrixXd texEV,
                     Eigen::MatrixXd P, double* alpha, double* beta,
                     Eigen::MatrixXd& tl, double* sT_bs2gt) {
  //    cout << "mmp " << endl;

  Vertex* vertices = new Vertex[53490];
  // Triangle *triangles = new Triangle[106466];
  SimpleMesh finale;

  for (int i = 0; i < 53490; i++) {
    double neutral_x = shapeMU(i * 3);
    double neutral_y = shapeMU(i * 3 + 1);
    double neutral_z = shapeMU(i * 3 + 2);

    double neutral_r = texMU(i * 3);
    double neutral_g = texMU(i * 3 + 1);
    double neutral_b = texMU(i * 3 + 2);

    Vector3d neutral_xyz(neutral_x, neutral_y, neutral_z);
    Vector3d neutral_rgb(neutral_r, neutral_g, neutral_b);

    VectorXd pc_x = shapePC.block(i * 3, 0, 1, PRIN_COMP_NUM);
    VectorXd pc_y = shapePC.block(i * 3 + 1, 0, 1, PRIN_COMP_NUM);
    VectorXd pc_z = shapePC.block(i * 3 + 2, 0, 1, PRIN_COMP_NUM);

    VectorXd pc_b = texPC.block(i * 3, 0, 1, PRIN_COMP_NUM);
    VectorXd pc_g = texPC.block(i * 3 + 1, 0, 1, PRIN_COMP_NUM);
    VectorXd pc_r = texPC.block(i * 3 + 2, 0, 1, PRIN_COMP_NUM);

    MatrixXd pc_xyz = MatrixXd::Zero(3, PRIN_COMP_NUM);
    MatrixXd pc_rgb = MatrixXd::Zero(3, PRIN_COMP_NUM);

    pc_xyz.block(0, 0, 1, PRIN_COMP_NUM) = pc_x;
    pc_xyz.block(1, 0, 1, PRIN_COMP_NUM) = pc_y;
    pc_xyz.block(2, 0, 1, PRIN_COMP_NUM) = pc_z;

    pc_rgb.block(0, 0, 1, PRIN_COMP_NUM) = pc_r;
    pc_rgb.block(1, 0, 1, PRIN_COMP_NUM) = pc_g;
    pc_rgb.block(2, 0, 1, PRIN_COMP_NUM) = pc_b;

    Eigen::Map<Matrix<double, PRIN_COMP_NUM, 1> const> const alpha_v(alpha);
    Eigen::Map<Matrix<double, PRIN_COMP_NUM, 1> const> const beta_v(beta);

    Vector3d outputs_xyz_tmp =
        neutral_xyz + pc_xyz * shapeEV.cwiseProduct(alpha_v);
    Vector3d outputs_rgb = neutral_rgb + pc_rgb * texEV.cwiseProduct(beta_v);
    PoseIncrement<double> pose_increment(sT_bs2gt);
    Matrix4d T_bs2gt =
        PoseIncrement<double>::convertToMatrix(pose_increment).cast<double>();
    Vector3d outputs_xyz =
        T_bs2gt.block(0, 0, 3, 3) * outputs_xyz_tmp + T_bs2gt.block(0, 3, 3, 1);

    //        Vector4d outputs_homo(outputs_xyz.x(), outputs_xyz.y(),
    //        outputs_xyz.z(), double(1)); outputs_xyz = P * outputs_homo;

    // cout << "camera space " << outputs_xyz.x() << ", " << outputs_xyz.y() <<
    // ", " << outputs_xyz.z() << endl;

    vertices[i].position =
        Vector4f(outputs_xyz.x(), outputs_xyz.y(), outputs_xyz.z(), float(1));
    vertices[i].color = Vector4uc(
        (unsigned char)(outputs_rgb.x()), (unsigned char)(outputs_rgb.y()),
        (unsigned char)(outputs_rgb.z()), (unsigned char)(1));

    finale.addVertex(vertices[i]);
  }
  if (!finale.writeMesh(PROJECT_DIR + std::string("/finale.off"))) {
    std::cout << "face generating failed!!!!" << std::endl;
  }
  return true;
}
//        double neutral_x = shapeMU(i * 3);
//        double neutral_y = shapeMU(i * 3 + 1);
//        double neutral_z = shapeMU(i * 3 + 2);
//
//        double neutral_r = texMU(i * 3);
//        double neutral_g = texMU(i * 3 + 1);
//        double neutral_b = texMU(i * 3 + 2);
//
//        cout << "neutral xyz" << neutral_x << ", " << neutral_y << ", "  <<
//        neutral_z << endl;
//
//        Vector3d neutral_xyz(neutral_x, neutral_y, neutral_z);
//        Vector3d neutral_rgb(neutral_r, neutral_g, neutral_b);
//
//        VectorXd pc_x = shapePC.block(i * 3, 0, 1, PRIN_COMP_NUM);
//        VectorXd pc_y = shapePC.block(i * 3 + 1, 0, 1, PRIN_COMP_NUM);
//        VectorXd pc_z = shapePC.block(i * 3 + 2, 0, 1, PRIN_COMP_NUM);
//
//        VectorXd pc_r = texPC.block(i * 3, 0, 1, PRIN_COMP_NUM);
//        VectorXd pc_g = texPC.block(i * 3 + 1, 0, 1, PRIN_COMP_NUM);
//        VectorXd pc_b = texPC.block(i * 3 + 2, 0, 1, PRIN_COMP_NUM);
//
//        MatrixXd pc_xyz = MatrixXd::Zero(3, PRIN_COMP_NUM);
//        MatrixXd pc_rgb = MatrixXd::Zero(3, PRIN_COMP_NUM);
//
//        pc_xyz.block(0, 0, 1, PRIN_COMP_NUM) = pc_x;
//        pc_xyz.block(1, 0, 1, PRIN_COMP_NUM) = pc_y;
//        pc_xyz.block(2, 0, 1, PRIN_COMP_NUM) = pc_z;
//
//        pc_rgb.block(0, 0, 1, PRIN_COMP_NUM) = pc_r;
//        pc_rgb.block(1, 0, 1, PRIN_COMP_NUM) = pc_g;
//        pc_rgb.block(2, 0, 1, PRIN_COMP_NUM) = pc_b;
//
//        Eigen::Map<Matrix<double, PRIN_COMP_NUM, 1> const> const
//        alpha_v(alpha); Eigen::Map<Matrix<double, PRIN_COMP_NUM, 1> const>
//        const beta_v(beta);
//
//        Vector3d outputs_xyz = neutral_xyz + pc_xyz *
//        shapeEV.cwiseProduct(alpha_v);
//
//        cout << "outputs xyz: " << outputs_xyz.x() << ", " << outputs_xyz.y()
//        << ", " << outputs_xyz.z() << endl;
//
//        Vector3d outputs_rgb = neutral_rgb + pc_xyz *
//        shapeEV.cwiseProduct(beta_v);
//
//        vertices[i].position =
//                Vector4f(outputs_xyz.x(), outputs_xyz.y(), outputs_xyz.z(),
//                float(1));
//        vertices[i].color = Vector4uc(
//                (unsigned char)(outputs_rgb.x()), (unsigned
//                char)(outputs_rgb.y()), (unsigned char)(outputs_rgb.z()),
//                (unsigned char)(1));
//        finale.addVertex(vertices[i]);
//    }
//    for (int i = 0; i < tl.rows(); i++) {
//        triangles[i].idx0 = int(tl(i, 1));
//        triangles[i].idx1 = int(tl(i, 0));
//        triangles[i].idx2 = int(tl(i, 2));
//        finale.addFace(int(tl(i, 1)), int(tl(i, 0)), int(tl(i, 2)));
//    }

//    if (!finale.writeMesh(PROJECT_DIR + std::string("/finale.off"))) {
//        std::cout << "face generating failed!!!!" << std::endl;
//    }
//  return true;
//}

int modelFitting(Eigen::Matrix4d& T_pro, Eigen::Matrix4d& T_icp,
                 double scale_bs2gt) {
  cout << PROJECT_DIR << endl;
  // Generate Ground Truth Maps
  // r_map, g_map, b_map, depth_map
  Eigen::Matrix<double, 3, 4> P;
  P << 1052.667867276341, 0, 962.4130834944134, 0, 0, 1052.020917785721,
      536.2206151001486, 0, 0, 0, 1, 0;
  SimpleMesh our_mesh;
  our_mesh.loadMesh(PROJECT_DIR + std::string("/data/entire_face.off"));
  //  cout << "load successful!!!!!! " << endl;
  PointCloud targets{our_mesh};
  //  cout << "point cloud sucessfull!!!! " << endl;
  auto target_points_xyz = targets.getPoints();
  auto target_points_rgba = targets.getRGBA();
  //  cout << "get points successfull!!!!!" << endl;
  int target_num_points = target_points_xyz.size();

  MatrixXd depth_map = MatrixXd::Zero(1080, 1920);
  cv::Mat depth_map_cv(1080, 1920, CV_32F);
  MatrixXd r_map = MatrixXd::Zero(1080, 1920);
  MatrixXd g_map = MatrixXd::Zero(1080, 1920);
  MatrixXd b_map = MatrixXd::Zero(1080, 1920);
  //  cv::Mat rgb_map(1080, 1920, CV_8UC3, cv::Scalar(0, 0, 0));
  //  cv::Mat depth_cvmap(1080, 1920, CV_8UC1, cv::Scalar(0));
  //  cv::Mat rgb_map_fixed(1080, 1920, CV_8UC3, cv::Scalar(0, 0, 0));
  //  cv::Mat depth_cvmap_fixed(1080, 1920, CV_8UC1, cv::Scalar(0));

  for (int i = 0; i < target_num_points; i++) {
    Vector3d target_xyz(double(target_points_xyz[i].x()),
                        double(target_points_xyz[i].y()),
                        double(target_points_xyz[i].z()));
    //    cout << "xyz got!!" << endl;

    Vector3d target_rgba(double(target_points_rgba[i].x()),
                         double(target_points_rgba[i].y()),
                         double(target_points_rgba[i].z()));

    //    cout << "rgb got!! " << endl;

    Vector4d target_homog(target_xyz[0], target_xyz[1], target_xyz[2],
                          double(1));
    Vector3d target_outputs = P * target_homog;
    if (target_outputs[0] == 0) {
      continue;
    }
    int pixel_x = int(target_outputs[0] / target_outputs[2]);
    int pixel_y = int(target_outputs[1] / target_outputs[2]);

    double target_depth = target_outputs[2];

    //    cout << "depth got" << endl;
    r_map(pixel_x, pixel_y) = double(target_rgba[2]);
    g_map(pixel_x, pixel_y) = double(target_rgba[1]);
    b_map(pixel_x, pixel_y) = double(target_rgba[0]);
    depth_map(pixel_x, pixel_y) = target_depth;
    depth_map_cv.at<float>(pixel_x, pixel_y) = target_depth;
    //    rgb_map.at<Vec3b>(pixel_y, pixel_x)[0] = (unsigned
    //    char)target_rgba[2]; rgb_map.at<Vec3b>(pixel_y, pixel_x)[1] =
    //    (unsigned char)target_rgba[1]; rgb_map.at<Vec3b>(pixel_y, pixel_x)[2]
    //    = (unsigned char)target_rgba[0]; depth_cvmap.at<uchar>(pixel_y,
    //    pixel_x) = (unsigned char)target_depth; cout << i << endl;
  }
  //  cout << "before bilateral filtering..." << endl;
  //  cv::Mat depth_map_filter_cv;
  //  cv::bilateralFilter(depth_map_cv, depth_map_filter_cv, 3, 20, 20);

  //  cout << "bilateral filtering succeed!" << endl;

  for (int r = 1; r < 1079; r++) {
    for (int c = 1; c < 1919; c++) {
      if (r_map(r, c) == 0) {
        r_map(r, c) = (unsigned char)((r_map(r - 1, c) + r_map(r + 1, c) +
                                       r_map(r, c - 1) + r_map(r, c + 1)) /
                                      4);
        g_map(r, c) = (unsigned char)((g_map(r - 1, c) + g_map(r + 1, c) +
                                       g_map(r, c - 1) + g_map(r, c + 1)) /
                                      4);
        b_map(r, c) = (unsigned char)((b_map(r - 1, c) + b_map(r + 1, c) +
                                       b_map(r, c - 1) + b_map(r, c + 1)) /
                                      4);
        depth_map(r, c) = ((depth_map(r - 1, c) + depth_map(r + 1, c) +
                            depth_map(r, c - 1) + depth_map(r, c + 1)) /
                           4);
      }
      //      rgb_map_fixed.at<Vec3b>(r, c)[0] = (unsigned char)r_map(r, c);
      //      rgb_map_fixed.at<Vec3b>(r, c)[1] = (unsigned char)g_map(r, c);
      //      rgb_map_fixed.at<Vec3b>(r, c)[2] = (unsigned char)b_map(r, c);
      //      depth_cvmap_fixed.at<uchar>(r, c) = (unsigned char)depth_map(r,
      //      c);
    }
  }

  //  imwrite(PROJECT_DIR + std::string("/data/projected_rgb.png"), rgb_map);
  //  imwrite(PROJECT_DIR + std::string("/data/projected_depth.png"),
  //  depth_cvmap); imwrite(PROJECT_DIR +
  //  std::string("/data/projected_rgb_fixed.png"),
  //          rgb_map_fixed);
  //  imwrite(PROJECT_DIR + std::string("/data/projected_depth_fixed.png"),
  //          depth_cvmap_fixed);
  std::cout << "depth and rgb ground truth map created successfully!!!"
            << std::endl;

  VectorXd target_r;
  VectorXd target_g;
  VectorXd target_b;
  VectorXd target_depth;

  // Load and transform basel model
  VectorXd basel_r;
  VectorXd basel_g;
  VectorXd basel_b;
  VectorXd basel_depth;

  BaselModelReader basel_model_reader;

  basel_model_reader.loadBaselModel(PROJECT_DIR + string("/data/basel_data/"));

  Matrix3d R_icp = T_icp.block(0, 0, 3, 3).cast<double>();
  Vector3d t_icp = T_icp.block(0, 3, 3, 1).cast<double>();

  Matrix3d R_pro = T_pro.block(0, 0, 3, 3).cast<double>();
  Vector3d t_pro = T_pro.block(0, 3, 3, 1).cast<double>();

  cout << "Basel model loaded successfully!" << endl;

  if (!basel_model_reader.transformBasel(R_icp, t_icp, R_pro, t_pro,
                                         scale_bs2gt)) {
    std::cout << "basel model transformation failed!!!" << std::endl;
  }

  std::cout << "basel model load successful!!!" << std::endl;

  Vector3d key_point_0(double(-55826), double(45570.4), double(85213.3));
  Vector3d key_point_1(double(54832.1), double(45249.5), double(85281.5));
  Vector3d key_point_2(double(-27423.4), double(-47856.2), double(96637.9));
  Vector3d key_point_3(double(24577.5), double(-48481.9), double(98537.6));

  cout << 1 << endl;

  Vector3d* key_points = new Vector3d[4];
  key_points[0] = key_point_0;
  key_points[1] = key_point_1;
  key_points[2] = key_point_2;
  key_points[3] = key_point_3;

  cout << 2 << endl;

  Vector2d* key_points_transformed = new Vector2d[4];

  for (int i = 0; i < 4; i++) {
    Vector3d result = (scale_bs2gt * R_pro * key_points[i]).colwise() + t_pro;
    result = (R_icp * result).colwise() + t_icp;
    Vector4d result_homo(result.x(), result.y(), result.z(), double(1));
    result = P * result_homo;

    Vector2d result_2d(result.x() / result.z(), result.y() / result.z());
    cout << i << endl;
    key_points_transformed[i] = result_2d;
    cout << i << endl;
  }

  int x_upper = int(key_points_transformed[3].x());
  int x_lower = int(key_points_transformed[2].x());
  int y_lower = int(key_points_transformed[1].y());
  int y_upper = int(key_points_transformed[2].y());

  cout << x_upper << ", " << x_lower << ", " << y_upper << " ," << y_lower
       << endl;

  const MatrixXd& shapeMU = basel_model_reader.getShapeMU();
  const MatrixXd& shapePC = basel_model_reader.getShapePC();
  const MatrixXd& shapeEV = basel_model_reader.getShapeEV();

  const MatrixXd& texMU = basel_model_reader.getTexMU();
  const MatrixXd& texPC = basel_model_reader.getTexPC();
  const MatrixXd& texEV = basel_model_reader.getTexEV();

  MatrixXd tl = basel_model_reader.getTL();

  MatrixXd XYZMatrix = MatrixXd::Zero(53490, 3);
  MatrixXd RGBMatrix = MatrixXd::Zero(54390, 3);

  const std::string filenameMask(PROJECT_DIR +
                                 std::string("/data/face_seg_mask.txt"));

  int mask[53490];
  ifstream f(filenameMask, std::ios::in | std::ios::binary);
  if (!f.is_open()) {
    cout << "load matrix file failed!" << endl;
    return false;
  }
  double data;
  for (int r = 0; r < 53490; r++) {
    f >> data;
    std::cout << data << std::endl;
    mask[r] = int(data);
  }
  f.close();

  cv::Mat rgb_map = cv::imread(PROJECT_DIR + std::string("/data/image_1.png"));

  double alpha[PRIN_COMP_NUM];
  double beta[PRIN_COMP_NUM];
  double sT_bs2gt[6];
  PoseIncrement<double> pose_increment(sT_bs2gt);
  pose_increment.setZero();

  //  cout << "alpha" << alpha << endl;
  //  cout << "beta" << beta << endl;

  // Initialize alpha and beta
  for (int i = 0; i < PRIN_COMP_NUM; i++) {
    alpha[i] = 0.05;
    beta[i] = 0.05;
  };

  ceres::Problem problem;
  int counter = 0;

  for (int i = 0; i < 53490; i++) {  // 53490

    // workable solution without T, input eigen matrix map
    if (1) {  // mask[i] < 3 || i > 42926) {  //
      problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<RGBDCostFunction, 2, PRIN_COMP_NUM,
                                          PRIN_COMP_NUM>(new RGBDCostFunction(
              shapeMU.block(i * 3, 0, 3, 1), texMU.block(i * 3, 0, 3, 1),
              shapePC.block(i * 3, 0, 3, PRIN_COMP_NUM),
              texPC.block(i * 3, 0, 3, PRIN_COMP_NUM), shapeEV, texEV,
              depth_map,
              // neutral_xyz, neutral_rgb, pc_xyz, pc_rgb,
              // shapeEV, texEV, depth_map,
              r_map, g_map, b_map, P)),
          nullptr, alpha, beta);  //, pose_increment.getData()
      cout << i << endl;
      counter++;
    }
  }
  //    // solution without T, input cv::Mat as rgb map
  //    if (mask[i] < 3 || i > 42926) {  //
  //      problem.AddResidualBlock(
  //          new ceres::AutoDiffCostFunction<RGBDCostFunction_v3, 2,
  //          PRIN_COMP_NUM,
  //                                          PRIN_COMP_NUM>(
  //              new RGBDCostFunction_v3(shapeMU.block(i * 3, 0, 3, 1),
  //                                      texMU.block(i * 3, 0, 3, 1),
  //                                      shapePC.block(i * 3, 0, 3,
  //                                      PRIN_COMP_NUM), texPC.block(i * 3,
  //                                      0, 3, PRIN_COMP_NUM), shapeEV,
  //                                      texEV, depth_map,
  //                                      // neutral_xyz, neutral_rgb,
  //                                      pc_xyz,
  //                                      // pc_rgb, shapeEV, texEV,
  //                                      depth_map, rgb_map, P)),
  //          nullptr, alpha, beta);  //, pose_increment.getData()

  //    }
  //  }

  cout << "number of residual added: " << counter << endl;
  // regularizer for alpha
  RegularizerCostFunctor* alpha_reg_functor =
      new RegularizerCostFunctor(PRIN_COMP_NUM, 1.00);
  ceres::CostFunction* alpha_reg_function =
      new ceres::AutoDiffCostFunction<RegularizerCostFunctor, PRIN_COMP_NUM,
                                      PRIN_COMP_NUM>(alpha_reg_functor);
  problem.AddResidualBlock(alpha_reg_function, nullptr, alpha);

  std::cout << "finished adding regularizer for alpha" << std::endl;

  // regularizer for beta
  RegularizerCostFunctor* beta_reg_functor =
      new RegularizerCostFunctor(PRIN_COMP_NUM, 1.0);
  ceres::CostFunction* beta_reg_function =
      new ceres::AutoDiffCostFunction<RegularizerCostFunctor, PRIN_COMP_NUM,
                                      PRIN_COMP_NUM>(beta_reg_functor);
  problem.AddResidualBlock(beta_reg_function, nullptr, beta);

  std::cout << "finished adding regularizer for beta" << std::endl;

  // set upper bound for beta
  for (int i = 0; i < PRIN_COMP_NUM; ++i) {
    problem.SetParameterLowerBound(&beta[0], i, -3.0);
    problem.SetParameterUpperBound(&beta[0], i, 3.0);
  }
  cout << "CERES SOLVER KICKS IN!!!!!!!" << endl;

  ceres::Solver::Options options;
  options.max_num_iterations = 50;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << std::endl;

  cout << "after ceres optimizer: alpha " << endl;

  for (int i = 0; i < PRIN_COMP_NUM; i++) {
    cout << alpha[i] << ",";
  }
  cout << endl << "after ceres optimizer: beta " << endl;
  for (int i = 0; i < PRIN_COMP_NUM; i++) {
    cout << beta[i] << ",";
  }

  cout << PoseIncrement<double>::convertToMatrix(pose_increment) << endl;

  generateFace_v2(shapeMU, shapePC, shapeEV, texMU, texPC, texEV, P, alpha,
                  beta, tl, pose_increment.getData());

  return 0;
}

int dataloader() {
  Mat frame = imread(PROJECT_DIR + String("/data/image_1.png"));

  frontal_face_detector detector = get_frontal_face_detector();
  shape_predictor sp;
  deserialize(PROJECT_DIR + String("/shape_predictor_68_face_landmarks.dat")) >>
      sp;

  //  image_window win, win_faces;
  array2d<rgb_pixel> img;
  load_image(img, PROJECT_DIR + String("/data/image_1.png"));

  // Make the image larger so we can detect small faces.
  pyramid_up(img);

  // Now tell the face detector to give us a list of bounding boxes
  // around all the faces in the image.
  std::vector<dlib::rectangle> dets = detector(img);
  cout << "Number of faces detected: " << dets.size() << endl;

  // Now we will go ask the shape_predictor to tell us the pose of
  // each face we detected.
  std::vector<full_object_detection> shapes;

  for (unsigned long j = 0; j < dets.size(); ++j) {
    full_object_detection shape = sp(img, dets[j]);
    // cout << "number of parts: " << shape.num_parts() << endl;
    // cout << "pixel position of first part:  " << shape.part(0) << endl;
    // cout << "pixel position of second part: " << shape.part(1) << endl;
    shapes.push_back(shape);
  }

  // Load point cloud:
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(
          PROJECT_DIR + std::string("/data/cloud_1.pcd"), *cloud) ==
      -1)  //* load the file
  {
    PCL_ERROR("Couldn't read the pcd file \n");
    return (-1);
  }

  Vertex* landmarks = new Vertex[1 * 6];
  std::vector<int> keypoints_ind_list = {8,  21, 22,
                                         30, 45, 48};  // the 5th is 36
  // visualize the landmarks in opencv
  int counter = 0;
  for (int i : keypoints_ind_list) {  // shapes[0].num_parts()
    int c = shapes[0].part(i).x();
    int r = shapes[0].part(i).y();
    // std::cout << i << ", " << r << ", " << c << std::endl;
    cv::Point right;
    right.x = int(c / 2);
    right.y = int(r / 2);

    // Load corresponding image:
    cv::Mat img = cv::imread("../data/image_1.png");

    // Select a 3D point in the point cloud with row and column indices:
    pcl::PointXYZRGB selected_point = cloud->at(right.x / 2, right.y / 2);
    std::cout << "Selected 3D point " << i << ": " << selected_point.x << " "
              << selected_point.y << " " << selected_point.z << std::endl;
    Vector4f cam_homog(selected_point.x, selected_point.y, selected_point.z,
                       float(1));
    landmarks[counter].position = cam_homog;
    Vector4i rgb_value((int)(selected_point.r), (int)(selected_point.g),
                       (int)(selected_point.b), (int)(255));
    landmarks[counter].color = rgb_value.cast<unsigned char>();
    counter++;

    // Define projection matrix from 3D to 2D:
    // P matrix is in camera_info.yaml
    Eigen::Matrix<float, 3, 4> P;
    P << 1052.667867276341, 0, 962.4130834944134, 0, 0, 1052.020917785721,
        536.2206151001486, 0, 0, 0, 1, 0;

    // 3D to 2D projection:
    // Let's do P*point and rescale X,Y
    Eigen::Vector4f homogeneous_point(selected_point.x, selected_point.y,
                                      selected_point.z, 1);
    Eigen::Vector3f output = P * homogeneous_point;
    output[0] /= output[2];
    output[1] /= output[2];

    std::cout << "Corresponding 2D point in the image " << i << ": "
              << output(0) << " " << output(1) << std::endl;
    //          // 3D Visualization:
    //          pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    //
    //            // Draw output point cloud:
    //            viewer.addCoordinateSystem(0.1);
    //            pcl::visualization::PointCloudColorHandlerRGBField
    //            <pcl::PointXYZRGB> rgb(cloud);
    //            viewer.addPointCloud<pcl::PointXYZRGB>(cloud, rgb,
    //            "cloud");

    // Draw selected 3D point in red:
    selected_point.r = 255;
    selected_point.g = 0;
    selected_point.b = 0;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_to_visualize(
        new pcl::PointCloud<pcl::PointXYZRGB>);
    point_to_visualize->points.push_back(selected_point);
  }
  //            pcl::visualization::PointCloudColorHandlerRGBField
  //            <pcl::PointXYZRGB> red(point_to_visualize);
  //            viewer.addPointCloud<pcl::PointXYZRGB>(point_to_visualize,
  //            red, "point");
  //            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
  //            15, "point"); viewer.setCameraPosition(-0.24917, -0.0187087,
  //            -1.29032, 0.0228136, -0.996651, 0.0785278);

  // Loop for visualization (so that the visualizers are continuously
  // updated):
  std::cout << "Visualization... " << std::endl;

  // back projection
  std::string filenameIn = "./data/";
  std::string filenameBaseOut = PROJECT_DIR + std::string("/data/");

  Vertex* vertices = new Vertex[400 * 400];

  int H = 746;
  int W = 960;
  counter = 0;
  cv::Mat depth_image(1080, 1920, CV_32FC1, Scalar(0));
  for (int h = 346; h < H; h++) {
    for (int w = 560; w < W; w++) {
      // Select a 3D point in the point cloud with row and column indices:
      pcl::PointXYZRGB selected_point = cloud->at(w / 2, h / 2);
      //        std::cout << "Selected 3D point " << counter << ": " <<
      //        selected_point.r
      //                  << " " << selected_point.g << " " <<
      //                  selected_point.b
      //                  << std::endl;
      if (isnan(selected_point.x)) {
        Vector4f cam_homog(float(0), float(0), float(0), float(1));
        vertices[counter].position = cam_homog;
      } else {
        Vector4f cam_homog(selected_point.x, selected_point.y, selected_point.z,
                           float(1));
        vertices[counter].position = cam_homog;
        Vector4i rgb_value((int)(selected_point.r), (int)(selected_point.g),
                           (int)(selected_point.b), (int)(255));
        vertices[counter].color = rgb_value.cast<unsigned char>();
      }

      // Define projection matrix from 3D to 2D:
      // P matrix is in camera_info.yaml
      Eigen::Matrix<float, 3, 4> P;
      P << 1052.667867276341, 0, 962.4130834944134, 0, 0, 1052.020917785721,
          536.2206151001486, 0, 0, 0, 1, 0;

      // 3D to 2D projection:
      // Let's do P*point and rescale X,Y
      Eigen::Vector4f homogeneous_point(selected_point.x, selected_point.y,
                                        selected_point.z, 1);
      Eigen::Vector3f output = P * homogeneous_point;
      output[0] /= output[2];
      output[1] /= output[2];

      counter++;
    }
  }

  // write mesh file
  std::stringstream ss_entire_face;
  std::stringstream ss_landmarks;

  ss_entire_face << filenameBaseOut << "entire_face"
                 << ".off";
  ss_landmarks << filenameBaseOut << "landmarks"
               << ".off";

  std::cout << ss_landmarks.str() << endl;

  if (!WriteMesh(vertices, 400, 400, ss_entire_face.str())) {
    std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
    return -1;
  }
  cout << "1 succeed" << endl;
  if (!WriteMesh(landmarks, 1, 6, ss_landmarks.str())) {
    std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
    return -1;
  }
  cout << "2 succeed" << endl;

  //    delete[] vertices;
  //    delete[] landmarks;
  cout << "delete ok" << endl;
  return 0;
}

using namespace std;
using namespace dlib;

int main() {
  int result = 0;
  // Dataloader
  dataloader();

  cout << "dataloader done" << endl;
  // Procrustes
  double scale_bs2gt = 1;
  Eigen::Matrix4d T_pro;
  Procrustes(T_pro, scale_bs2gt);

  cout << "procrustes done" << endl;

  // ICP
  Eigen::Matrix4d T_icp = Eigen::Matrix4d::Identity();
  getICPTransformation(T_icp);

  cout << "ICP done" << endl;
  // Model Fitting

  // first hard code T_pro, T_icp and scale factor
  //  T_icp << 0.995896, 0.0209807, -0.0880384, 0.0754728, -0.0122658,
  //  0.995073,
  //      0.0983873, -0.0767759, 0.0896689, -0.0969037, 0.991246, 0.020257,
  //      0, 0, 0, 1;
  //  T_pro << 6.95329e-310, 6.95313e-310, 0, 2.97079e-313, 2.14825e-314,
  //      2.24551e-314, 6.95313e-310, 0, 6.95313e-310,
  //      6.95313e-310, 6.95313e-310, 0, 6.95313e-310, 0, 2.14825e-314, 0;
  //  scale_bs2gt = 1.18713e-06;

  //  T_pro << 0.997596, 0.0226103, 0.0655126, -0.13836, 0.00338662,
  //  -0.96006,
  //      0.279774, -0.0322969, 0.0692218, -0.27888, -0.957828, 0.813423, 0,
  //      0, 0, 1;
  //  T_icp << 0.997734, 0.0161618, -0.0653084, 0.0588476, -0.0078826,
  //  0.992113,
  //      0.125094, -0.0977496, 0.0668152, -0.124295, 0.989993, 0.0153022,
  //      0, 0, 0, 1;
  //  scale_bs2gt = 1.18713e-06;

  cout << T_pro << endl;
  cout << T_icp << endl;
  cout << scale_bs2gt << endl;

  result = modelFitting(T_pro, T_icp, scale_bs2gt);
  cout << "New face generated!!!!" << endl;
  return result;
}
