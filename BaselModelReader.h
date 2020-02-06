#pragma once

#include <fstream>
#include <iostream>

#include "Eigen.h"

#define PRIN_COMP_NUM_BS 199
using namespace std;

class BaselModelReader {
 public:
  BaselModelReader() {}

  bool loadBaselModel(const string bs_foldername) {
    std::string tl_filename = bs_foldername + string("basel_tl.txt");
    std::string shapeMU_filename = bs_foldername + string("basel_shapeMU.txt");
    std::string shapePC_filename = bs_foldername + string("basel_shapePC.txt");
    std::string shapeEV_filename = bs_foldername + string("basel_shapeEV.txt");
    std::string texMU_filename = bs_foldername + string("basel_texMU.txt");
    std::string texPC_filename = bs_foldername + string("basel_texPC.txt");
    std::string texEV_filename = bs_foldername + string("basel_texEV.txt");

    if (!loadMatrixfromFile(tl_filename, tl, tl.rows(), tl.cols())) {
      cout << "load basel_tl.txt error!" << endl;
      return false;
    }
    if (!loadMatrixfromFile(shapeMU_filename, shapeMU, shapeMU.rows(),
                            shapeMU.cols())) {
      cout << "load basel_shapeMU.txt error!" << endl;
      return false;
    }
    if (!loadMatrixfromFile(shapePC_filename, shapePC, shapePC.rows(),
                            shapePC.cols())) {
      cout << "load basel_shapePC.txt error!" << endl;
      return false;
    }
    if (!loadMatrixfromFile(shapeEV_filename, shapeEV, shapeEV.rows(),
                            shapeEV.cols())) {
      cout << "load basel_shapeEV.txt error!" << endl;
      return false;
    }
    if (!loadMatrixfromFile(texMU_filename, texMU, texMU.rows(),
                            texMU.cols())) {
      cout << "load basel_texMU.txt error!" << endl;
      return false;
    }
    if (!loadMatrixfromFile(texPC_filename, texPC, texPC.rows(),
                            texPC.cols())) {
      cout << "load basel_texPC.txt error!" << endl;
      return false;
    }
    if (!loadMatrixfromFile(texEV_filename, texEV, texEV.rows(),
                            texEV.cols())) {
      cout << "load basel_texEV.txt error!" << endl;
      return false;
    }
    return true;
  }

  // transform Basel model to ground truth depth map using transformation
  // matrixes acquired from procrustes and icp
  // T_icp * (scale_bs2gt * R_pro * (MU + PC * EV) + t_pro) =
  // T_icp * (scale_bs2gt * R_pro * MU + t_pro) + T_icp * scale_bs2gt * R_pro *
  // PC * EV
  bool transformBasel(const Eigen::Matrix3d R_icp, const Eigen::Vector3d t_icp,
                      const Eigen::Matrix3d R_pro, const Eigen::Vector3d t_pro,
                      const double scale_bs2gt) {
    for (int i = 0; i < (shapeMU.rows() / 3); i++) {
      std::cout << i << std::endl;

      shapeMU.block(3 * i, 0, 3, 1) =
          R_icp *
              ((scale_bs2gt * R_pro * shapeMU.block(3 * i, 0, 3, 1)).colwise() +
               t_pro) +
          t_icp;

      Eigen::MatrixXd test = R_pro * shapePC.block(3 * i, 0, 3, 1);

      shapePC.block(3 * i, 0, 3, shapePC.cols()) =
          R_icp *
          ((scale_bs2gt * R_pro * shapePC.block(3 * i, 0, 3, shapePC.cols())));
      //      std::cout << shapePC.block(3 * i, 0, 3, shapePC.cols()) <<
      //      std::endl;
    }
    return true;
  }
  Eigen::MatrixXd getTL() { return tl; }

  Eigen::MatrixXd getShapeMU() { return shapeMU; }

  Eigen::MatrixXd getShapePC() { return shapePC; }

  Eigen::MatrixXd getShapeEV() { return shapeEV; }

  Eigen::MatrixXd getTexMU() { return texMU; }

  Eigen::MatrixXd getTexPC() { return texPC; }

  Eigen::MatrixXd getTexEV() { return texEV; }

  Eigen::MatrixXd getShapeMUat(const int& vertex_ind) const {
    return shapeMU.block(3 * vertex_ind, 0, 3, 1);
  }

  Eigen::MatrixXd getShapePCat(const int& vertex_ind,
                               const int& prin_comp) const {
    return shapePC.block(3 * vertex_ind, 0, 3, prin_comp);
  }

  Eigen::MatrixXd getShapeEVat(const int& prin_comp) const {
    return shapeEV.block(0, 0, prin_comp, 1);
  }

  Eigen::MatrixXd getTexMUat(const int& vertex_ind) const {
    return texMU.block(3 * vertex_ind, 0, 3, 1);
  }

  Eigen::MatrixXd getTexPCat(const int& vertex_ind,
                             const int& prin_comp) const {
    return texPC.block(3 * vertex_ind, 0, 3, prin_comp);
  }

  const Eigen::MatrixXd getTexEVat(const int& prin_comp) const {
    return texEV.block(0, 0, prin_comp, 1);
  }

 private:
  Eigen::MatrixXd tl = Eigen::Matrix<double, 106466, 3>::Zero();
  Eigen::MatrixXd shapeMU = Eigen::Matrix<double, 160470, 1>::Zero();
  Eigen::MatrixXd shapePC =
      Eigen::Matrix<double, 160470, PRIN_COMP_NUM_BS>::Zero();
  Eigen::MatrixXd shapeEV = Eigen::Matrix<double, PRIN_COMP_NUM_BS, 1>::Zero();
  Eigen::MatrixXd texMU = Eigen::Matrix<double, 160470, 1>::Zero();
  Eigen::MatrixXd texPC =
      Eigen::Matrix<double, 160470, PRIN_COMP_NUM_BS>::Zero();
  Eigen::MatrixXd texEV = Eigen::Matrix<double, PRIN_COMP_NUM_BS, 1>::Zero();

  bool loadMatrixfromFile(const string filename, Eigen::MatrixXd& mat,
                          const int rownum, const int colnum) {
    ifstream f(filename, std::ios::in | std::ios::binary);

    if (!f.is_open()) {
      cout << "load matrix file failed!" << endl;
      return false;
    }
    double data;
    for (int r = 0; r < rownum; r++) {
      for (int c = 0; c < colnum; c++) {
        f >> data;
        mat(r, c) = data;
      }
    }
    f.close();
    return true;
  }
};
