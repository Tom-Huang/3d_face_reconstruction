#pragma once
#include "SimpleMesh.h"

class ProcrustesAligner {
 public:
  Matrix4d estimatePose(const std::vector<Vector3d>& sourcePoints,
                        const std::vector<Vector3d>& targetPoints,
                        double& scale_source2target) {
    ASSERT(sourcePoints.size() == targetPoints.size() &&
           "The number of source and target points should be the same, since "
           "every source point is matched with corresponding target point.");

    // We estimate the pose between source and target points using Procrustes
    // algorithm. Our shapes have the same scale, therefore we don't estimate
    // scale. We estimated rotation and translation from source points to target
    // points.

    auto sourceMean = computeMean(sourcePoints);
    auto targetMean = computeMean(targetPoints);
    scale_source2target = 1;

    // print points for debugging
    for (int i = 0; i < sourcePoints.size(); i++) {
      std::cout << i << std::endl
                << sourcePoints[i] << "..." << std::endl
                << targetPoints[i] << std::endl;
      std::cout << "target points 1st: " << targetPoints[i](0) << std::endl;
    }

    Matrix3d rotation =
        estimateRotation_scaled(sourcePoints, sourceMean, targetPoints,
                                targetMean, scale_source2target);
    Vector3d translation = computeTranslation(sourceMean, targetMean);

    // To apply the pose to point x on shape X in the case of Procrustes, we
    // execute:
    // 1. Translation of a point to the shape Y: x' = x + t
    // 2. Rotation of the point around the mean of shape Y:
    //    y = R (x' - yMean) + yMean = R (x + t - yMean) + yMean = R x + (R t -
    //    R yMean + yMean)

    Matrix4d estimatedPose = Matrix4d::Identity();
    estimatedPose.block(0, 0, 3, 3) = rotation;
    estimatedPose.block(0, 3, 3, 1) =
        scale_source2target * (rotation * translation - rotation * targetMean) +
        targetMean;

    return estimatedPose;
  }

 private:
  Vector3d computeMean(const std::vector<Vector3d>& points) {
    // TODO: Compute the mean of input points.
    Vector3d mean = Vector3d::Zero();
    for (const auto p : points) {
      mean = mean + p;
    }
    mean = mean / float(points.size());

    return mean;
  }

  Matrix3d estimateRotation(const std::vector<Vector3d>& sourcePoints,
                            const Vector3d& sourceMean,
                            const std::vector<Vector3d>& targetPoints,
                            const Vector3d& targetMean) {
    // TODO: Estimate the rotation from source to target points, following the
    // Procrustes algorithm. To compute the singular value decomposition you can
    // use JacobiSVD() from Eigen. Important: The covariance matrices should
    // contain mean-centered source/target points.
    MatrixXd X_hat(sourcePoints.size(), 3);
    MatrixXd X(targetPoints.size(), 3);
    for (int i = 0; i < sourcePoints.size(); i++) {
      Vector3d sp = (sourcePoints[i] - sourceMean);
      Vector3d tp = (targetPoints[i] - targetMean);
      X(i, 0) = tp[0];
      X(i, 1) = tp[1];
      X(i, 2) = tp[2];
      X_hat(i, 0) = sp[0];
      X_hat(i, 1) = sp[1];
      X_hat(i, 2) = sp[2];
    }
    // std::cout << X.size() << X_hat.size() << std::endl;
    JacobiSVD<MatrixXd> svd(X.transpose() * X_hat, ComputeThinU | ComputeThinV);
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();
    return U * V.transpose();
  }

  Matrix3d estimateRotation_scaled(const std::vector<Vector3d>& sourcePoints,
                                   const Vector3d& sourceMean,
                                   const std::vector<Vector3d>& targetPoints,
                                   const Vector3d& targetMean,
                                   double& scale_source2target) {
    // TODO: Estimate the rotation from source to target points, following the
    // Procrustes algorithm. To compute the singular value decomposition you can
    // use JacobiSVD() from Eigen. Important: The covariance matrices should
    // contain mean-centered source/target points.

    MatrixX3d X_hat(sourcePoints.size(), 3);
    MatrixX3d X(targetPoints.size(), 3);
    MatrixX3d X_norm(targetPoints.size(), 3);
    MatrixX3d X_hat_norm(sourcePoints.size(), 3);
    for (int i = 0; i < sourcePoints.size(); i++) {
      Vector3d sp = (sourcePoints[i] - sourceMean);
      Vector3d tp = (targetPoints[i] - targetMean);

      X.row(i) = tp;
      X_hat.row(i) = sp;
    }
    double scale_target = normalizesourcePoints(X, X_norm);
    double scale_source = normalizesourcePoints(X_hat, X_hat_norm);
    scale_source2target = scale_target / scale_source;

    std::cout << X_norm.row(0) << ", " << X_hat_norm.row(0) << std::endl;
    JacobiSVD<MatrixXd> svd(X_norm.transpose() * X_hat_norm,
                            ComputeThinU | ComputeThinV);
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();
    std::cout << "X_norm " << std::endl << X_norm << std::endl;
    std::cout << "X_hat_norm " << std::endl << X_hat_norm << std::endl;
    std::cout << "scale target: " << scale_target << std::endl;
    std::cout << "scale source: " << scale_source << std::endl;
    std::cout << "A" << std::endl
              << X_norm.transpose() * X_hat_norm << std::endl;
    std::cout << "U " << std::endl << U << std::endl;
    std::cout << "V " << std::endl << V << std::endl;
    return U * V.transpose();
  }

  Vector3d computeTranslation(const Vector3d& sourceMean,
                              const Vector3d& targetMean) {
    // TODO: Compute the translation vector from source to target points.

    return targetMean - sourceMean;
  }

  // TODO PROJECT: compute scale
  double calculateScale(const MatrixX3d& centeredPointsMatrix) {
    double scale = centeredPointsMatrix.colwise().norm().mean();
    return scale;
  }

  double normalizesourcePoints(const MatrixX3d& centeredPointsMatrix,
                               MatrixX3d& normalizedsourcePointsMatrix) {
    double scale = calculateScale(centeredPointsMatrix);

    normalizedsourcePointsMatrix = centeredPointsMatrix / scale;
    return scale;
  }
};
