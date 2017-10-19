#ifndef RMPE_DATASET_SERVER_DATATRANSFORMER_H
#define RMPE_DATASET_SERVER_DATATRANSFORMER_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"

using namespace cv;
using namespace std;

struct TransformationParameter {
  bool mirror = false;
  int crop_size = 0;
  int stride = 4;
  float flip_prob = 0.5;
  float max_rotate_degree = 5.0;
  int crop_size_x = 368;
  int crop_size_y = 368;
  float scale_prob = 0.5;
  float scale_min = 0.9;
  float scale_max = 1.1;
  float target_dist = 1.0;
  float center_perterb_max = 10.0;
  float sigma = 7.0;
  float clahe_tile_size = 8.0;
  float clahe_clip_limit = 4.0;
  bool do_clahe = false;
  int num_parts = 14;
  int num_parts_in_annot = 16;
  int num_total_augs = 82;
  string aug_way = "rand";
  int gray = 0;
  bool transform_body_joint = true;
};

class DataTransformer {
public:
  explicit DataTransformer(const TransformationParameter& param);

  /**
  * @brief Initialize the Random number generations if needed by the transformation.
  */
  void InitRand();

  /**
  * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
  *
  * @param n
  *    The upperbound (exclusive) value of the random number.
  * @return
  *    A uniformly random integer value from ({0, 1, ..., n-1}).
  */
  int Rand(int n);

  /**
  * @brief Transform a sample.
  *
  * @param data
  *    Datum buffer
  * @param datum_channels
  *    Number of channels. 3 channels of image + 1 channel of metadata + (1 or 2) channels for masks
  * @param datum_height
  *    Height
  * @param datum_width
  *    Weight
  * @param transformed_data
  *    Output transformed sample
  * @param transformed_label
  *    Output label
  */
  void Transform(const uchar *data, const int datum_channels, const int datum_height, const int datum_width, uchar* transformed_data, double* transformed_label);
private:

  struct AugmentSelection {
    bool flip;
    float degree;
    Size crop;
    float scale;
  };

  struct Joints {
    vector<Point2f> joints;
    vector<float> is_visible;
  };

  struct MetaData {
    string dataset;
    Size img_size;
    bool is_validation;
    int num_other_people;
    int people_index;
    int annolist_index;
    int write_number;
    int total_write_number;
    int epoch;
    Point2f objpos;
    float scale_self;
    Joints joint_self;

    vector<Point2f> objpos_other;
    vector<float> scale_other;
    vector<Joints> joint_others;
  };

  void TransformMetaJoints(MetaData& meta);
  void TransformJoints(Joints& joints);
  bool OnPlane(Point p, Size img_size);
  bool AugmentationFlip(Mat& img, Mat& img_aug, Mat& mask_miss, MetaData& meta);
  float AugmentationRotate(Mat& img_src, Mat& img_aug, Mat& mask_miss, MetaData& meta);
  float AugmentationScale(Mat& img, Mat& img_temp, Mat& mask_miss, MetaData& meta);
  Size AugmentationCroppad(Mat& img_temp, Mat& img_aug, Mat& mask_miss, Mat& mask_all_aug, MetaData& meta);
  void GenerateLabelMap(double*, Mat&, MetaData meta);
  void PutGaussianMaps(double* entry, Point2f center, int stride, int grid_x, int grid_y, float sigma);
  void PutVecMaps(double* entryX, double* entryY, Mat& count, Point2f centerA, Point2f centerB, int stride, int grid_x, int grid_y, float sigma, int thre);
  void Clahe(Mat& img, int, int);
  void ReadMetaData(MetaData& meta, const uchar *data, size_t offset3, size_t offset1);
  void SwapLeftRight(Joints& j);
  void SetAugTable(int numData);
  void RotatePoint(Point2f& p, Mat R);

  // parameters
  TransformationParameter param_;

  // random numb generator
  boost::shared_ptr<RNGen::RNG> rng_;

  // if augmentation based on table
  vector<vector<float> > aug_degs_;
  vector<vector<int> > aug_flips_;
  bool is_table_set_;

  // number of parts
  int np;

  // number of parts in annotation
  int np_ann;
};


#endif //RMPE_DATASET_SERVER_DATATRANSFORMER_H
