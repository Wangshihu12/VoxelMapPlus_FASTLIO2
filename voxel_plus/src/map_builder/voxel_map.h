#pragma once
#include <list>
#include <vector>
#include <memory>
#include <cstdint>
#include <Eigen/Eigen>
#include <unordered_map>
#include <chrono>
#include <iostream>

#define HASH_P 116101
#define MAX_N 10000000000

namespace lio
{

    class VoxelKey
    {
    public:
        int64_t x, y, z;
        VoxelKey(int64_t _x = 0, int64_t _y = 0, int64_t _z = 0) : x(_x), y(_y), z(_z) {}

        bool operator==(const VoxelKey &other) const
        {
            return (x == other.x && y == other.y && z == other.z);
        }
        // 为 VoxelKey 类型的对象生成哈希值
        struct Hasher
        {
            int64_t operator()(const VoxelKey &k) const
            {
                return ((((k.z) * HASH_P) % MAX_N + (k.y)) * HASH_P) % MAX_N + (k.x);
            }
        };
    };

    struct PointWithCov
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d point;  // 世界坐标系下
        Eigen::Matrix3d cov;    // 世界坐标系下
    };

    struct Plane
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d mean = Eigen::Vector3d::Zero(); // 平面的均值
        Eigen::Matrix3d ppt = Eigen::Matrix3d::Zero();  // 点的乘积
        Eigen::Vector3d norm = Eigen::Vector3d::Zero(); // 法向量
        Eigen::Matrix<double, 6, 6> cov;    // 协方差
        int n = 0;  // 平面中的点数
    };

    struct ResidualData
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d point_lidar;
        Eigen::Vector3d point_world;
        Eigen::Vector3d plane_mean;
        Eigen::Vector3d plane_norm;
        Eigen::Matrix<double, 6, 6> plane_cov;
        Eigen::Matrix3d cov_lidar;
        Eigen::Matrix3d cov_world;
        bool is_valid = false;
        double residual = 0.0;
    };

    class VoxelMap;

    class VoxelGrid
    {
    public:
        VoxelGrid(int _max_point_thresh, int _update_point_thresh, double _plane_thresh, VoxelKey _position, VoxelMap *_map);

        void updatePlane();

        void addToPlane(const PointWithCov &pv);

        void addPoint(const PointWithCov &pv);

        void pushPoint(const PointWithCov &pv);

        void merge();

    public:
        static uint64_t count;
        int max_point_thresh;
        int update_point_thresh;
        double plane_thresh;
        bool is_init;               // 是否初始化
        bool is_plane;              // 是否是平面
        bool update_enable;         // 能否更新
        int newly_add_point;        // 新加入的点
        bool merged;                // 是否被融合
        uint64_t group_id;          // 网格ID
        std::vector<PointWithCov> temp_points;  // 缓存的点
        VoxelKey position;  // 键
        VoxelMap *map;
        std::shared_ptr<Plane> plane;
        Eigen::Vector3d center;
        std::list<VoxelKey>::iterator cache_it;     // 最近更新点云的迭代器
        static double merge_thresh_for_angle;
        static double merge_thresh_for_distance;
    };

    typedef std::unordered_map<VoxelKey, std::shared_ptr<VoxelGrid>, VoxelKey::Hasher> Featmap;

    class VoxelMap
    {
    public:
        VoxelMap(int _max_point_thresh, int _update_point_thresh, double _plane_thresh, double _voxel_size, int capacity = 2000000);

        VoxelKey index(const Eigen::Vector3d &point);

        void build(std::vector<PointWithCov> &pvs);

        void update(std::vector<PointWithCov> &pvs);

        bool buildResidual(ResidualData &data, std::shared_ptr<VoxelGrid> voxel_grid);

    public:
        int max_point_thresh;       // 一个网格包含的最大点阈值
        int update_point_thresh;    // 更新点阈值，超过这个值就更新网格内的平面
        double plane_thresh;        // 平面阈值，是否是平面
        double voxel_size;
        Featmap featmap;
        std::list<VoxelKey> cache;
        int capacity;               // 网格容量
    };

} // namespace lio
