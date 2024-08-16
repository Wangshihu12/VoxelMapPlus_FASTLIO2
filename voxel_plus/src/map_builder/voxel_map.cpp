#include "voxel_map.h"

namespace lio
{

    uint64_t VoxelGrid::count = 0;

    double VoxelGrid::merge_thresh_for_angle = 0.1;
    double VoxelGrid::merge_thresh_for_distance = 0.04;

    VoxelGrid::VoxelGrid(int _max_point_thresh, int _update_point_thresh, double _plane_thresh, VoxelKey _position, VoxelMap *_map)
        : max_point_thresh(_max_point_thresh),
          update_point_thresh(_update_point_thresh),
          plane_thresh(_plane_thresh),
          position(_position.x, _position.y, _position.z)
    {
        merged = false;                         // 是否合并
        group_id = VoxelGrid::count++;          // 网格ID
        is_init = false;                        // 是否初始化
        is_plane = false;                       // 是否为平面
        temp_points.reserve(max_point_thresh);  // 点的缓存
        newly_add_point = 0;
        plane = std::make_shared<Plane>();
        update_enable = true;                   // 更新可用
        map = _map;
        // center = Eigen::Vector3d(position.x + 0.5, position.y + 0.5, position.z + 0.5) * map->voxel_size;
    }

    void VoxelGrid::addToPlane(const PointWithCov &pv)
    {
        // 对平面的均值进行增量更新
        plane->mean += (pv.point - plane->mean) / (plane->n + 1.0);
        // 更新点的乘积和
        plane->ppt += pv.point * pv.point.transpose();
        // 更新点数
        plane->n += 1;
    }

    void VoxelGrid::addPoint(const PointWithCov &pv)
    {
        // 加入平面和缓存
        addToPlane(pv);
        temp_points.push_back(pv);
    }

    void VoxelGrid::pushPoint(const PointWithCov &pv)
    {
        if (!is_init)
        {
            // 已经初始化，加入点到平面和缓存，更新平面
            addToPlane(pv);
            temp_points.push_back(pv);
            updatePlane();
        }
        else
        {
            // 未初始化
            if (is_plane)
            {
                if (update_enable)
                {
                    addToPlane(pv);
                    temp_points.push_back(pv);
                    newly_add_point++;
                    // 更新点达到阈值，更新平面并清零
                    if (newly_add_point >= update_point_thresh)
                    {
                        updatePlane();
                        newly_add_point = 0;
                    }
                    // 缓存点达到阈值，禁用更新并清空缓存点
                    if (temp_points.size() >= max_point_thresh)
                    {
                        update_enable = false;
                        std::vector<PointWithCov>().swap(temp_points);
                    }
                }
                else
                {
                    // 如果是平面但无法更新，融合平面
                    merge();
                }
            }
            else
            {
                // 如果不是平面但更新可用
                if (update_enable)
                {
                    addToPlane(pv);
                    temp_points.push_back(pv);
                    newly_add_point++;
                    if (newly_add_point >= update_point_thresh)
                    {
                        updatePlane();
                        newly_add_point = 0;
                    }
                    if (temp_points.size() >= max_point_thresh)
                    {
                        update_enable = false;
                        std::vector<PointWithCov>().swap(temp_points);
                    }
                }
            }
        }
    }

    void VoxelGrid::updatePlane()
    {
        assert(temp_points.size() == plane->n);
        if (plane->n < update_point_thresh)
            return;
        is_init = true;
        // 计算平面的协方差
        Eigen::Matrix3d cov = plane->ppt / static_cast<double>(plane->n) - plane->mean * plane->mean.transpose();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
        Eigen::Matrix3d evecs = es.eigenvectors();  // 特征向量
        Eigen::Vector3d evals = es.eigenvalues();   // 特征值
        if (evals(0) > plane_thresh)
        {
            // 大于阈值，则认为不是平面
            is_plane = false;
            return;
        }
        is_plane = true;
        Eigen::Matrix3d J_Q = Eigen::Matrix3d::Identity() / static_cast<double>(plane->n);
        Eigen::Vector3d plane_norm = evecs.col(0);
        for (PointWithCov &pv : temp_points)
        {

            Eigen::Matrix<double, 6, 3> J;
            Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
            for (int m = 1; m < 3; m++)
            {
                Eigen::Matrix<double, 1, 3> F_m = (pv.point - plane->mean).transpose() / ((plane->n) * (evals(0) - evals(m))) *
                                                  (evecs.col(m) * plane_norm.transpose() +
                                                   plane_norm * evecs.col(m).transpose());
                F.row(m) = F_m;
            }
            J.block<3, 3>(0, 0) = evecs * F;
            J.block<3, 3>(3, 0) = J_Q;
            // 更新协方差
            plane->cov += J * pv.cov * J.transpose();
        }
        double axis_distance = -plane->mean.dot(plane_norm);
        if (axis_distance < 0.0)
            plane_norm = -plane_norm;
        // 更新法向量和中心点
        plane->norm = plane_norm;
        center = plane->mean;
    }

    void VoxelGrid::merge()
    {
        std::vector<VoxelKey> near;
        near.push_back(VoxelKey(position.x - 1, position.y, position.z));
        near.push_back(VoxelKey(position.x, position.y - 1, position.z));
        near.push_back(VoxelKey(position.x, position.y, position.z - 1));
        near.push_back(VoxelKey(position.x + 1, position.y, position.z));
        near.push_back(VoxelKey(position.x, position.y + 1, position.z));
        near.push_back(VoxelKey(position.x, position.y, position.z + 1));

        for (VoxelKey &k : near)
        {
            auto it = map->featmap.find(k);
            if (it != map->featmap.end())
            {
                std::shared_ptr<VoxelGrid> near_node = it->second;
                if (near_node->group_id == group_id || near_node->update_enable || !near_node->is_plane)
                    continue;
                // 计算平面法向量的夹角距离和平面距离
                double norm_distance = 1.0 - near_node->plane->norm.dot(plane->norm);
                double axis_distance = std::abs(near_node->plane->norm.dot(near_node->plane->mean) - plane->norm.dot(plane->mean));
                // 如果大于阈值，则不能合并
                if (norm_distance > merge_thresh_for_angle || axis_distance > merge_thresh_for_distance)
                    continue;
                // 计算当前体素和相邻体素的平面模型的协方差迹 tn0, tm0, tn1, tm1。
                // 使用加权平均法计算新的平面均值 new_mean 和法向量 new_norm。
                // 更新平面模型的协方差 new_cov
                double tn0 = plane->cov.block<3, 3>(0, 0).trace(),
                       tm0 = plane->cov.block<3, 3>(3, 3).trace(),
                       tn1 = near_node->plane->cov.block<3, 3>(0, 0).trace(),
                       tm1 = near_node->plane->cov.block<3, 3>(3, 3).trace();
                double tc0 = tn0 + tm0, tc1 = tn1 + tm1;
                Eigen::Vector3d new_mean = tm0 * near_node->plane->mean + tm1 * plane->mean / (tm0 + tm1);
                Eigen::Vector3d new_norm = tn0 * near_node->plane->norm + tn1 * plane->norm / (tn0 + tn1);
                Eigen::Matrix<double, 6, 6> new_cov = (tc0 * tc0 * near_node->plane->cov + tc1 * tc1 * plane->cov) / ((tc0 + tc1) * (tc0 + tc1));

                near_node->group_id = group_id;     // 把当前平面的 id 赋值给合并后的平面
                merged = true;  // 表示已经合并过
                near_node->merged = true;

                if (-new_mean.dot(new_norm) < 0.0)
                    new_norm = -new_norm;

                plane->mean = new_mean;
                plane->norm = new_norm;
                plane->cov = new_cov;

                near_node->plane->mean = new_mean;
                near_node->plane->norm = new_norm;
                near_node->plane->cov = new_cov;
            }
        }
    }

    VoxelMap::VoxelMap(int _max_point_thresh, int _update_point_thresh, double _plane_thresh, double _voxel_size, int _capacity) : max_point_thresh(_max_point_thresh), update_point_thresh(_update_point_thresh), plane_thresh(_plane_thresh), voxel_size(_voxel_size), capacity(_capacity)
    {
        featmap.clear();
        cache.clear();
    }

    // 将三维坐标转换为体素网格的索引
    VoxelKey VoxelMap::index(const Eigen::Vector3d &point)
    {
        Eigen::Vector3d idx = (point / voxel_size).array().floor();
        return VoxelKey(static_cast<int64_t>(idx(0)), static_cast<int64_t>(idx(1)), static_cast<int64_t>(idx(2)));
    }

    void VoxelMap::build(std::vector<PointWithCov> &pvs)
    {
        // 遍历点云
        for (PointWithCov &pv : pvs)
        {
            // 计算体素位置
            VoxelKey k = index(pv.point);
            auto it = featmap.find(k);
            if (it == featmap.end())
            {
                // 如果不存在,创建一个新的体素
                featmap[k] = std::make_shared<VoxelGrid>(max_point_thresh, update_point_thresh, plane_thresh, k, this);
                cache.push_front(k);
                featmap[k]->cache_it = cache.begin();

                if (cache.size() > capacity)
                {
                    featmap.erase(cache.back());
                    cache.pop_back();
                }
            }
            else
            {
                // 如果体素存在,将该体素移动到 cache 的前端，表示最近使用的
                cache.splice(cache.begin(), cache, featmap[k]->cache_it);
            }
            // 加入该体素的平面
            featmap[k]->addPoint(pv);
        }

        for (auto it = featmap.begin(); it != featmap.end(); it++)
        {
            // 遍历 featmap 更新平面
            it->second->updatePlane();
        }
    }

    void VoxelMap::update(std::vector<PointWithCov> &pvs)
    {

        for (PointWithCov &pv : pvs)
        {
            VoxelKey k = index(pv.point);
            auto it = featmap.find(k);
            if (it == featmap.end())
            {
                featmap[k] = std::make_shared<VoxelGrid>(max_point_thresh, update_point_thresh, plane_thresh, k, this);
                cache.push_front(k);
                featmap[k]->cache_it = cache.begin();
                if (cache.size() > capacity)
                {
                    featmap.erase(cache.back());
                    cache.pop_back();
                }
            }
            else
            {
                cache.splice(cache.begin(), cache, featmap[k]->cache_it);
            }
            featmap[k]->pushPoint(pv);
        }
    }

    bool VoxelMap::buildResidual(ResidualData &data, std::shared_ptr<VoxelGrid> voxel_grid)
    {
        data.is_valid = false;
        if (voxel_grid->is_plane)
        {
            Eigen::Vector3d p2m = (data.point_world - voxel_grid->plane->mean);
            data.plane_norm = voxel_grid->plane->norm;
            data.plane_mean = voxel_grid->plane->mean;
            data.residual = data.plane_norm.dot(p2m);
            Eigen::Matrix<double, 1, 6> J_nq;
            J_nq.block<1, 3>(0, 0) = p2m;
            J_nq.block<1, 3>(0, 3) = -data.plane_norm;
            double sigma_l = J_nq * data.plane_cov * J_nq.transpose();
            sigma_l += data.plane_norm.transpose() * data.cov_world * data.plane_norm;
            if (std::abs(data.residual) < 3.0 * sqrt(sigma_l))
                data.is_valid = true;
        }
        return data.is_valid;
    }

} // namespace lio
