#include <queue>
#include <chrono>
#include <ros/ros.h>
#include "std_manager/descriptor.h"
#include "interface/PointCloudWithOdom.h"
#include <tf2_ros/transform_broadcaster.h>
#include <pcl_conversions/pcl_conversions.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

geometry_msgs::TransformStamped eigen2Transform(const Eigen::Matrix3d &rot, const Eigen::Vector3d &pos, const std::string &frame_id, const std::string &child_frame_id, const double &timestamp)
{
    geometry_msgs::TransformStamped transform;
    transform.header.frame_id = frame_id;
    transform.header.stamp = ros::Time().fromSec(timestamp);
    transform.child_frame_id = child_frame_id;
    transform.transform.translation.x = pos(0);
    transform.transform.translation.y = pos(1);
    transform.transform.translation.z = pos(2);
    Eigen::Quaterniond q = Eigen::Quaterniond(rot);

    transform.transform.rotation.w = q.w();
    transform.transform.rotation.x = q.x();
    transform.transform.rotation.y = q.y();
    transform.transform.rotation.z = q.z();
    return transform;
}

struct Config
{
    std::string map_frame = "map";
    std::string local_frame = "lidar";
    double ds_size = 0.25;
    int sub_frame_num = 10;
    std::string odom_cloud_topic = "/lio_node/cloud_with_odom";
};

struct DataGroup
{
    std::mutex buffer_mutex;
    std::queue<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloud_buffer;
    std::queue<Eigen::Affine3d> pose_buffer;
    std::queue<double> time_buffer;

    double current_time;
    Eigen::Affine3d current_pose;
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr key_cloud;

    size_t cloud_idx = 0;
    std::vector<Eigen::Affine3d> pose_vec;
    std::vector<Eigen::Affine3d> origin_pose_vec;
    std::vector<std::pair<int, int>> loop_container;

    gtsam::Values initial;
    gtsam::NonlinearFactorGraph graph;
    gtsam::noiseModel::Diagonal::shared_ptr odometryNoise;
    gtsam::noiseModel::Base::shared_ptr robustLoopNoise;
    std::shared_ptr<gtsam::ISAM2> isam;

    bool has_loop_flag = false;
};

class PGONode
{
public:
    PGONode() : nh("~")
    {
        loadParams();
        initSubScribers();
        initPublishers();
        initSAM();
        main_loop = nh.createTimer(ros::Duration(0.05), &PGONode::mainLoopCB, this);
        std_manager = std::make_shared<std_desc::STDManager>(std_config);
    }

    void loadParams()
    {
        nh.param<std::string>("map_frame", node_config.map_frame, "map");
        nh.param<std::string>("odom_cloud_topic", node_config.odom_cloud_topic, "/lio_node/cloud_with_odom");
        nh.param<double>("ds_size", node_config.ds_size, 0.25);
        nh.param<int>("sub_frame_num", node_config.sub_frame_num, 10);

        nh.param<double>("voxel_size", std_config.voxel_size, 1.0);
        nh.param<int>("voxel_min_point", std_config.voxel_min_point, 10);
        nh.param<double>("voxel_plane_thresh", std_config.voxel_plane_thresh, 0.01);
        nh.param<double>("norm_merge_thresh", std_config.norm_merge_thresh, 0.1);

        nh.param<double>("proj_2d_resolution", std_config.proj_2d_resolution, 0.25);
        nh.param<double>("proj_min_dis", std_config.proj_min_dis, 0.0001);
        nh.param<double>("proj_max_dis", std_config.proj_max_dis, 5.0);

        nh.param<int>("nms_2d_range", std_config.nms_2d_range, 5);
        nh.param<double>("nms_3d_range", std_config.nms_3d_range, 2.0);
        nh.param<double>("corner_thresh", std_config.corner_thresh, 10.0);
        nh.param<int>("max_corner_num", std_config.max_corner_num, 100);
        nh.param<double>("min_side_len", std_config.min_side_len, 2.0);
        nh.param<double>("max_side_len", std_config.proj_max_dis, 30.0);

        nh.param<int>("desc_search_range", std_config.desc_search_range, 15);

        nh.param<double>("side_resolution", std_config.side_resolution, 0.2);
        nh.param<double>("rough_dis_threshold", std_config.rough_dis_threshold, 0.03);
        nh.param<int>("skip_near_num", std_config.skip_near_num, 50);
        nh.param<int>("candidate_num", std_config.candidate_num, 50);
        nh.param<double>("vertex_diff_threshold", std_config.vertex_diff_threshold, 0.7);
        nh.param<double>("verify_dis_thresh", std_config.verify_dis_thresh, 3.0);
        nh.param<double>("geo_verify_dis_thresh", std_config.geo_verify_dis_thresh, 0.3);
        nh.param<double>("icp_thresh", std_config.icp_thresh, 0.5);

        nh.param<double>("iter_eps", std_config.iter_eps, 0.001);
    }

    void initSAM()
    {
        data_group.odometryNoise = gtsam::noiseModel::Diagonal::Variances(
            (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        gtsam::Vector robustNoiseVector6(6);
        robustNoiseVector6 << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;
        gtsam::noiseModel::Base::shared_ptr robustLoopNoise =
            gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Cauchy::Create(1), gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6));
        gtsam::ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.01;
        parameters.relinearizeSkip = 1;
        data_group.isam = std::make_shared<gtsam::ISAM2>(parameters);
    }

    void initSubScribers()
    {
        odom_cloud_sub = nh.subscribe(node_config.odom_cloud_topic, 100, &PGONode::odomCloudCB, this);
    }

    void publishPath(ros::Publisher &pub, std::vector<Eigen::Affine3d> &pose_list)
    {
        if (pub.getNumSubscribers() < 1)
            return;
        nav_msgs::Path path;
        for (int i = 0; i < pose_list.size(); i++)
        {
            geometry_msgs::PoseStamped msg_pose;
            msg_pose.pose.position.x = pose_list[i].translation()[0];
            msg_pose.pose.position.y = pose_list[i].translation()[1];
            msg_pose.pose.position.z = pose_list[i].translation()[2];
            Eigen::Quaterniond pose_q(pose_list[i].rotation());
            msg_pose.header.frame_id = node_config.map_frame;
            msg_pose.pose.orientation.x = pose_q.x();
            msg_pose.pose.orientation.y = pose_q.y();
            msg_pose.pose.orientation.z = pose_q.z();
            msg_pose.pose.orientation.w = pose_q.w();
            path.poses.push_back(msg_pose);
        }

        path.header.stamp = ros::Time().fromSec(data_group.current_time);
        path.header.frame_id = node_config.map_frame;
        pub.publish(path);
    }

    void publishLoopConstraints()
    {
        if (loop_contraints_pub.getNumSubscribers() < 1)
            return;
        if (data_group.loop_container.size() == 0)
            return;
        visualization_msgs::MarkerArray marker_array;
        visualization_msgs::Marker marker_node;
        marker_node.header.frame_id = node_config.map_frame;
        marker_node.action = visualization_msgs::Marker::ADD;
        marker_node.type = visualization_msgs::Marker::SPHERE_LIST;
        marker_node.ns = "loop_nodes";
        marker_node.id = 0;
        marker_node.pose.orientation.w = 1;
        marker_node.scale.x = 0.3;
        marker_node.scale.y = 0.3;
        marker_node.scale.z = 0.3;
        marker_node.color.r = 0;
        marker_node.color.g = 0.8;
        marker_node.color.b = 1;
        marker_node.color.a = 1;

        visualization_msgs::Marker marker_edge;
        marker_edge.header.frame_id = node_config.map_frame;
        marker_edge.action = visualization_msgs::Marker::ADD;
        marker_edge.type = visualization_msgs::Marker::LINE_LIST;
        marker_edge.ns = "loop_edges";
        marker_edge.id = 1;
        marker_edge.pose.orientation.w = 1;
        marker_edge.scale.x = 0.1;
        marker_edge.color.r = 0.9;
        marker_edge.color.g = 0.9;
        marker_edge.color.b = 0;
        marker_edge.color.a = 1;

        for (auto it = data_group.loop_container.begin(); it != data_group.loop_container.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            p.x = data_group.pose_vec[key_cur * node_config.sub_frame_num].translation().x();
            p.y = data_group.pose_vec[key_cur * node_config.sub_frame_num].translation().y();
            p.z = data_group.pose_vec[key_cur * node_config.sub_frame_num].translation().z();
            marker_node.points.push_back(p);
            marker_edge.points.push_back(p);
            p.x = data_group.pose_vec[key_pre * node_config.sub_frame_num].translation().x();
            p.y = data_group.pose_vec[key_pre * node_config.sub_frame_num].translation().y();
            p.z = data_group.pose_vec[key_pre * node_config.sub_frame_num].translation().z();
            marker_node.points.push_back(p);
            marker_edge.points.push_back(p);
        }

        marker_array.markers.push_back(marker_node);
        marker_array.markers.push_back(marker_edge);
        loop_contraints_pub.publish(marker_array);
    }

    void initPublishers()
    {
        path_pub = nh.advertise<nav_msgs::Path>("origin_path", 10000);
        correct_path_pub = nh.advertise<nav_msgs::Path>("correct_path", 10000);
        loop_contraints_pub = nh.advertise<visualization_msgs::MarkerArray>("loop_contriants", 10);
    }

    void odomCloudCB(const interface::PointCloudWithOdom::ConstPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_group.buffer_mutex);
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(msg->cloud, *cloud);
        node_config.local_frame = msg->header.frame_id;
        data_group.cloud_buffer.push(cloud);
        Eigen::Quaterniond rotation(msg->pose.pose.orientation.w,
                                    msg->pose.pose.orientation.x,
                                    msg->pose.pose.orientation.y,
                                    msg->pose.pose.orientation.z);
        Eigen::Vector3d translation(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
        Eigen::Affine3d pose = Eigen::Affine3d::Identity();
        pose.linear() = rotation.toRotationMatrix();
        pose.translation() = translation;
        data_group.pose_buffer.push(pose);
        data_group.time_buffer.push(msg->header.stamp.toSec());
    }

    /**
     * [功能描述]：位姿图优化（PGO）主循环回调函数，负责处理点云数据、构建因子图、
     *            执行回环检测、进行增量式位姿图优化，并发布优化后的结果。
     * @param e：ROS定时器事件对象，包含定时触发的相关信息。
     */
    void mainLoopCB(const ros::TimerEvent &e)
    {
        // ==================== 第一部分：从缓冲区获取数据 ====================
        {
            // 使用互斥锁保护缓冲区的线程安全访问
            std::lock_guard<std::mutex> lock(data_group.buffer_mutex);
            // 如果点云缓冲区为空，直接返回
            if (data_group.cloud_buffer.size() < 1)
                return;
            // 获取缓冲区队首的点云、位姿和时间戳数据
            data_group.current_cloud = data_group.cloud_buffer.front();
            data_group.current_pose = data_group.pose_buffer.front();
            data_group.current_time = data_group.time_buffer.front();
            // 清空所有缓冲区，只保留最新数据（丢弃中间帧以保持实时性）
            while (!data_group.cloud_buffer.empty())
            {
                data_group.cloud_buffer.pop();
                data_group.pose_buffer.pop();
                data_group.time_buffer.pop();
            }
        }

        // ==================== 第二部分：点云预处理 ====================
        // 将点云从局部坐标系变换到世界坐标系
        pcl::transformPointCloud(*data_group.current_cloud, *data_group.current_cloud, data_group.current_pose);

        // 对点云进行体素滤波下采样，减少点数以提高后续处理效率
        std_desc::voxelFilter(data_group.current_cloud, node_config.ds_size);

        // 初始化关键帧点云容器（如果为空）
        if (data_group.key_cloud == nullptr)
            data_group.key_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
        // 将当前帧点云累加到关键帧点云中（用于后续STD特征提取）
        *data_group.key_cloud += *data_group.current_cloud;

        // ==================== 第三部分：构建因子图 ====================
        // 获取当前帧的位姿变换矩阵
        Eigen::Affine3d pose = data_group.current_pose;
        // pose.linear() = data_group.current_pose.first;
        // pose.translation() = data_group.current_pose.second;

        // 将当前位姿作为初始值插入到iSAM2的初始估计中
        data_group.initial.insert(data_group.cloud_idx, gtsam::Pose3(pose.matrix()));

        if (!data_group.cloud_idx)
        {
            // 对于第一帧，添加先验因子作为全局锚点，固定起始位姿
            data_group.graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3(pose.matrix()), data_group.odometryNoise));
        }
        else
        {
            // 对于后续帧，添加里程计因子（相邻帧之间的相对位姿约束）
            auto prev_pose = gtsam::Pose3(data_group.origin_pose_vec[data_group.cloud_idx - 1].matrix());
            auto curr_pose = gtsam::Pose3(pose.matrix());
            // 计算前后帧之间的相对变换，并添加为BetweenFactor
            data_group.graph.add(gtsam::BetweenFactor<gtsam::Pose3>(data_group.cloud_idx - 1, data_group.cloud_idx,
                                                                    prev_pose.between(curr_pose), data_group.odometryNoise));
        }

        // 保存原始位姿（用于后续回环检测时计算相对变换）
        data_group.origin_pose_vec.push_back(pose);
        // 保存优化后的位姿（初始时与原始位姿相同）
        data_group.pose_vec.push_back(pose);

        // ==================== 第四部分：回环检测与约束添加 ====================
        // 每隔sub_frame_num帧进行一次回环检测（跳过第0帧）
        if (data_group.cloud_idx % node_config.sub_frame_num == 0 && data_group.cloud_idx != 0)
        {
            // if (data_group.cloud_idx == 660)
            // {
            //     pcl::PCDWriter writer;
            //     writer.write("/home/zhouzhou/temp/660_new.pcd", *data_group.key_cloud);
            // }

            // 从累积的关键帧点云中提取STD（Scan-To-Descriptor）特征
            std_desc::STDFeature feature = std_manager->extract(data_group.key_cloud);
            ROS_INFO("ID: %lu  FEATRUE SIZE: %lu CLOUD SIZE: %lu", data_group.cloud_idx, feature.descs.size(), data_group.key_cloud->size());
            std_desc::LoopResult result;

            int64_t duration;
            // 只有当帧索引超过skip_near_num时才进行回环搜索（避免与临近帧误匹配）
            if (data_group.cloud_idx > std_config.skip_near_num)
            {
                // 记录回环搜索的耗时
                auto start = std::chrono::high_resolution_clock::now();
                // 在历史特征库中搜索回环候选
                result = std_manager->searchLoop(feature);
                auto end = std::chrono::high_resolution_clock::now();
                duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            }

            // 将当前特征插入到特征管理器中，用于后续帧的回环检测
            std_manager->insert(feature);

            // 如果检测到有效的回环
            if (result.valid)
            {
                // 这里得到是新旧世界坐标系下的差值 T_old_new;
                ROS_WARN("FIND MATCHED LOOP! CURRENT_ID: %lu, LOOP_ID: %lu MATCH SCORE: %.4f, TIME COST: %lu ms", feature.id, result.match_id, result.match_score, duration);

                // 使用几何平面ICP验证回环结果并精化位姿变换
                double score = std_manager->verifyGeoPlaneICP(feature.cloud, std_manager->cloud_vec[result.match_id], result.rotation, result.translation);

                // 标记检测到回环
                data_group.has_loop_flag = true;
                // 保存回环约束对（历史帧ID, 当前帧ID）
                data_group.loop_container.emplace_back(result.match_id, feature.id);

                // 为子帧添加回环约束（关键帧由多个子帧组成）
                // 例如：sub_frame_num=10时，关键帧包含10个子帧
                for (size_t j = 1; j <= node_config.sub_frame_num; j++)
                {
                    // 计算当前关键帧中的子帧索引
                    int src_frame = data_group.cloud_idx + j - node_config.sub_frame_num;
                    // 计算历史匹配关键帧中对应的子帧索引
                    int tar_frame = result.match_id * node_config.sub_frame_num + j;

                    // 构造回环检测得到的位姿变换矩阵（从当前帧到历史帧的变换）
                    Eigen::Affine3d delta_pose = Eigen::Affine3d::Identity();
                    delta_pose.linear() = result.rotation;          // 旋转部分
                    delta_pose.translation() = result.translation;  // 平移部分

                    // 使用回环变换矫正当前子帧的位姿
                    Eigen::Affine3d refined_src = delta_pose * data_group.origin_pose_vec[src_frame];
                    // 获取历史目标子帧的位姿
                    Eigen::Affine3d tar_pose = data_group.origin_pose_vec[tar_frame];

                    // 添加回环因子（带有鲁棒噪声模型以抑制错误回环的影响）
                    data_group.graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                        tar_frame, src_frame,
                        gtsam::Pose3(tar_pose.matrix()).between(gtsam::Pose3(refined_src.matrix())),
                        data_group.robustLoopNoise));
                }
            }
            // 清空关键帧点云，准备下一个关键帧的累积
            data_group.key_cloud->clear();
        }

        // ==================== 第五部分：增量式位姿图优化 ====================
        // 使用iSAM2进行增量式优化，传入新增的因子和初始值
        data_group.isam->update(data_group.graph, data_group.initial);
        // 额外进行一次更新以提高收敛性
        data_group.isam->update();

        // 如果检测到回环，多次更新iSAM2以确保回环约束充分收敛
        if (data_group.has_loop_flag)
        {
            data_group.isam->update();
            data_group.isam->update();
            data_group.isam->update();
            data_group.isam->update();
            data_group.isam->update();
        }

        // 清空因子图和初始值容器，为下一次迭代准备
        data_group.graph.resize(0);
        data_group.initial.clear();

        // ==================== 第六部分：获取优化结果并更新位姿 ====================
        // 从iSAM2获取当前所有节点的优化估计值
        gtsam::Values curr_estimates = data_group.isam->calculateEstimate();

        // 确保优化结果数量与位姿向量长度一致
        assert(curr_estimates.size() == data_group.pose_vec.size());

        // 将优化后的位姿更新到pose_vec中
        for (int i = 0; i < curr_estimates.size(); i++)
        {
            gtsam::Pose3 est = curr_estimates.at<gtsam::Pose3>(i);
            Eigen::Affine3d est_affine3d(est.matrix());
            data_group.pose_vec[i] = est_affine3d;
        }

        // ==================== 第七部分：发布结果 ====================
        // 获取最后一帧的优化位姿
        Eigen::Affine3d last_pose = data_group.pose_vec.back();

        // 计算优化后位姿与原始位姿之间的差异变换（用于TF广播）
        // frame_delta_pose = T_optimized * T_original^(-1)
        Eigen::Affine3d frame_delta_pose = last_pose * data_group.origin_pose_vec.back().inverse();

        // 广播TF变换，将局部坐标系与地图坐标系连接起来
        br.sendTransform(eigen2Transform(frame_delta_pose.linear(), frame_delta_pose.translation(), node_config.map_frame, node_config.local_frame, data_group.current_time));

        // 发布原始里程计路径（未优化）
        publishPath(path_pub, data_group.origin_pose_vec);

        // 发布优化后的路径
        publishPath(correct_path_pub, data_group.pose_vec);

        // 发布回环约束可视化标记
        publishLoopConstraints();

        // 帧索引递增
        data_group.cloud_idx++;

        // 重置回环标志位
        data_group.has_loop_flag = false;
    }

public:
    ros::NodeHandle nh;
    Config node_config;
    tf2_ros::TransformBroadcaster br;
    ros::Timer main_loop;
    ros::Subscriber odom_cloud_sub;
    DataGroup data_group;
    std_desc::Config std_config;
    std::shared_ptr<std_desc::STDManager> std_manager;

    ros::Publisher correct_path_pub;
    ros::Publisher path_pub;
    ros::Publisher loop_contraints_pub;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pgo_node");
    PGONode pgo_node;
    ros::spin();
    return 0;
}