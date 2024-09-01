#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

#include <Eigen/Dense>
#include <vector>

// 定义Settings结构体
struct Settings {
    double altitude;
    double latitude;
    double g;
    double Ts;
    std::string path;
    double init_heading;
    std::vector<double> init_pos;
    std::string detector_type;
    double sigma_a;
    double sigma_g;
    int Window_size;
    double gamma;
    bool biases;
    bool scalefactors;
    std::vector<double> sigma_acc;
    std::vector<double> sigma_gyro;
    std::vector<double> acc_bias_driving_noise;
    std::vector<double> gyro_bias_driving_noise;
    std::vector<double> sigma_vel;
    std::vector<double> sigma_initial_pos;
    std::vector<double> sigma_initial_vel;
    std::vector<double> sigma_initial_att;
    std::vector<double> sigma_initial_acc_bias;
    std::vector<double> sigma_initial_gyro_bias;
    std::vector<double> sigma_initial_acc_scale;
    std::vector<double> sigma_initial_gyro_scale;
    double acc_bias_instability_time_constant_filter;
    double gyro_bias_instability_time_constant_filter;
};

// 声明全局变量
Settings simdata;


double gravity(double latitude, double altitude) {  // gravity函数，计算局部重力加速度
    double lambda = M_PI / 180.0 * latitude;
    double gamma = 9.780327 * (1 + 0.0053024 * std::sin(lambda) * std::sin(lambda) - 0.0000058 * std::sin(2 * lambda) * std::sin(2 * lambda));
    double g = gamma - ((3.0877e-6) - (0.004e-6) * std::sin(lambda) * std::sin(lambda)) * altitude + (0.072e-12) * altitude * altitude;
    return g;
}


#include <fstream>
#include <sstream>

std::vector<std::vector<double>> load_dataset() {
    std::ifstream infile(simdata.path + "data_inert.txt");
    std::string line;
    
    // 跳过文件头
    for (int i = 0; i < 32; ++i) {
        infile >> line;
    }

    std::vector<std::vector<double>> data_inert;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::vector<double> row(17);
        for (int i = 0; i < 17; ++i) {
            iss >> row[i];
        }
        data_inert.push_back(row);
    }
    infile.close();

    // 将数据缩放到SI单位并存储在矩阵中
    double imu_scalefactor = 9.80665; // From the Microstrain IMU data sheet
    std::vector<std::vector<double>> f_imu(3), omega_imu(3);
    for (size_t i = 0; i < data_inert.size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            f_imu[j].push_back(data_inert[i][j + 1] * imu_scalefactor);
            omega_imu[j].push_back(data_inert[i][j + 4]);
        }
    }

    // 将f_imu和omega_imu组合成一个u矩阵
    std::vector<std::vector<double>> u(6);
    for (int j = 0; j < 3; ++j) {
        u[j] = f_imu[j];
        u[j + 3] = omega_imu[j];
    }
    return u;
}



std::vector<std::vector<double>> settings() {
    // 经度
    simdata.altitude = 100;

    // 纬度
    simdata.latitude = 58;

    // 估算重力加速度
    simdata.g = gravity(simdata.latitude, simdata.altitude);

    // 采样周期 [s]
    simdata.Ts = 1.0 / 250;

    // 要处理的IMU数据文件所在文件夹的路径
    simdata.path = "Measurement_100521_2/";

    // 加载数据
    auto u = load_dataset();

    // 初始航向 [rad]
    simdata.init_heading = 0 * M_PI / 180;

    // 初始位置（注意列向量） (x,y,z)-axis [m]
    simdata.init_pos = {0, 0, 0};

    // Detector Settings
    simdata.detector_type = "GLRT";
    simdata.sigma_a = 0.01;  // 加速度计噪声的标准差 [m/s^2]
    simdata.sigma_g = 0.1 * M_PI / 180;  // 陀螺仪噪声的标准差 [rad/s]
    simdata.Window_size = 3;  // 零速检测窗口大小 [samples]
    simdata.gamma = 0.3e5;  // 零速检测阈值

    // FILTER PARAMETERS
    simdata.biases = true;  // 传感器偏置作为状态量
    simdata.scalefactors = true;  // 传感器缩放因子作为状态量
    simdata.sigma_acc = {0.5, 0.5, 0.5};
    simdata.sigma_gyro = {0.5 * M_PI / 180, 0.5 * M_PI / 180, 0.5 * M_PI / 180};
    simdata.acc_bias_driving_noise = {0.0000001, 0.0000001, 0.0000001};
    simdata.gyro_bias_driving_noise = {0.0000001 * M_PI / 180, 0.0000001 * M_PI / 180, 0.0000001 * M_PI / 180};
    simdata.sigma_vel = {0.01, 0.01, 0.01};
    simdata.sigma_initial_pos = {1e-5, 1e-5, 1e-5};
    simdata.sigma_initial_vel = {1e-5, 1e-5, 1e-5};
    simdata.sigma_initial_att = {0.1 * M_PI / 180, 0.1 * M_PI / 180, 0.1 * M_PI / 180};
    simdata.sigma_initial_acc_bias = {0.3, 0.3, 0.3};
    simdata.sigma_initial_gyro_bias = {0.3 * M_PI / 180, 0.3 * M_PI / 180, 0.3 * M_PI / 180};
    simdata.sigma_initial_acc_scale = {0.0001, 0.0001, 0.0001};
    simdata.sigma_initial_gyro_scale = {0.00001, 0.00001, 0.00001};
    simdata.acc_bias_instability_time_constant_filter = std::numeric_limits<double>::infinity();
    simdata.gyro_bias_instability_time_constant_filter = std::numeric_limits<double>::infinity();

    return u;
}


std::vector<double> GLRT(const std::vector<std::vector<double>>& u) {
    std::vector<double> T(u[0].size() - simdata.Window_size + 1, 0.0);

    for (size_t k = 0; k <= u[0].size() - simdata.Window_size; ++k) {
        std::vector<double> ya_m(3, 0.0);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < simdata.Window_size; ++j) {
                ya_m[i] += u[i][k + j];
            }
            ya_m[i] /= simdata.Window_size;
        }

        double norm_ya_m = std::sqrt(ya_m[0] * ya_m[0] + ya_m[1] * ya_m[1] + ya_m[2] * ya_m[2]);

        for (int l = k; l < k + simdata.Window_size; ++l) {
            for (int i = 0; i < 3; ++i) {
                double tmp = u[i][l] - simdata.g * ya_m[i] / norm_ya_m;
                T[k] += (u[i + 3][l] * u[i + 3][l]) / simdata.sigma_g + (tmp * tmp) / simdata.sigma_a;
            }
        }
        T[k] /= simdata.Window_size;
    }

    return T;
}

std::pair<std::vector<bool>, std::vector<double>> zero_velocity_detector(const std::vector<std::vector<double>>& u) {
    std::vector<bool> zupt(u[0].size(), false);
    std::vector<double> T;

    // 根据选择的检测类型调用相应的函数
    if (simdata.detector_type == "GLRT") {
        T = GLRT(u);
    } else {
        std::cerr << "The chosen detector type is not recognized. The GLRT detector is used by default." << std::endl;
        T = GLRT(u);
    }

    for (size_t i = 0; i < T.size(); ++i) {
        if (T[i] < simdata.gamma) {
            zupt[i] = true;
        }
    }

    return {zupt, T};
}




// 函数声明
void init_filter(Eigen::MatrixXd& P, Eigen::MatrixXd& Q, Eigen::MatrixXd& R, Eigen::MatrixXd& H);
void init_vec(int N, Eigen::MatrixXd& P, std::vector<Eigen::VectorXd>& x_h, Eigen::MatrixXd& cov, Eigen::MatrixXd& Id);
void init_Nav_eq(const Eigen::MatrixXd& u, Eigen::VectorXd& x_h, Eigen::VectorXd& quat);
Eigen::VectorXd comp_imu_errors(const Eigen::VectorXd& u, const Eigen::VectorXd& x_h);
void Navigation_equations(Eigen::VectorXd& x_h, const Eigen::VectorXd& u_h, Eigen::VectorXd& quat);
void state_matrix(const Eigen::VectorXd& quat, const Eigen::VectorXd& u_h, Eigen::MatrixXd& F, Eigen::MatrixXd& G);
void comp_internal_states(Eigen::VectorXd& x_h, const Eigen::VectorXd& dx, Eigen::VectorXd& quat);

std::pair<std::vector<Eigen::VectorXd>, Eigen::MatrixXd> ZUPTaidedINS(const Eigen::MatrixXd& u, const std::vector<bool>& zupt) {
    // 初始化数据融合
    int N = u.cols(); // 获取IMU数据向量的长度

    Eigen::MatrixXd P, Q, R, H;
    init_filter(P, Q, R, H); // 初始化滤波器矩阵

    Eigen::MatrixXd cov(N, P.rows());
    Eigen::MatrixXd Id = Eigen::MatrixXd::Identity(P.rows(), P.cols());
    std::vector<Eigen::VectorXd> x_h(N, Eigen::VectorXd::Zero(15)); // 状态向量
    Eigen::VectorXd quat(4);

    init_vec(N, P, x_h, cov, Id); // 初始化
    init_Nav_eq(u, x_h[0], quat);

    // 运行过滤算法
    for (int k = 1; k < N; ++k) {

        // 用当前估计的传感器误差补偿IMU测量值
        Eigen::VectorXd u_h = comp_imu_errors(u.col(k), x_h[k - 1]);

        // 更新导航方程
        Navigation_equations(x_h[k], u_h, quat);

        // 更新状态转移矩阵
        Eigen::MatrixXd F, G;
        state_matrix(quat, u_h, F, G);

        // 更新滤波器状态协方差矩阵P
        P = F * P * F.transpose() + G * Q * G.transpose();

        // 确保 P 矩阵是对称的
        P = (P + P.transpose()) / 2;

        // 存储状态协方差矩阵 P 的对角线元素
        cov.col(k) = P.diagonal();

        // 零速度更新
        if (zupt[k]) {
            // 计算卡尔曼滤波增益
            Eigen::MatrixXd K = P * H.transpose() * (H * P * H.transpose() + R).inverse();

            // 计算预测误差（负的估计速度）
            Eigen::VectorXd z = -x_h[k].segment(3, 3); // 假设速度在第4到6位

            // 估计导航状态中的扰动
            Eigen::VectorXd dx = K * z;

            // 使用估计的扰动修正导航状态
            comp_internal_states(x_h[k], dx, quat);

            // 更新滤波器状态协方差矩阵 P
            P = (Id - K * H) * P;

            // 确保 P 矩阵是对称的
            P = (P + P.transpose()) / 2;

            // 存储状态协方差矩阵 P 的对角线元素
            cov.col(k) = P.diagonal();
        }
    }

    return {x_h, cov};
}




int main() {
    auto u = settings();
    auto [zupt, T] = zero_velocity_detector(u);
    
    // 输出检测结果
    for (size_t i = 0; i < zupt.size(); ++i) {
        std::cout << "Sample " << i << ": ZUPT = " << zupt[i] << ", T = " << T[i] << std::endl;
    }

    return 0;
}