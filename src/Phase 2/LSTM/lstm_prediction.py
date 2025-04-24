import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os
import joblib

# 设置输出目录
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/1536data result/Phase 2/LSTM'
os.makedirs(output_dir, exist_ok=True)

# 输入文件
input_file = '/Users/shaoxinyi/Downloads/FYP2/Data_collected/1536_CompassGyroSumHeadingData.txt'

# 加载数据
print(f"正在加载数据: {input_file}")
try:
    data = pd.read_csv(input_file, delimiter=';')
    print(f"成功加载数据")
    
    # 检查数据帧是否有标题行
    if 'Type' not in data.columns:
        column_names = [
            'Timestamp_(ms)', 'Type', 'step', 'value_1', 'value_2', 'value_3', 
            'GroundTruth', 'value_4', 'value_5', 'turns'
        ]
        data = pd.read_csv(input_file, delimiter=';', names=column_names)
        
        if data.iloc[0]['Type'] == 'Type':
            data = data.iloc[1:].reset_index(drop=True)
except Exception as e:
    print(f"加载数据失败: {e}")
    exit(1)

# 数据预处理
ground_truth_location_data = data[data['Type'] == 'Ground_truth_Location'].copy()
initial_location_data = data[data['Type'] == 'Initial_Location'].copy()

# 创建Ground Truth数据框
if len(ground_truth_location_data) > 0 and len(initial_location_data) > 0:
    df_gt = pd.concat([initial_location_data, ground_truth_location_data], ignore_index=True)
    df_gt.sort_values(by='Timestamp_(ms)', inplace=True)
    df_gt.reset_index(drop=True, inplace=True)
else:
    print("缺少Ground Truth或初始位置数据")
    exit(1)

# 计算Ground Truth航向
def calculate_bearing(lat1, lon1, lat2, lon2):
    from math import atan2, degrees, radians, sin, cos
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    delta_lon = lon2 - lon1
    x = atan2(
        sin(delta_lon) * cos(lat2),
        cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(delta_lon)
    )
    bearing = (degrees(x) + 360) % 360
    return bearing

# 添加Ground Truth航向列
df_gt["GroundTruthHeadingComputed"] = np.nan
for i in range(1, len(df_gt)):
    df_gt.loc[i, "GroundTruthHeadingComputed"] = calculate_bearing(
        df_gt.loc[i-1, "value_5"], df_gt.loc[i-1, "value_4"],
        df_gt.loc[i, "value_5"], df_gt.loc[i, "value_4"]
    )
if len(df_gt) > 1:
    df_gt.loc[0, "GroundTruthHeadingComputed"] = df_gt.loc[1, "GroundTruthHeadingComputed"]

# 确保数据按时间戳排序
data.sort_values(by="Timestamp_(ms)", inplace=True)
df_gt.sort_values(by="Timestamp_(ms)", inplace=True)

# 合并Ground Truth航向
data = data.merge(df_gt[["Timestamp_(ms)", "GroundTruthHeadingComputed"]], on="Timestamp_(ms)", how="left")
data["GroundTruthHeadingComputed"] = data["GroundTruthHeadingComputed"].bfill()

# 将数值列转换为浮点数
for col in ['value_1', 'value_2', 'value_3', 'GroundTruthHeadingComputed', 'GroundTruth', 'value_4', 'value_5', 'step']:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# 分离陀螺仪和罗盘数据
gyro_data = data[data['Type'] == 'Gyro'].reset_index(drop=True)
compass_data = data[data['Type'] == 'Compass'].reset_index(drop=True)

# 重命名列
compass_data.rename(columns={'value_1': 'Magnetic_Field_Magnitude'}, inplace=True)
compass_data.rename(columns={'value_2': 'gyroSumFromstart0'}, inplace=True)
compass_data.rename(columns={'value_3': 'compass'}, inplace=True)

gyro_data.rename(columns={'value_1': 'axisZAngle'}, inplace=True)
gyro_data.rename(columns={'value_2': 'gyroSumFromstart0'}, inplace=True)
gyro_data.rename(columns={'value_3': 'compass'}, inplace=True)

# 计算传统方法的陀螺仪航向
first_ground_truth = initial_location_data['GroundTruth'].iloc[0] if len(initial_location_data) > 0 else 0
compass_data['GyroStartByGroundTruth'] = first_ground_truth + compass_data['gyroSumFromstart0'] - compass_data['gyroSumFromstart0'].iloc[0]
compass_data['GyroStartByGroundTruth'] = (compass_data['GyroStartByGroundTruth'] + 360) % 360
gyro_data['GyroStartByGroundTruth'] = first_ground_truth + gyro_data['gyroSumFromstart0'] - gyro_data['gyroSumFromstart0'].iloc[0]
gyro_data['GyroStartByGroundTruth'] = (gyro_data['GyroStartByGroundTruth'] + 360) % 360

print("数据预处理完成")

# 保存处理后的数据
gyro_data.to_csv(os.path.join(output_dir, 'processed_gyro_data.csv'), index=False)
compass_data.to_csv(os.path.join(output_dir, 'processed_compass_data.csv'), index=False)
print(f"处理后的数据保存至 {output_dir}")

# 提取特征
gyro_features = gyro_data[['axisZAngle', 'gyroSumFromstart0']].values
compass_features = compass_data[['Magnetic_Field_Magnitude', 'gyroSumFromstart0']].values

# 目标变量
gyro_target = gyro_data['GroundTruthHeadingComputed'].values.reshape(-1, 1)
compass_target = compass_data['GroundTruthHeadingComputed'].values.reshape(-1, 1)

# 加载已训练好的模型和缩放器
print("正在加载已训练好的LSTM模型和缩放器...")
try:
    # 模型文件路径
    gyro_model_path = os.path.join(output_dir, 'gyro_heading_lstm_model.keras')
    compass_model_path = os.path.join(output_dir, 'compass_heading_lstm_model.keras')
    
    # 缩放器文件路径
    gyro_scaler_X_path = os.path.join(output_dir, 'gyro_scaler_X.pkl')
    gyro_scaler_y_path = os.path.join(output_dir, 'gyro_scaler_y.pkl')
    compass_scaler_X_path = os.path.join(output_dir, 'compass_scaler_X.pkl')
    compass_scaler_y_path = os.path.join(output_dir, 'compass_scaler_y.pkl')
    
    # 配置文件路径
    config_path = os.path.join(output_dir, 'lstm_config.txt')
    
    # 加载模型
    if os.path.exists(gyro_model_path) and os.path.exists(compass_model_path):
        gyro_model = tf.keras.models.load_model(gyro_model_path)
        compass_model = tf.keras.models.load_model(compass_model_path)
        print("模型加载成功")
    else:
        print(f"模型文件不存在: {gyro_model_path} 或 {compass_model_path}")
        exit(1)
    
    # 加载缩放器
    if (os.path.exists(gyro_scaler_X_path) and os.path.exists(gyro_scaler_y_path) and
        os.path.exists(compass_scaler_X_path) and os.path.exists(compass_scaler_y_path)):
        gyro_scaler_X = joblib.load(gyro_scaler_X_path)
        gyro_scaler_y = joblib.load(gyro_scaler_y_path)
        compass_scaler_X = joblib.load(compass_scaler_X_path)
        compass_scaler_y = joblib.load(compass_scaler_y_path)
        print("缩放器加载成功")
    else:
        print("缩放器文件不存在，将创建新的缩放器")
        # 创建并拟合缩放器
        gyro_scaler_X = MinMaxScaler()
        gyro_scaler_y = MinMaxScaler()
        compass_scaler_X = MinMaxScaler()
        compass_scaler_y = MinMaxScaler()
        
        # 拟合缩放器
        gyro_features_scaled = gyro_scaler_X.fit_transform(gyro_features)
        gyro_target_scaled = gyro_scaler_y.fit_transform(gyro_target)
        compass_features_scaled = compass_scaler_X.fit_transform(compass_features)
        compass_target_scaled = compass_scaler_y.fit_transform(compass_target)
    
    # 加载窗口大小
    window_size = 20  # 默认值
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            for line in f:
                if line.startswith('window_size='):
                    window_size = int(line.split('=')[1])
                    break
        print(f"从配置文件加载窗口大小: {window_size}")
    else:
        print(f"使用默认窗口大小: {window_size}")
        
except Exception as e:
    print(f"加载模型或缩放器失败: {e}")
    exit(1)

# 使用加载的缩放器转换特征
gyro_features_scaled = gyro_scaler_X.transform(gyro_features)
compass_features_scaled = compass_scaler_X.transform(compass_features)

# 创建序列数据函数
def create_sequences(X, window_size):
    X_seq = []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i + window_size])
    return np.array(X_seq)

# 创建序列
gyro_X_seq = create_sequences(gyro_features_scaled, window_size)
compass_X_seq = create_sequences(compass_features_scaled, window_size)

# 预测
print("使用LSTM模型预测航向...")
gyro_pred_scaled = gyro_model.predict(gyro_X_seq)
compass_pred_scaled = compass_model.predict(compass_X_seq)

# 反向转换预测结果
gyro_pred = gyro_scaler_y.inverse_transform(gyro_pred_scaled)
compass_pred = compass_scaler_y.inverse_transform(compass_pred_scaled)

# 处理预测结果
gyro_predictions = np.zeros(len(gyro_data))
compass_predictions = np.zeros(len(compass_data))

# 填充预测值，前window_size个点为0
gyro_predictions[window_size:window_size+len(gyro_pred)] = gyro_pred.flatten()
compass_predictions[window_size:window_size+len(compass_pred)] = compass_pred.flatten()

# 标准化到0-360度
gyro_predictions = (gyro_predictions + 360) % 360
compass_predictions = (compass_predictions + 360) % 360

# 将预测结果添加到数据中
gyro_data['ML_Predicted_Heading'] = gyro_predictions
compass_data['ML_Predicted_Heading'] = compass_predictions

# 保存带有预测结果的数据
gyro_data.to_csv(os.path.join(output_dir, 'gyro_data_with_predictions.csv'), index=False)
compass_data.to_csv(os.path.join(output_dir, 'compass_data_with_predictions.csv'), index=False)
print(f"带有预测结果的数据保存至 {output_dir}")

print("LSTM预测完成")