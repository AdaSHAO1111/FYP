import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import os
import time
from heading_prediction_model import HeadingPredictor

# 设置输出目录
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2'
os.makedirs(output_dir, exist_ok=True)

# 输入文件
input_file = '/Users/shaoxinyi/Downloads/FYP2/Data_collected/1536_CompassGyroSumHeadingData.txt'

# 数据加载与处理部分
print(f"正在从 {input_file} 加载数据...")
try:
    # 尝试以分号分隔的CSV格式读取
    data = pd.read_csv(input_file, delimiter=';')
    print(f"成功加载文件")
    
    # 检查数据帧是否有标题行
    if 'Type' not in data.columns:
        # 文件没有标题，尝试推断它们
        column_names = [
            'Timestamp_(ms)', 'Type', 'step', 'value_1', 'value_2', 'value_3', 
            'GroundTruth', 'value_4', 'value_5', 'turns'
        ]
        
        # 再次尝试使用列名
        data = pd.read_csv(input_file, delimiter=';', names=column_names)
        
        # 如果第一行包含标题值，则删除它
        if data.iloc[0]['Type'] == 'Type':
            data = data.iloc[1:].reset_index(drop=True)
except Exception as e:
    print(f"加载数据时出错: {e}")
    exit(1)

# 提取Ground Truth位置数据
ground_truth_location_data = data[data['Type'] == 'Ground_truth_Location'].copy()
initial_location_data = data[data['Type'] == 'Initial_Location'].copy()

# 创建单独的数据框来存储Ground Truth航向
if len(ground_truth_location_data) > 0 and len(initial_location_data) > 0:
    df_gt = pd.concat([initial_location_data, ground_truth_location_data], ignore_index=True)
    df_gt.sort_values(by='Timestamp_(ms)', inplace=True)
    df_gt.reset_index(drop=True, inplace=True)
else:
    print("缺少Ground Truth或初始位置数据")
    exit(1)

# 计算Ground Truth航向
def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    计算两点之间的方位（方位角）
    """
    from math import atan2, degrees, radians, sin, cos
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    delta_lon = lon2 - lon1
    x = atan2(
        sin(delta_lon) * cos(lat2),
        cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(delta_lon)
    )
    
    bearing = (degrees(x) + 360) % 360  # 标准化为0-360度
    return bearing

# 添加Ground Truth航向列
df_gt["GroundTruthHeadingComputed"] = np.nan

# 计算连续点之间的航向
for i in range(1, len(df_gt)):
    df_gt.loc[i, "GroundTruthHeadingComputed"] = calculate_bearing(
        df_gt.loc[i-1, "value_5"], df_gt.loc[i-1, "value_4"],
        df_gt.loc[i, "value_5"], df_gt.loc[i, "value_4"]
    )

# 用第二个条目的航向填充第一个条目
if len(df_gt) > 1:
    df_gt.loc[0, "GroundTruthHeadingComputed"] = df_gt.loc[1, "GroundTruthHeadingComputed"]

# 确保数据和df_gt按时间戳排序
data.sort_values(by="Timestamp_(ms)", inplace=True)
df_gt.sort_values(by="Timestamp_(ms)", inplace=True)

# 使用向后填充来传播GroundTruthHeadingComputed值
data = data.merge(df_gt[["Timestamp_(ms)", "GroundTruthHeadingComputed"]], on="Timestamp_(ms)", how="left")
data["GroundTruthHeadingComputed"] = data["GroundTruthHeadingComputed"].fillna(method="bfill")

# 将数值列转换为浮点数
for col in ['value_1', 'value_2', 'value_3', 'GroundTruthHeadingComputed']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# 分离陀螺仪和罗盘数据
gyro_data = data[data['Type'] == 'Gyro'].reset_index(drop=True)
compass_data = data[data['Type'] == 'Compass'].reset_index(drop=True)

# 重命名列以提高清晰度
compass_data.rename(columns={'value_1': 'Magnetic_Field_Magnitude'}, inplace=True)
compass_data.rename(columns={'value_2': 'gyroSumFromstart0'}, inplace=True)
compass_data.rename(columns={'value_3': 'compass'}, inplace=True)

gyro_data.rename(columns={'value_1': 'axisZAngle'}, inplace=True)
gyro_data.rename(columns={'value_2': 'gyroSumFromstart0'}, inplace=True)
gyro_data.rename(columns={'value_3': 'compass'}, inplace=True)

# 获取初始Ground Truth航向
first_ground_truth = initial_location_data['GroundTruth'].iloc[0] if len(initial_location_data) > 0 else 0

# 计算从Ground Truth开始的陀螺仪航向
compass_data['GyroStartByGroundTruth'] = first_ground_truth + compass_data['gyroSumFromstart0'] - compass_data['gyroSumFromstart0'].iloc[0]
compass_data['GyroStartByGroundTruth'] = (compass_data['GyroStartByGroundTruth'] + 360) % 360

gyro_data['GyroStartByGroundTruth'] = first_ground_truth + gyro_data['gyroSumFromstart0'] - gyro_data['gyroSumFromstart0'].iloc[0]
gyro_data['GyroStartByGroundTruth'] = (gyro_data['GyroStartByGroundTruth'] + 360) % 360

print("数据预处理完成")

# 训练机器学习模型
print("初始化深度学习模型...")
heading_predictor = HeadingPredictor(window_size=20)  # 使用20个时间步作为窗口大小

print("训练陀螺仪航向预测模型...")
start_time = time.time()
gyro_history = heading_predictor.train_gyro_model(
    gyro_data, 
    gyro_data['GroundTruthHeadingComputed'],
    epochs=30,
    batch_size=32,
    validation_split=0.2
)
gyro_train_time = time.time() - start_time
print(f"陀螺仪模型训练完成，用时 {gyro_train_time:.2f} 秒")

print("训练罗盘航向预测模型...")
start_time = time.time()
compass_history = heading_predictor.train_compass_model(
    compass_data, 
    compass_data['GroundTruthHeadingComputed'],
    epochs=30,
    batch_size=32,
    validation_split=0.2
)
compass_train_time = time.time() - start_time
print(f"罗盘模型训练完成，用时 {compass_train_time:.2f} 秒")

# 保存模型
heading_predictor.save_models(output_dir)

# 使用模型进行预测
print("使用深度学习模型预测航向...")
start_time = time.time()
gyro_predictions = heading_predictor.predict_gyro_heading(gyro_data)
compass_predictions = heading_predictor.predict_compass_heading(compass_data)
predict_time = time.time() - start_time
print(f"预测完成，用时 {predict_time:.2f} 秒")

# 为预测结果创建新列
gyro_data['ML_Predicted_Heading'] = gyro_predictions
compass_data['ML_Predicted_Heading'] = compass_predictions

# 评估模型性能
print("评估模型性能...")
metrics = heading_predictor.evaluate_models(
    gyro_data, 
    compass_data,
    gyro_data['GroundTruthHeadingComputed'],
    compass_data['GroundTruthHeadingComputed']
)

# 打印评估指标
print("\n模型性能评估：")
print(f"陀螺仪 - 传统方法 MAE: {metrics['gyro_traditional_mae']:.2f}°, RMSE: {metrics['gyro_traditional_rmse']:.2f}°")
print(f"陀螺仪 - 深度学习 MAE: {metrics['gyro_ml_mae']:.2f}°, RMSE: {metrics['gyro_ml_rmse']:.2f}°")
print(f"罗盘 - 传统方法 MAE: {metrics['compass_traditional_mae']:.2f}°, RMSE: {metrics['compass_traditional_rmse']:.2f}°")
print(f"罗盘 - 深度学习 MAE: {metrics['compass_ml_mae']:.2f}°, RMSE: {metrics['compass_ml_rmse']:.2f}°")

# 将评估指标保存到文件
metrics_df = pd.DataFrame({
    'Method': ['Gyro - Traditional', 'Gyro - LSTM', 'Compass - Traditional', 'Compass - LSTM'],
    'MAE': [
        metrics['gyro_traditional_mae'], 
        metrics['gyro_ml_mae'],
        metrics['compass_traditional_mae'],
        metrics['compass_ml_mae']
    ],
    'RMSE': [
        metrics['gyro_traditional_rmse'],
        metrics['gyro_ml_rmse'],
        metrics['compass_traditional_rmse'],
        metrics['compass_ml_rmse']
    ]
})
metrics_df.to_csv(os.path.join(output_dir, 'heading_prediction_metrics.csv'), index=False)
print(f"评估指标已保存到 {os.path.join(output_dir, 'heading_prediction_metrics.csv')}")

# 设置绘图参数，符合IEEE格式
fontSizeAll = 5  # 保持小字体大小
plt.rcParams.update({
    'xtick.major.pad': '1',
    'ytick.major.pad': '1',
    'legend.fontsize': fontSizeAll,
    'legend.handlelength': 2,
    'font.size': fontSizeAll,
    'axes.linewidth': 0.2,
    'patch.linewidth': 0.2,
    'font.family': "Times New Roman"
})

# 格式化时间戳以便更好地可读性
# 将时间戳转换为从开始的相对时间（秒）
min_timestamp = min(gyro_data['Timestamp_(ms)'].min(), compass_data['Timestamp_(ms)'].min())
gyro_data['Time_relative'] = (gyro_data['Timestamp_(ms)'] - min_timestamp) / 1000  # 转换为秒
compass_data['Time_relative'] = (compass_data['Timestamp_(ms)'] - min_timestamp) / 1000  # 转换为秒

# 1. 绘制Ground Truth Heading与Gyro Heading的比较（传统方法 vs ML方法）
fig1, ax1 = plt.subplots(figsize=(5, 3), dpi=300)  # 稍微降低高度
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.92)  # 为标签留出更多空间

# 绘制Ground Truth航向
plt.plot(gyro_data["Time_relative"], gyro_data["GroundTruthHeadingComputed"], 
         color='red', linestyle='-', linewidth=1.2, label='Ground Truth')

# 绘制传统方法的陀螺仪航向
plt.plot(gyro_data["Time_relative"], gyro_data["GyroStartByGroundTruth"], 
         color='blue', linestyle='--', linewidth=0.8, label='Traditional')

# 绘制ML方法的陀螺仪航向
plt.plot(gyro_data["Time_relative"], gyro_data["ML_Predicted_Heading"], 
         color='green', linestyle='-', linewidth=0.8, label='LSTM')

# 坐标轴格式化
ax1.yaxis.set_major_locator(MultipleLocator(45))  # Y轴主刻度间隔：45度
ax1.yaxis.set_minor_locator(MultipleLocator(15))  # Y轴次刻度间隔：15度

# 使用较少的刻度格式化X轴
max_time = max(gyro_data['Time_relative'].max(), compass_data['Time_relative'].max())
ax1.set_xlim(0, max_time)
ax1.xaxis.set_major_locator(plt.MaxNLocator(8))  # 减少到8个刻度，减少拥挤

# 旋转X轴标签以提高可读性
plt.xticks(rotation=30, ha='right')

# 坐标轴标签
plt.xlabel("Time (s)", labelpad=5)
plt.ylabel("Heading (Degrees)", labelpad=5)

# 刻度和网格
plt.tick_params(axis='both', bottom=True, top=True, left=True, right=True, direction='in', 
               which='major', grid_color='blue', width=0.3, length=2.5)
plt.tick_params(axis='both', bottom=True, top=True, left=True, right=True, direction='in', 
               which='minor', grid_color='blue', width=0.15, length=1)

# 自定义图例
legend_elements = [
    Line2D([0], [0], color='red', linestyle='-', linewidth=1.2, label='Ground Truth'),
    Line2D([0], [0], color='blue', linestyle='--', linewidth=0.8, label='Traditional'),
    Line2D([0], [0], color='green', linestyle='-', linewidth=0.8, label='LSTM')
]

plt.legend(handles=legend_elements, loc='best')

# 网格
plt.grid(linestyle=':', linewidth=0.5, alpha=0.15, color='k')

# 添加标题
plt.title("Ground Truth vs Gyro Heading Methods Comparison", fontsize=fontSizeAll+1)

# 添加MAE和RMSE信息文本框
textstr = '\n'.join((
    f"Traditional: MAE={metrics['gyro_traditional_mae']:.1f}°, RMSE={metrics['gyro_traditional_rmse']:.1f}°",
    f"LSTM: MAE={metrics['gyro_ml_mae']:.1f}°, RMSE={metrics['gyro_ml_rmse']:.1f}°"
))
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax1.text(0.05, 0.05, textstr, transform=ax1.transAxes, fontsize=fontSizeAll-1,
        verticalalignment='bottom', bbox=props)

# 保存图表
gyro_plot_file = os.path.join(output_dir, 'gyro_heading_methods_comparison.png')
plt.savefig(gyro_plot_file, bbox_inches='tight')
print(f"已保存陀螺仪航向方法比较图至 {gyro_plot_file}")

# 2. 绘制Ground Truth Heading与Compass Heading的比较（传统方法 vs ML方法）
fig2, ax2 = plt.subplots(figsize=(5, 3), dpi=300)  # 稍微降低高度
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.92)  # 为标签留出更多空间

# 绘制Ground Truth航向
plt.plot(compass_data["Time_relative"], compass_data["GroundTruthHeadingComputed"], 
         color='red', linestyle='-', linewidth=1.2, label='Ground Truth')

# 绘制传统方法的罗盘航向
plt.plot(compass_data["Time_relative"], compass_data["compass"], 
         color='blue', linestyle='--', linewidth=0.8, label='Traditional')

# 绘制ML方法的罗盘航向
plt.plot(compass_data["Time_relative"], compass_data["ML_Predicted_Heading"], 
         color='green', linestyle='-', linewidth=0.8, label='LSTM')

# 坐标轴格式化
ax2.yaxis.set_major_locator(MultipleLocator(45))  # Y轴主刻度间隔：45度
ax2.yaxis.set_minor_locator(MultipleLocator(15))  # Y轴次刻度间隔：15度

# 使用较少的刻度格式化X轴
ax2.set_xlim(0, max_time)
ax2.xaxis.set_major_locator(plt.MaxNLocator(8))  # 减少到8个刻度，减少拥挤

# 旋转X轴标签以提高可读性
plt.xticks(rotation=30, ha='right')

# 坐标轴标签
plt.xlabel("Time (s)", labelpad=5)
plt.ylabel("Heading (Degrees)", labelpad=5)

# 刻度和网格
plt.tick_params(axis='both', bottom=True, top=True, left=True, right=True, direction='in', 
               which='major', grid_color='blue', width=0.3, length=2.5)
plt.tick_params(axis='both', bottom=True, top=True, left=True, right=True, direction='in', 
               which='minor', grid_color='blue', width=0.15, length=1)

# 自定义图例
legend_elements = [
    Line2D([0], [0], color='red', linestyle='-', linewidth=1.2, label='Ground Truth'),
    Line2D([0], [0], color='blue', linestyle='--', linewidth=0.8, label='Traditional'),
    Line2D([0], [0], color='green', linestyle='-', linewidth=0.8, label='LSTM')
]

plt.legend(handles=legend_elements, loc='best')

# 网格
plt.grid(linestyle=':', linewidth=0.5, alpha=0.15, color='k')

# 添加标题
plt.title("Ground Truth vs Compass Heading Methods Comparison", fontsize=fontSizeAll+1)

# 添加MAE和RMSE信息文本框
textstr = '\n'.join((
    f"Traditional: MAE={metrics['compass_traditional_mae']:.1f}°, RMSE={metrics['compass_traditional_rmse']:.1f}°",
    f"LSTM: MAE={metrics['compass_ml_mae']:.1f}°, RMSE={metrics['compass_ml_rmse']:.1f}°"
))
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax2.text(0.05, 0.05, textstr, transform=ax2.transAxes, fontsize=fontSizeAll-1,
        verticalalignment='bottom', bbox=props)

# 保存图表
compass_plot_file = os.path.join(output_dir, 'compass_heading_methods_comparison.png')
plt.savefig(compass_plot_file, bbox_inches='tight')
print(f"已保存罗盘航向方法比较图至 {compass_plot_file}")

# 3. 绘制训练损失和验证损失曲线
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(6, 2.5), dpi=300)  # 两列图表
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, wspace=0.3)  # 调整间距

# 陀螺仪模型训练历史
ax3a.plot(gyro_history.history['loss'], color='blue', linestyle='-', linewidth=1, label='Train')
ax3a.plot(gyro_history.history['val_loss'], color='red', linestyle='--', linewidth=1, label='Validation')
ax3a.set_title('Gyro LSTM Training History', fontsize=fontSizeAll+1)
ax3a.set_xlabel('Epoch', labelpad=3)
ax3a.set_ylabel('Loss (MSE)', labelpad=4)
ax3a.tick_params(axis='both', which='major', labelsize=fontSizeAll-1)
ax3a.grid(linestyle=':', linewidth=0.5, alpha=0.3)
ax3a.legend(fontsize=fontSizeAll-1)

# 罗盘模型训练历史
ax3b.plot(compass_history.history['loss'], color='blue', linestyle='-', linewidth=1, label='Train')
ax3b.plot(compass_history.history['val_loss'], color='red', linestyle='--', linewidth=1, label='Validation')
ax3b.set_title('Compass LSTM Training History', fontsize=fontSizeAll+1)
ax3b.set_xlabel('Epoch', labelpad=3)
ax3b.set_ylabel('Loss (MSE)', labelpad=4)
ax3b.tick_params(axis='both', which='major', labelsize=fontSizeAll-1)
ax3b.grid(linestyle=':', linewidth=0.5, alpha=0.3)
ax3b.legend(fontsize=fontSizeAll-1)

# 保存图表
history_plot_file = os.path.join(output_dir, 'lstm_training_history.png')
plt.savefig(history_plot_file, bbox_inches='tight')
print(f"已保存训练历史图至 {history_plot_file}")

print("所有处理和绘图任务完成") 