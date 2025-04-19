import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import os

# 设置输出目录
output_dir = '/Users/shaoxinyi/Downloads/FYP2/Output/Phase 2'
os.makedirs(output_dir, exist_ok=True)

# 输入文件
input_file = '/Users/shaoxinyi/Downloads/FYP2/Data_collected/1536_CompassGyroSumHeadingData.txt'

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
data["GroundTruthHeadingComputed"] = data["GroundTruthHeadingComputed"].bfill()  # 使用bfill代替fillna(method="bfill")

# 将数值列转换为浮点数
for col in ['value_1', 'value_2', 'value_3', 'GroundTruthHeadingComputed', 'GroundTruth', 'value_4', 'value_5', 'step']:
    if col in data.columns:
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

# 计算位置
print("计算位置轨迹...")

# 设置步长和初始位置
step_length = 0.66  # 每步长度（米）
initial_position = (0, 0)  # 初始位置坐标 (x, y)

# 根据地面真实数据和初始位置，计算真实轨迹
def calculate_gt_positions(df_gt, initial_position=(0, 0)):
    # 确保地面真实数据按步骤排序
    df_gt = df_gt.sort_values(by='step')
    
    # 从dataframe中提取位置数据
    gt_positions = []
    
    # 如果数据中有坐标，直接使用
    if 'value_4' in df_gt.columns and 'value_5' in df_gt.columns:
        # 确保坐标已转换为数值类型
        df_gt['value_4'] = pd.to_numeric(df_gt['value_4'], errors='coerce')
        df_gt['value_5'] = pd.to_numeric(df_gt['value_5'], errors='coerce')
        
        # 将第一个Ground Truth点作为原点
        origin_x = df_gt['value_4'].iloc[0]
        origin_y = df_gt['value_5'].iloc[0]
        
        # 提取相对于原点的坐标
        for i in range(len(df_gt)):
            x = df_gt['value_4'].iloc[i] - origin_x
            y = df_gt['value_5'].iloc[i] - origin_y
            gt_positions.append((x, y))
    
    return gt_positions

# 根据步数和航向计算位置
def calculate_positions(data, heading_column, step_length=0.66, initial_position=(0, 0)):
    positions = [initial_position]  # 从初始位置开始
    current_position = initial_position
    prev_step = data['step'].iloc[0]
    
    for i in range(1, len(data)):
        # 计算步数变化
        change_in_step = data['step'].iloc[i] - prev_step
        
        # 如果步数变化，计算新位置
        if change_in_step != 0:
            # 计算距离变化
            change_in_distance = change_in_step * step_length
            
            # 获取heading值（注意航向角计算：0度为北，90度为东）
            heading = data[heading_column].iloc[i]
            
            # 计算新位置（东向为x轴，北向为y轴）
            new_x = current_position[0] + change_in_distance * np.sin(np.radians(heading))
            new_y = current_position[1] + change_in_distance * np.cos(np.radians(heading))
            
            # 更新当前位置
            current_position = (new_x, new_y)
            positions.append(current_position)
            
            # 更新前一步的步数
            prev_step = data['step'].iloc[i]
    
    return positions

# 获取Ground Truth位置
gt_positions = calculate_gt_positions(df_gt)
print(f"Ground Truth坐标点数量: {len(gt_positions)}")

# 确保按照正确的顺序连接ground truth点
# 按照df_gt中的顺序提取坐标，确保轨迹的连续性
gt_positions_ordered = []
for i in range(len(df_gt)):
    x = df_gt['value_4'].iloc[i] - df_gt['value_4'].iloc[0]
    y = df_gt['value_5'].iloc[i] - df_gt['value_5'].iloc[0]
    gt_positions_ordered.append((x, y))

# 使用传统方法计算位置
print("计算传统方法的位置...")
positions_compass_trad = calculate_positions(compass_data, 'compass', step_length, initial_position)
positions_gyro_trad = calculate_positions(compass_data, 'GyroStartByGroundTruth', step_length, initial_position)
print(f"Compass轨迹点数量: {len(positions_compass_trad)}")
print(f"Gyro轨迹点数量: {len(positions_gyro_trad)}")

# 提取坐标
x_gt = [pos[0] for pos in gt_positions_ordered]  # 使用有序的ground truth点
y_gt = [pos[1] for pos in gt_positions_ordered]  # 使用有序的ground truth点

x_compass_trad = [pos[0] for pos in positions_compass_trad]
y_compass_trad = [pos[1] for pos in positions_compass_trad]

x_gyro_trad = [pos[0] for pos in positions_gyro_trad]
y_gyro_trad = [pos[1] for pos in positions_gyro_trad]

# 可视化位置轨迹
print("生成位置轨迹对比图...")

# 设置绘图参数
fontSizeAll = 12  # 增大字体大小以提高可读性
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

# 使用传统方法计算的位置轨迹图
fig1, ax1 = plt.subplots(figsize=(8, 8), dpi=300)  # 增大图像尺寸
plt.subplots_adjust(left=0.12, bottom=0.1, right=0.95, top=0.92)

# 绘制Ground Truth轨迹 - 确保按照正确的顺序连接点
plt.plot(x_gt, y_gt, color='blue', linestyle='-', linewidth=1.5, label='Ground Truth')

# 绘制传统方法的罗盘轨迹
plt.plot(x_compass_trad, y_compass_trad, color='green', linestyle='--', linewidth=1, label='Compass (Traditional)')

# 绘制传统方法的陀螺仪轨迹
plt.plot(x_gyro_trad, y_gyro_trad, color='red', linestyle='-.', linewidth=1, label='Gyro (Traditional)')

# 标记起点和终点
plt.scatter(x_gt[0], y_gt[0], color='black', marker='o', s=80, label='Start')
plt.scatter(x_gt[-1], y_gt[-1], color='black', marker='x', s=80, label='End')

# 坐标轴格式化
ax1.set_aspect('equal')
ax1.set_xlabel('East (m)', labelpad=5)
ax1.set_ylabel('North (m)', labelpad=5)
ax1.grid(True, linestyle=':', alpha=0.5)

# 添加标题
plt.title('Position Trajectories Comparison (Traditional Methods)', fontsize=fontSizeAll+2)

# 添加图例
plt.legend(loc='best')

# 保存图像
trad_position_file = os.path.join(output_dir, 'position_comparison.png')
plt.savefig(trad_position_file, bbox_inches='tight')
print(f"位置对比图保存至: {trad_position_file}")

# 计算位置误差
print("计算位置误差...")

# 计算位置误差函数
def calculate_position_error(positions, gt_positions):
    """计算位置序列与Ground Truth之间的平均误差和累积误差"""
    # 确保两个位置序列长度相同，取最小长度进行比较
    min_length = min(len(positions), len(gt_positions))
    
    # 计算每个点的位置误差
    errors = []
    for i in range(min_length):
        # 计算欧氏距离
        error = np.sqrt((positions[i][0] - gt_positions[i][0])**2 + 
                         (positions[i][1] - gt_positions[i][1])**2)
        errors.append(error)
    
    # 计算平均误差和累积误差
    avg_error = np.mean(errors)
    cumulative_error = np.sum(errors)
    
    return avg_error, cumulative_error, errors

# 计算各方法的位置误差
try:
    compass_trad_avg_error, compass_trad_cum_error, compass_trad_errors = calculate_position_error(positions_compass_trad, gt_positions_ordered)
    gyro_trad_avg_error, gyro_trad_cum_error, gyro_trad_errors = calculate_position_error(positions_gyro_trad, gt_positions_ordered)

    # 创建位置误差表格
    error_df = pd.DataFrame({
        'Method': ['Compass (Traditional)', 'Gyro (Traditional)'],
        'Average Error (m)': [compass_trad_avg_error, gyro_trad_avg_error],
        'Cumulative Error (m)': [compass_trad_cum_error, gyro_trad_cum_error]
    })

    # 保存误差表格
    error_csv = os.path.join(output_dir, 'position_error_metrics.csv')
    error_df.to_csv(error_csv, index=False)
    print(f"位置误差指标保存至: {error_csv}")
    
    # 输出误差比较结果
    print("\n位置误差比较:")
    print(f"Compass (Traditional): 平均误差 = {compass_trad_avg_error:.2f}m, 累计误差 = {compass_trad_cum_error:.2f}m")
    print(f"Gyro (Traditional): 平均误差 = {gyro_trad_avg_error:.2f}m, 累计误差 = {gyro_trad_cum_error:.2f}m")
    
    # 绘制误差对比图
    fig3, ax3 = plt.subplots(figsize=(8, 4), dpi=300)
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9)

    x = np.arange(len(error_df['Method']))
    width = 0.35

    # 绘制平均误差条形图
    rects1 = ax3.bar(x - width/2, error_df['Average Error (m)'], width, label='Average Error (m)')
    # 添加数值标签
    for rect in rects1:
        height = rect.get_height()
        ax3.annotate(f'{height:.2f}m',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3点垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=fontSizeAll-1)

    # 设置x轴标签
    ax3.set_xticks(x)
    ax3.set_xticklabels(error_df['Method'], rotation=45, ha='right')
    ax3.set_ylabel('Error (m)')
    ax3.set_title('Position Error Comparison (Traditional Methods)')
    ax3.grid(axis='y', linestyle=':', alpha=0.5)

    # 保存图像
    error_plot_file = os.path.join(output_dir, 'position_error_comparison.png')
    plt.savefig(error_plot_file, bbox_inches='tight')
    print(f"位置误差对比图保存至: {error_plot_file}")
except Exception as e:
    print(f"计算误差时出错: {e}")

print("位置轨迹对比分析完成") 