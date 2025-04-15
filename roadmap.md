# Roadmap for Developing AI/ML Algorithms for Indoor Navigation System

This roadmap outlines the development plan for implementing AI/ML algorithms to improve an indoor navigation system that combines gyroscope and compass data.

## Phase 1: Data Preprocessing and Classification

- [x] Develop a data parser that can automatically identify and classify different sensor data
- [x] Create a visualization comparing raw and cleaned data to illustrate the impact of the cleaning process on anomaly amplitude/frequency and data reliability
- [x] Generate a flowchart documenting the data processing steps in Phase 1

* **解释:** 此步骤的目的是从原始的传感器数据文件中读取数据，并将其按照不同的传感器类型（陀螺仪、指南针、地面真实值）进行分类。同时，为每一条数据记录添加一个步数或者时间戳的标识，方便后续的时序分析。还需要设置一个初始位置作为导航的起点。
* **具体操作步骤:**
* **读取原始数据:** 使用适当的编程语言（如Python）读取存储传感器数据的 `collected_data` 文件。该文件可能包含多种格式（例如CSV, JSON, TXT），需要根据实际情况选择合适的读取方法。
* **数据解析:** 分析原始数据的结构，确定陀螺仪、指南针和地面真实值数据分别对应的字段和格式。
* **数据分类:** 根据解析结果，将读取到的每一行数据识别其传感器类型，并将其存储到不同的数据结构中（例如，可以使用字典或Pandas DataFrame）。
* **添加步数/时间戳:** 为每一条数据记录添加一个表示步数或者时间戳的字段。如果原始数据中没有明确的步数信息，可以简单地按照数据记录的顺序进行编号。
* **设置初始位置:** 根据实验或应用场景的需求，确定导航的起始位置，并在数据处理过程中记录下来。
* **输出清洗后的数据:** 将分类、添加步数/时间戳后的数据保存为 `cleaned_data`，方便后续步骤使用。

* **解释:** 通过可视化原始数据和清洗后的数据，可以直观地看到数据清洗的效果，例如是否有效地去除了异常值、重复数据或填补了缺失值，从而提高数据的质量和可靠性。
* **具体操作步骤:**
* **选择可视化工具:** 使用数据可视化库（如Matplotlib, Seaborn, Plotly）创建图表。
* **绘制原始数据图:** 针对每种传感器类型（陀螺仪、指南针、地面真实值），绘制其原始数据随步数/时间戳变化的曲线图。重点展示数据中的异常波动、重复值或不一致性。
* **绘制清洗后数据图:** 同样地，针对每种传感器类型，绘制清洗后的数据随步数/时间戳变化的曲线图。
* **对比分析:** 将原始数据图和清洗后数据图进行对比，观察异常值的去除情况、数据平滑程度的变化等。可以使用子图的形式将清洗前后的数据并排显示，或者在同一张图上使用不同的颜色进行区分。
* **指标量化 (可选):** 可以计算清洗前后数据的统计指标（如均值、标准差、最大值、最小值等），进一步量化数据清洗的效果。

* **解释:** 创建一个流程图，清晰地展示Phase 1中数据的导入、读取、分类和清洗的具体步骤，以及各个步骤之间的逻辑关系，方便理解整个数据预处理流程。
* **具体操作步骤:**
* **确定流程节点:** 识别Phase 1中的关键步骤，例如"读取原始数据"、"解析数据"、"分类数据"、"处理重复值"、"处理异常值"等。
* **绘制流程图:** 使用流程图绘制工具（如draw.io, Lucidchart）或简单的绘图软件，将各个步骤用不同的图形表示（例如，矩形表示处理步骤，菱形表示判断），并使用箭头连接各个步骤，表示数据的流向和处理顺序。
* **添加说明:** 为每个流程节点添加简要的文字说明，解释该步骤的具体操作和目的。

## Phase 2: Processing and Visualization of Different Data Categories

- [x] Visualize the Ground Truth position path,[interpolating using AI/ML or environment adaptation techniques]
- [x] Create models for Heading and position
- [x] Explore deep learning approaches (LSTM, GRU networks) for the Heading estimation
- [x] Generate a flowchart documenting the data processing steps in ML
- [x] Visualize the position paths of Ground Truth, Gyroscope, and Compass on a single coordinate graph for direct comparison.

* **解释:** 将地面真实值的位置数据在二维或三维坐标系中绘制出来，形成一个轨迹图，直观地展示用户的实际移动路径。如果地面真实值数据的采样频率较低，或者存在缺失，可以使用插值方法（包括传统的数学插值方法或基于AI/ML的环境适应方法）来平滑轨迹。
* **具体操作步骤:**
* **选择可视化工具:** 使用Matplotlib, Seaborn, Plotly等库进行可视化。
* **提取地面真实值位置数据:** 从 `cleaned_data` 中提取地面真实值的经纬度或在室内坐标系下的位置信息。
* **数据插值 (如果需要):** 如果地面真实值数据点稀疏或不连续，可以使用插值算法（如线性插值、样条插值）来生成更密集的轨迹点。也可以尝试使用简单的AI/ML模型（如回归模型）或基于环境信息的模型进行插值。
* **绘制轨迹图:** 使用提取或插值后的位置数据，在坐标系中绘制轨迹线。可以使用不同的颜色或标记来表示轨迹的不同部分。

* **解释:** 分别使用陀螺仪和指南针的数据来估算设备的朝向（Heading）。陀螺仪可以测量角速度，通过积分可以得到角度变化，从而估算Heading；指南针可以直接测量磁北方向，从而得到Heading。
* **具体操作步骤:**
* **陀螺仪Heading计算:**
* **角速度积分:** 从 `cleaned_data` 中提取陀螺仪的角速度数据。
* **初始Heading:** 需要一个初始的Heading作为积分的起点，可以从指南针的初始读数或地面真实值的初始方向获取。
* **角度累积:** 对陀螺仪的角速度数据进行时间上的积分，得到角度的变化量。将这些变化量累加到初始Heading上，即可得到随时间变化的陀螺仪Heading。需要注意陀螺仪的漂移误差。
* **指南针Heading计算:**
* **磁场角度转换:** 从 `cleaned_data` 中提取指南针的磁场强度数据（通常是三轴数据）。
* **Heading计算:** 使用公式将三轴磁场强度数据转换为相对于磁北方向的Heading。需要考虑设备的姿态（俯仰角和横滚角）对指南针读数的影响。

* **解释:** 分别使用陀螺仪和指南针的数据来估算设备的位置。陀螺仪可以通过步长检测和方向估计进行航位推算（Dead Reckoning）；指南针可以提供方向信息，但单独无法确定位置。
* **具体操作步骤:**
* **陀螺仪Position计算 (航位推算):**
* **步长检测:** 使用陀螺仪的加速度数据或其他传感器数据检测用户的步数。
* **步长估计:** 根据经验或模型估计每一步的长度。
* **方向估计:** 使用之前计算得到的陀螺仪Heading作为每一步的移动方向。
* **位置更新:** 从初始位置开始，根据每一步的步长和方向，逐步计算出设备的位置。航位推算会随着时间的推移产生累积误差。
* **指南针Position计算:**
* **方向信息:** 指南针只能提供设备的朝向信息，无法直接确定位置。可以将其视为位置估计的一个约束条件，例如，可以知道设备在某个时刻朝向某个方向，但不知道具体在哪里。

* **解释:** 将地面真实值的位置轨迹、通过陀螺仪估算的位置轨迹和（如果可能）通过指南针信息辅助估算的位置轨迹绘制在同一个坐标系中，方便直观地比较不同方法的位置估计结果与真实路径之间的差异。
* **具体操作步骤:**
* **选择可视化工具:** 使用Matplotlib, Seaborn, Plotly等库。
* **提取位置数据:** 从之前计算得到的结果中提取地面真实值位置、陀螺仪估算位置和指南针辅助估算的位置数据。
* **绘制轨迹图:** 在同一个坐标系中绘制三条不同的轨迹线，可以使用不同的颜色或线型来区分。添加图例说明每条轨迹代表的含义。
* **分析比较:** 观察三条轨迹之间的差异，例如陀螺仪轨迹的漂移情况，指南针信息对轨迹的约束作用等。

## Phase 3: Sensor Fusion and Improved Heading Estimation
- [ ] Develop AI models for sensor fusion (Gyro + Compass)
   - [ ] Explore deep learning approaches (LSTM, GRU networks)
   - [ ] Design end-to-end neural networks for heading estimation
- [ ] Create a benchmark system to evaluate heading estimation accuracy
- [ ] Visualize the results, comparing Gyroscope and Compass headings alongside relevant factors/changes to demonstrate the model's reliability and the improvements achieved.

* **解释:** 使用循环神经网络（RNN）的变体，如长短期记忆网络（LSTM）和门控循环单元（GRU），来融合陀螺仪和指南针的时序数据，从而更准确地估计设备的朝向。这些模型能够学习传感器数据中的时间依赖关系和特征，并自动进行传感器信息的加权和融合。
* **详细步骤指导:**
* **数据准备:**
* 收集并预处理陀螺仪和指南针的时序数据，确保数据对齐和同步。
* 将数据划分为训练集、验证集和测试集。
* 对数据进行归一化或标准化，使其数值范围在一定区间内，有助于模型训练。
* 将时序数据转换为适合LSTM/GRU网络输入的格式，通常是三维数组 `(batch_size, time_steps, features)`，其中 `features` 包括陀螺仪的角速度和指南针的磁场强度等。
* 准备对应的标签数据，即真实的Heading值（可以从地面真实值数据中获取）。
* **模型构建:**
* 使用深度学习框架（如TensorFlow, PyTorch）构建LSTM或GRU模型。
* 模型结构可以包括一个或多个LSTM/GRU层，以及一些全连接层。
* 可以尝试不同的网络结构、隐藏层大小和激活函数，以找到最佳模型。
* **模型训练:**
* 选择合适的损失函数（如均方误差）来衡量模型预测的Heading与真实Heading之间的差异。
* 选择优化器（如Adam, RMSprop）来更新模型参数，最小化损失函数。
* 使用训练集对模型进行训练，并在验证集上监控模型的性能，调整超参数，防止过拟合。
* **模型评估:**
* 使用测试集评估训练好的模型的性能，计算Heading估计的准确度指标（如平均绝对误差、均方根误差）。

* **解释:** end-to-end networks 设计一个可以直接从原始传感器数据（陀螺仪和指南针）输入到输出Heading估计的端到端神经网络。这种方法可以避免传统方法中复杂的信号处理步骤，让模型自动学习最优的特征提取和Heading估计策略。
* **详细步骤指导:**
* **数据准备:** 与步骤(1)类似，准备陀螺仪和指南针的时序数据以及对应的真实Heading标签。
* **模型构建:**
* 可以尝试使用不同类型的神经网络结构，例如：
* **卷积神经网络 (CNN) + 循环神经网络 (RNN):** 使用CNN提取传感器数据中的空间特征，然后使用RNN处理时间序列信息。
* **Transformer 网络:** 利用自注意力机制捕捉传感器数据中的长距离依赖关系。
* **混合模型:** 结合不同类型的神经网络层，以充分利用各种模型的优点。
* 设计合适的网络层数、每层的神经元/滤波器数量、激活函数等。
* **模型训练和评估:** 与步骤(1)中的模型训练和评估过程类似，选择合适的损失函数、优化器，使用训练集训练模型，并在验证集和测试集上评估模型的性能。

* **解释:** 通过可视化比较单独使用陀螺仪和指南针估计的Heading、以及使用深度学习模型融合后的Heading，来展示传感器融合的优势和模型的有效性。同时，可以将一些可能影响Heading估计的因素（如用户的运动状态、环境磁场干扰等）也进行可视化，以便更好地理解模型在不同情况下的表现。
* **具体操作步骤:**
* **绘制Heading对比图:** 在同一张图上绘制以下曲线：
* 使用陀螺仪单独估计的Heading随时间变化的曲线。
* 使用指南针单独估计的Heading随时间变化的曲线。
* 使用深度学习模型融合陀螺仪和指南针数据后估计的Heading随时间变化的曲线。
* 真实的Heading（从地面真实值数据中获取）作为参考。
* **可视化相关因素 (可选):** 如果有关于用户运动状态（例如，是否静止、行走速度等）或环境信息（例如，磁场强度变化）的数据，也可以将其在同一张图或单独的子图中进行可视化，以便分析这些因素对Heading估计的影响。
* **分析和突出改进:** 观察对比图，分析深度学习模型融合后的Heading是否比单独使用陀螺仪或指南针的估计更接近真实值，是否更平滑、更稳定。重点突出模型在哪些情况下表现更好，以及模型的可信度和意义。

* **Create a benchmark system to evaluate heading estimation accuracy:**
* **解释:** 建立一套完善的评估体系，用于量化不同Heading估计算法的性能，包括传统方法和基于AI/ML的方法。
* **详细步骤指导:**
* **选择评估指标:** 确定用于评估Heading估计准确度的指标，例如：
* **平均绝对误差 (MAE):** 预测Heading与真实Heading之间绝对差值的平均值。
* **均方根误差 (RMSE):** 预测Heading与真实Heading之间差值平方的平均值的平方根。
* **角度误差分布:** 统计不同误差范围内的样本比例。
* **构建评估数据集:** 准备一个包含各种运动场景和环境条件的测试数据集，并包含准确的地面真实值Heading作为评估的基准。
* **实现评估工具:** 编写代码或使用现有工具，能够针对不同的Heading估计算法，在评估数据集上计算所选的评估指标。
* **设定基准:** 可以选择一个简单的传统方法（例如，直接使用指南针数据进行Heading估计）作为基准，将其他算法的性能与之进行比较。


## Phase 4: Adaptive Quasi-Static Detection
- [ ] Implement traditional Quasi-Static detection on Gyro data
- [x] Develop an ML algorithm to optimize quasi-static detection parameters
   - [x] Implement a genetic algorithm for parameter optimization
   - [x] Use reinforcement learning to adapt parameters in real-time
   - [optional] Create a CNN-based model for direct quasi-static state classification
- [ ] Compare and visualize the QS regions detected by traditional and ML-enhanced methods.

* **解释:** 2识别用户提供的传统QS检测方法中可以调节的参数，并提出使用机器学习方法来自动优化这些参数的建议，以提高QS检测的准确性和鲁棒性。
* **详细步骤指导:**
* **分析传统方法代码:** 仔细阅读用户提供的QS检测代码，理解其实现原理，并识别出可以调节的参数。这些参数可能包括滑动窗口的大小、阈值的设定等。
* **提出改进参数建议:** 基于对传统方法的理解和对陀螺仪数据特性的分析，提出可能影响QS检测性能的关键参数。例如，滑动窗口的大小会影响检测的灵敏度，阈值的设定会影响检测的严格程度。
* **机器学习优化思路:** 建议使用机器学习算法（如遗传算法、强化学习）来自动搜索最优的参数组合。这需要定义一个评估QS检测性能的指标（例如，准确率、召回率、F1值），并构建一个能够根据这些指标自动调整参数的优化框架。

* **解释:** 3使用遗传算法（Genetic Algorithm, GA）来寻找传统QS检测方法的最优参数组合。遗传算法是一种模拟生物进化过程的优化算法，通过不断地选择、交叉和变异操作，逐步搜索到最优解。
* **详细步骤指导:**
* **定义优化目标:** 确定需要优化的目标，例如最大化QS检测的准确率或F1值。这需要一个带有真实QS标签的数据集作为评估的基础。
* **参数编码:** 将传统QS检测方法的参数编码成遗传算法中的染色体。
* **初始化种群:** 随机生成一组染色体作为初始种群。
* **适应度评估:** 对种群中的每个染色体（即每组参数），使用其对应的参数运行QS检测算法，并在评估数据集上计算适应度值（例如，准确率或F1值）。
* **选择:** 根据适应度值选择优秀的染色体作为下一代的父代。
* **交叉:** 对选中的父代染色体进行交叉操作，生成新的子代染色体。
* **变异:** 对子代染色体进行变异操作，引入新的参数组合。
* **迭代:** 重复选择、交叉和变异操作，直到满足停止条件（例如，达到最大迭代次数或找到满足要求的参数组合）。
* **获取最优参数:** 从最终的种群中选择适应度最高的染色体，解码得到最优的参数组合。

* **解释:** 4使用强化学习（Reinforcement Learning, RL）方法来动态地调整QS检测的参数。强化学习通过让一个智能体（Agent）在环境中进行交互，并根据其行为获得的奖励或惩罚来学习最优策略。
* **详细步骤指导:**
* **定义状态:** 将当前传感器数据的某些特征（例如，陀螺仪角速度的统计特征）作为强化学习智能体的状态。
* **定义动作:** 将QS检测参数的调整作为智能体的动作。
* **定义奖励:** 设计一个奖励函数，用于衡量当前参数设置下QS检测的性能。例如，如果检测到的QS区域能够有效地用于校正Heading，则给予正向奖励；如果检测错误，则给予负向奖励。
* **构建强化学习环境:** 将QS检测过程和评估过程构建成一个强化学习环境。
* **选择强化学习算法:** 选择合适的强化学习算法，例如Q-learning, Deep Q-Network (DQN) 或策略梯度方法。
* **训练智能体:** 使用历史数据或模拟数据训练强化学习智能体，使其学习在不同的状态下采取最优的参数调整动作，以最大化累积奖励。
* **实时部署:** 将训练好的强化学习模型部署到实际的导航系统中，使其能够根据实时的传感器数据动态地调整QS检测参数.

* **解释:** 通过可视化对比传统方法和使用机器学习优化后的方法（例如，遗传算法或强化学习）检测到的QS区域，以及（如果使用了CNN）CNN模型预测的QS状态，来展示改进方法的优势。
* **具体操作步骤:**
* **绘制陀螺仪数据图:** 绘制陀螺仪的角速度随时间变化的曲线图。
* **标记传统方法检测结果:** 在陀螺仪数据图上，使用一种颜色或标记标出传统方法检测到的QS区域。
* **标记改进方法检测结果:** 在同一张图上，使用不同的颜色或标记标出使用遗传算法、强化学习或CNN模型检测到的QS区域。
* **对比分析:** 观察不同方法检测到的QS区域的大小、位置和与实际QS状态的吻合程度。如果能够获取真实的QS标签，可以将检测结果与真实标签进行比较，计算准确率等指标，并在图上进行标注。
* **突出改进:** 分析可视化结果，重点展示改进方法在哪些方面优于传统方法，例如能够更准确地识别QS区域，减少误检或漏检，从而提高QS检测的可靠性和有效性。

## Phase 5: QS-Based Heading Correction and Positioning Optimization
- [ ] QS-Based Heading Correction:
   Use the QS regions detected in Phase (4 with different methods) to correct the raw heading data. Output the corrected heading data for each method.
- [ ] Position Calculation & Visualization:
   Calculate the corresponding positions using the corrected heading values. Output and visualize the resulting position trajectories.
- [ ] Comparative Analysis of QS Methods:
   Compare and analyze the credibility and significance of QS-based correction methods by evaluating factors such as:
	•	Average heading error (mean error) before and after correction.
	•	Variation rate (standard deviation or variance) of heading values.
	•	Overall position drift and error over time.
	•	Percentage improvement in heading estimation and positioning accuracy.
	•	Consistency and stability of the corrected trajectories (e.g., confidence intervals).


* **解释** Detailed Explanations and Operational Steps
	1.	QS-Based Heading Correction:
	•	Operation:
	•	For each QS detection method developed in Phase 4, apply its detected QS regions to recalibrate the raw heading estimates.
	•	Generate corrected heading data arrays corresponding to each method.
	•	Purpose:
	•	To leverage stable periods (QS) as reference points, reducing the accumulated error from sensor drift and noise.
	2.	Position Calculation & Visualization:
	•	Operation:
	•	Use the corrected heading data (from each QS method) in conjunction with step length or displacement information to compute the trajectory (position) over time.
	•	Visualize the computed trajectories on a 2D coordinate plot, overlaying the results from different QS-based corrections.
	•	Purpose:
	•	To observe how the heading correction affects the overall positioning and to directly compare different QS methods.
	3.	Comparative Analysis of QS Methods:
	•	Factors to Calculate & Compare:
	•	Average Heading Error: Compute the mean absolute error between the raw heading and the corrected heading (or between corrected heading and Ground Truth, if available).
	•	Variation Rate: Determine the standard deviation or variance of heading measurements before and after correction to evaluate stability improvements.
	•	Position Drift/Error: Calculate the positional error or drift over time, comparing the displacement differences.
	•	Improvement Percentage: Assess the percentage reduction in error (both heading and position) after applying the QS-based correction.
	•	Stability Metrics: Use confidence intervals or consistency measures to quantify how stable the corrected trajectories are over time.

### 基于QS检测结果的Gyro Heading纠正与位置优化实施步骤

* **QS数据的预处理与分析**
  1. **读取QS检测结果数据**
     * 加载Phase 4中生成的quasi_static_data.csv和quasi_static_averages.csv文件
     * 提取每个QS区间的时间戳范围、QS区间编号、平均Compass Heading值和True Heading值
  
  2. **Gyro原始数据分析**
     * 加载陀螺仪原始数据，包括角速度和已计算的原始Heading
     * 计算Gyro Heading与真实Heading之间的初始误差和偏差趋势
     * 识别Gyro数据的漂移特性和模式

* **Heading纠正方法设计与实现**
  1. **QS区间基准值确定**
     * 对每个QS区间，计算该区间内Compass Heading的均值作为参考值
     * 使用IQR或其他统计方法去除异常值，确保参考值可靠性
     * 将Compass Heading参考值与True Heading进行比对，评估参考值质量

  2. **Gyro Heading漂移模型**
     * 分析相邻QS区间间的Gyro Heading漂移特性
     * 建立漂移模型（线性、指数或多项式拟合）
     * 估计Gyro Heading在非QS区间内的累积误差

  3. **Heading纠正算法实现**
     * **段间补偿法**：在两个QS区间之间，基于漂移模型对Gyro Heading进行线性或非线性补偿
     * **实时校准法**：在每个时间点，根据最近的QS参考值和预测的漂移来调整当前Heading
     * **融合滤波法**：使用滤波器(卡尔曼、粒子滤波)将Gyro Heading与QS区间提供的参考值融合
     * **加权平均法**：根据距离QS区间的时间间隔，为Gyro Heading和Compass参考值分配权重

  4. **纠正结果评估**
     * 计算纠正后Heading与True Heading之间的误差
     * 对比不同纠正方法的精度和稳定性
     * 分析纠正前后Heading的稳定性提升

* **改进位置计算**
  1. **位置计算模型更新**
     * 使用纠正后的Heading数据重新计算位置坐标
     * 结合步长检测和步长估计优化位置推算
     * 针对不同楼层/场景调整位置计算参数

  2. **分段位置优化**
     * 在QS区间处重置累积误差
     * 对QS区间之间的轨迹段进行独立优化
     * 通过轨迹平滑算法减少位置跳变

  3. **位置修正验证**
     * 计算修正前后的轨迹与Ground Truth的平均距离误差
     * 分析轨迹形状相似度和关键点准确性
     * 评估位置估计的稳定性和可靠性

* **可视化与结果分析**
  1. **多层次可视化**
     * 2D平面图展示不同修正方法的轨迹对比
     * 3D轨迹图包含楼层信息和QS区间标记
     * 时序图展示Heading和位置误差随时间的变化

  2. **性能指标统计与展示**
     * 创建不同方法的Heading误差分布直方图
     * 生成位置误差的箱线图或小提琴图
     * 计算并展示各方法在不同指标上的改进百分比

  3. **综合分析报告**
     * 编写详细分析报告，比较不同QS检测和Heading纠正方法的优缺点
     * 总结最佳参数组合和适用场景
     * 提出进一步优化方向和建议

* **自适应参数调整与优化**
  1. **参数敏感性分析**
     * 测试不同参数设置对纠正结果的影响
     * 识别关键参数及其最佳取值范围
  
  2. **场景适应性增强**
     * 根据场景特征(室内/室外、楼层变化等)自动调整算法参数
     * 开发能根据用户移动模式动态适应的纠正策略
  
  3. **算法集成与模块化**
     * 设计模块化框架，支持不同QS检测和Heading纠正方法的灵活组合
     * 创建统一的评估接口，便于新算法的集成和比较