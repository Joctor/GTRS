import numpy as np
import math

# ==============================
# 1. 加载并合并全量轨迹数据
# ==============================
data16384 = np.load('traj_final/16384.npy') 
data8192 = np.load('traj_final/8192.npy')
data = np.concatenate([data16384, data8192], axis=0)
print(f"✅ 拼接后的总数据形状: {data.shape}")

# ==============================
# 2. 下采样与计算差分
# ==============================
data = data[:, ::5, :] # 时间步下采样
x = data[:, :, 0]
y = data[:, :, 1]
theta = data[:, :, 2]
B, L = x.shape

x_diff = np.diff(x, n=1, axis=1, prepend=np.zeros((B, 1)))
y_diff = np.diff(y, n=1, axis=1, prepend=np.zeros((B, 1)))
delta_theta = np.diff(theta, n=1, axis=1, prepend=np.zeros((B, 1)))
delta_theta = (delta_theta + math.pi) % (2 * math.pi) - math.pi

# ==============================
# 3. 混合策略标准化处理（Sin/Cos 方案）
# ==============================

# 【X轴】带符号的 Log 变换 + Z-score
x_diff_log = np.sign(x_diff) * np.log1p(np.abs(x_diff))
x_mean, x_std = x_diff_log.mean(), x_diff_log.std()
x_final = (x_diff_log - x_mean) / (x_std + 1e-8)

# 【Y轴】常规 Z-score 标准化
y_mean, y_std = y_diff.mean(), y_diff.std()
y_final = (y_diff - y_mean) / (y_std + 1e-8)

# 【角度】Sin/Cos 映射
sin_theta = np.sin(delta_theta)
cos_theta = np.cos(delta_theta)

# ==============================
# 4. 打印所有均值与方差统计量
# ==============================
print("\n📊 详细统计量报告:")
print("-" * 45)

# X 轴统计量
print(f"{'X轴 (Log变换)':<15} | 原始均值: {x_diff_log.mean() } | 原始标准差: {x_diff_log.std() }")
print("-" * 45)

# Y 轴统计量
print(f"{'Y轴 (原始)':<15} | 原始均值: {y_diff.mean() } | 原始标准差: {y_diff.std() }")
print("-" * 45)

# 角度统计量 (Sin/Cos 不需要额外缩放，原始即最终)
print(f"{'Sin_Theta':<15} | 均值: {sin_theta.mean() } | 标准差: {sin_theta.std() }")
print(f"{'Cos_Theta':<15} | 均值: {cos_theta.mean() } | 标准差: {cos_theta.std() }")
print("-" * 45)

# ==============================
# 5. 拼接特征并保存
# ==============================
final_features = np.stack([x_final, y_final, sin_theta, cos_theta], axis=-1).astype(np.float32)
print(f"\n🚀 最终特征形状: {final_features.shape}")

np.save('traj_final/24576_processed.npy', final_features)
print("💾 已成功保存至 traj_final/24576_processed.npy")