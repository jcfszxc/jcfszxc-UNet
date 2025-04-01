


## 1. 设计calculate_uncertainty计算样本不确定性，

### 1.1 根据不确定性值来替换原来的alpha权重
    alpha = 1 - uncertainty.mean()
    loss = alpha * bce_loss + (1 - alpha) * dice

    实验结果：没有提升

### 1.2 根据不确定性值更新probs，调整采样概率
    # 计算不确定性和筛选困难样本
    uncertainty_np = calculate_uncertainty(masks_pred, batch_labels)
        
    # 更新采样概率 - 这是新增加的关键部分
    memory_factor = 0.99  # 控制历史概率的保留程度 (0.9-0.99)
    uncertainty_factor = 5.0  # 控制不确定性的影响强度 (1.0-5.0)
    
    # 更高效的向量化实现
    if not hasattr(model, 'seen_mask'):
        model.seen_mask = np.zeros(len(probs), dtype=bool)

    # 向量化衰减操作 - 只对已见样本应用
    decay_mask = np.ones_like(probs)
    decay_mask[model.seen_mask] = memory_factor
    probs *= decay_mask

    # 更新当前批次
    for idx, u in zip(sampled_indices, uncertainty_np):
        probs[idx] += u * uncertainty_factor
        model.seen_mask[idx] = True
    
    # 重新归一化概率
    probs = probs / np.sum(probs)
    

    实验结果：没有提升


## 2. 尝试使用clDice

    实验结果：没有提升
    

## 3. 尝试使用畸变矫正作为数据增强的策略

    实验结果：没有提升