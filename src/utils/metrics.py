import os
import json
import torch
import numpy as np
import pandas as pd
import datetime
import yaml
from sklearn.metrics import confusion_matrix, accuracy_score

class MetricRecorder:
    def __init__(self, save_dir, config, class_names=None):
        """
        Args:
            save_dir (str): 根日志目录 (如 'log')
            config (dict): 包含 data.source_wc 和 data.target_wcs 的配置字典
            class_names (list): 类别名称列表
        """
        # 1. 提取实验名称 (数据集名称)
        experiment_name = config['data']['dataset_name']

        # 2. 解析 Source WC 的数字 (例如 ["WC1"] -> "1")
        src_list = config['data']['source_wc']
        src_nums = sorted(["".join(filter(str.isdigit, x)) for x in src_list], key=int)
        src_tag = "_".join(src_nums)

        # 3. 解析 Target WCs 的数字 (例如 ["WC2", "WC3"] -> "2_3")
        tgt_list = config['data']['target_wcs']
        tgt_nums = sorted(["".join(filter(str.isdigit, x)) for x in tgt_list], key=int)
        tgt_tag = "_".join(tgt_nums)

        # 4. 生成带逻辑信息和时间戳的文件夹名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"train_{src_tag}_meta_{tgt_tag}_{timestamp}"

        # 5. 最终保存路径: log/gearbox/train_1_meta_2_3_2026...
        self.base_dir = os.path.join(save_dir, experiment_name, folder_name)

        self.class_names = class_names
        self.buffer = {}

        self.best_avg_acc = 0.0  # 记录历史最高平均准确率

        # 确保目录存在
        os.makedirs(self.base_dir, exist_ok=True)
        print(f"[Recorder] 日志目录已创建: {self.base_dir}")

    def save_config(self, config):
        """保存配置文件到当前时间戳目录下"""
        yaml_path = os.path.join(self.base_dir, 'config.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)

    def reset(self):
        """每轮评估前重置缓冲区"""
        self.buffer = {}

    def update(self, condition_name, preds, targets):
        """累积 Batch 数据"""
        if condition_name not in self.buffer:
            self.buffer[condition_name] = {'preds': [], 'targets': []}

        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        self.buffer[condition_name]['preds'].append(preds)
        self.buffer[condition_name]['targets'].append(targets)

    def calculate_and_save(self, epoch):
        """计算指标并保存 CSV 和 JSON"""
        summary_data = []

        # for wc, data in self.buffer.items():
        #     y_pred = np.concatenate(data['preds'])
        #     y_true = np.concatenate(data['targets'])

        #     cm = confusion_matrix(y_true, y_pred)
        #     total_acc = accuracy_score(y_true, y_pred) * 100


        all_wc_accs = [] # 新增：用于存当前所有工况的准确率

        for wc, data in self.buffer.items():
            y_pred = np.concatenate(data['preds'])
            y_true = np.concatenate(data['targets'])
            cm = confusion_matrix(y_true, y_pred)
            total_acc = accuracy_score(y_true, y_pred) * 100
            all_wc_accs.append(total_acc) # 记录当前准确率

            per_class_acc = (cm.diagonal() / (cm.sum(axis=1) + 1e-6)) * 100

            if self.class_names is None:
                labels = [f"Class {i}" for i in range(len(cm))]
            else:
                labels = self.class_names

            # --- 插入开始：输出混淆矩阵到终端 ---
            print(f"\n[Epoch {epoch}] Confusion Matrix - 工况: {wc}")
            print(pd.DataFrame(cm, index=labels, columns=labels))
            # --- 插入结束 ---


            # 保存详细 JSON
            details = {
                "Epoch": epoch,
                "Condition": wc,
                "Total_Accuracy": float(total_acc),
                "Per_Class_Accuracy": {label: float(acc) for label, acc in zip(labels, per_class_acc)},
                "Confusion_Matrix": cm.tolist()
            }

            json_path = os.path.join(self.base_dir, f"metrics_{wc}.json")
            with open(json_path, 'w') as f:
                json.dump(details, f, indent=4)

            # 收集汇总数据
            row = {"Condition": wc, "Total Acc": f"{total_acc:.2f}%"}
            for label, acc in zip(labels, per_class_acc):
                row[f"{label} Acc"] = f"{acc:.2f}%"
            summary_data.append(row)

        current_avg_acc = np.mean(all_wc_accs) if all_wc_accs else 0

        if current_avg_acc > self.best_avg_acc:
            self.best_avg_acc = current_avg_acc # 更新最高分

            # 执行保存操作
            df = pd.DataFrame(summary_data).sort_values(by="Condition")
            csv_path = os.path.join(self.base_dir, "best_summary_report.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            # 同时更新对应的 JSON
            # (如果需要 JSON 也只留最好的，就把 JSON 写入也挪到这个 if 块里)

            print(f"[Metrics] 发现更好结果! Avg Acc: {current_avg_acc:.2f}%, 已更新 best_summary_report.csv")
        # else:
        #     print(f"[Metrics] 当前 Avg Acc: {current_avg_acc:.2f}%, 未超过历史最高 {self.best_avg_acc:.2f}%, 跳过保存")
