import os
import matplotlib.pyplot as plt

desktop = os.path.join(os.path.expanduser('~'), 'Desktop')  # 自动获取桌面路径
ha6_folder = os.path.join(desktop, 'HA6')  # HA6 文件夹路径

all_experiments = [{'name': 'Experiment1','epochs': [1, 2, 3],'train_accuracy': [0.8566, 0.9260, 0.9260],'test_accuracy': [0.8863, 0.8935, 0.8883]},
                   {'name': 'Experiment2', 'epochs': [1, 2, 3],'train_accuracy': [0.6650, 0.8675, 0.9500],'test_accuracy': [0.7700, 0.8000, 0.6800]},
                   {'name': 'Experiment3','epochs': [1, 2, 3],'train_accuracy': [0.5869, 0.6527, 0.6758],'test_accuracy': [0.6634, 0.6916, 0.7000]},
                   {'name': 'Experiment4','epochs': [1, 2, 3], 'train_accuracy': [0.8666, 0.9283, 0.9672],'test_accuracy': [0.8650, 0.9013, 0.8911]},
                   {'name': 'Experiment5', 'epochs': [1, 2, 3],'train_accuracy': [0.8633, 0.9302, 0.9689],'test_accuracy': [0.8927, 0.8960, 0.8925]},
                   {'name': 'Experiment6','epochs': [1, 2, 3, 4, 5, 6],'train_accuracy': [0.8594, 0.9282, 0.9641, 0.9789, 0.9849, 0.9867],'test_accuracy': [0.8868, 0.8905, 0.8873, 0.8888, 0.8851, 0.8893]},
                   {'name': 'Experiment7', 'epochs': [1, 2, 3],'train_accuracy': [0.8653, 0.9265, 0.9634],'test_accuracy': [0.8944, 0.8953, 0.8883]},
                   {'name': 'Experiment8','epochs': [1, 2, 3],'train_accuracy': [0.8909, 0.9445, 0.9714],'test_accuracy': [0.9203, 0.9173, 0.9208]},
                   {'name': 'Experiment9', 'epochs': [1, 2, 3],'train_accuracy': [0.8418, 0.8827, 0.9073],'test_accuracy': [0.8642, 0.8774, 0.8908]},
                   {'name': 'Experiment10', 'epochs': [1, 2, 3],'train_accuracy': [0.8639, 0.9278, 0.9654],'test_accuracy': [0.8757, 0.8961, 0.8889]}
                  ]

print("📁 所有图片将保存到：", ha6_folder)

for exp in all_experiments:
    plt.figure(figsize=(8,8))
    plt.plot(exp['epochs'], exp['train_accuracy'], marker = 'o', label='Train Accuracy', color='blue')
    plt.plot(exp['epochs'], exp['test_accuracy'], marker = 'x', label='Test Accuracy', color='orange')

    plt.title(f'{exp["name"]}: Training & Test Accuracy per Epoch', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    save_path = os.path.join(ha6_folder, f'{exp["name"].replace(" ", "_").lower()}_accuracy_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 高分辨率保存
    print('图片已保存到HA6文件夹')