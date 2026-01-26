"""
快速对比所有 t-SNE 可视化结果
"""
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def main():
    base_dir = "experiments/SNE"
    datasets = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

    # 收集图片：分 teacher 和 mcid
    data = []  # [(dataset, teacher_img, mcid_img), ...]

    for ds in datasets:
        ds_path = os.path.join(base_dir, ds)
        pngs = sorted([f for f in os.listdir(ds_path) if f.endswith('.png')])

        teacher_img, mcid_img = None, None
        for f in pngs:
            path = os.path.join(ds_path, f)
            if 'mcid' in f.lower():
                mcid_img = path
            else:
                teacher_img = path

        if teacher_img or mcid_img:
            data.append((ds, teacher_img, mcid_img))

    if not data:
        print("No images found!")
        return

    n = len(data)

    # 横向排列：列=数据集，行=Teacher/MCID
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))

    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (ds, teacher_path, mcid_path) in enumerate(data):
        # Teacher (上面一行)
        ax_t = axes[0, col]
        if teacher_path:
            img = mpimg.imread(teacher_path)
            ax_t.imshow(img)
        ax_t.set_title(f"{ds}\nTeacher", fontsize=11, weight='bold')
        ax_t.axis('off')

        # MCID (下面一行)
        ax_m = axes[1, col]
        if mcid_path:
            img = mpimg.imread(mcid_path)
            ax_m.imshow(img)
        ax_m.set_title("MCID", fontsize=11, weight='bold')
        ax_m.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
