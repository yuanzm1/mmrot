import json
import matplotlib.pyplot as plt


def extract_and_plot_mAP(json_file_path, out_path):
    mAP_values = []
    epochs = []

    with open(json_file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if "mode" in data and data["mode"] == "val":
                mAP_values.append(data["mAP"])
                epochs.append(data["epoch"])
                
    plt.plot(epochs, mAP_values)
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('mAP over Epochs')
    plt.savefig(out_path)
    # plt.show()


if __name__ == "__main__":
    json_file_path = '/home/yuanzm/mmrotate/work_dirs/ow_oriented_rcnn_r50_untrained_fpn_1x_dior_le90/20241211_004731.log.json'
    out_path = '/home/yuanzm/mmrotate/work_dirs/ow_oriented_rcnn_r50_untrained_fpn_1x_dior_le90/20241211_004731-mAP.png'
    extract_and_plot_mAP(json_file_path, out_path)