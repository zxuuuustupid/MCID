from huggingface_hub import HfApi

# 1. 直接在这里填入你的 Token
your_token = "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" # 替换成你真实的 Token
import os
import time

# 1. 保持核武级屏蔽（防止代理干扰）
os.environ['no_proxy'] = '*'
os.environ['HTTP_PROXY'] = ""
os.environ['HTTPS_PROXY'] = ""
os.environ['http_proxy'] = ""
os.environ['https_proxy'] = ""
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import HfApi
from requests.exceptions import RequestException

# 配置
# your_token = "你的hf_token"
repo_id = "Jusin0305/mcid"
local_folder_path = r"F:\Project\mid\S-MID\data\gearbox"

api = HfApi(token=your_token, endpoint="https://hf-mirror.com")

def start_upload():
    retry_count = 0
    max_retries = 50 # 自动重试50次

    while retry_count < max_retries:
        try:
            print(f"\n🚀 第 {retry_count + 1} 次尝试上传...")
            api.upload_folder(
                folder_path=local_folder_path,
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo=".",
                commit_message=f"Upload batch {retry_count}",
                # 核心参数：如果你的 huggingface_hub 版本较新，开启下面这个可以提高稳定性
                # multi_commits=True,
                # multi_commits_threshold=100 * 1024 * 1024 # 100MB
            )
            print("✅ 【全部上传成功！】")
            break
        except Exception as e:
            retry_count += 1
            print(f"⚠️ 本次上传中断（可能是网络波动），3秒后自动续传... \n错误信息: {e}")
            time.sleep(3) # 等待3秒后重试

if __name__ == "__main__":
    start_upload()
