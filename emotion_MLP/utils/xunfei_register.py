import base64

import hashlib

import hmac

import json

import os

import time

from datetime import datetime

from urllib.parse import urlencode

from wsgiref.handlers import format_date_time

import requests

import librosa

import numpy as np

# --- 你的凭据 ---

APPID = "34d6c5db"

APIKey = "2ea750fed0dc74d3abcbba4e5f0c7759"

APISecret = "M2QwNDEwYmQ5OWMyOGU2MTcxMWE0MmFm"

GROUP_ID = "morpheus_vip_group"


class IFlyVPR:

    def __init__(self, appid, api_key, api_secret):

        self.APPID = appid

        self.APIKey = api_key

        self.APISecret = api_secret

        self.RequestUrl = 'https://api.xf-yun.com/v1/private/s782b4996'

        self.Host = "api.xf-yun.com"

        self.Path = "/v1/private/s782b4996"

    def assemble_auth_url(self, method="POST"):

        """生成官方要求的带鉴权参数的 URL"""

        now = datetime.now()

        date = format_date_time(time.mktime(now.timetuple()))

        # 1. 签名字符串

        signature_origin = f"host: {self.Host}\ndate: {date}\n{method} {self.Path} HTTP/1.1"

        # 2. HMAC-SHA256 加密

        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),

                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        # 3. 构造 Authorization 原始串

        auth_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(auth_origin.encode('utf-8')).decode(encoding='utf-8')

        # 4. 拼接到 URL 后面

        values = {

            "host": self.Host,

            "date": date,

            "authorization": authorization

        }

        return self.RequestUrl + "?" + urlencode(values)

    def run(self, func, p_dict, audio_bytes=None):

        url = self.assemble_auth_url()

        body = {

            "header": {"app_id": self.APPID, "status": 3},

            "parameter": {

                "s782b4996": {

                    "func": func,

                    **p_dict

                }

            }

        }

        if audio_bytes:
            body["payload"] = {

                "resource": {

                    "encoding": "raw",  # 官方文档写lame(mp3)，但传16k raw通常更稳

                    "sample_rate": 16000,

                    "channels": 1,

                    "bit_depth": 16,

                    "status": 3,

                    "audio": base64.b64encode(audio_bytes).decode('utf-8')

                }

            }

        headers = {'content-type': "application/json"}

        try:

            response = requests.post(url, data=json.dumps(body), headers=headers, timeout=10)

            return response.json()

        except Exception as e:

            return {"header": {"code": -1, "message": str(e)}}


def force_convert_16k(path):
    """强制转码为 16000Hz, 16bit, 单声道 PCM"""

    y, _ = librosa.load(path, sr=16000, mono=True)

    return (y * 32767).astype(np.int16).tobytes()


if __name__ == '__main__':

    vpr = IFlyVPR(APPID, APIKey, APISecret)
    pr = IFlyVPR(APPID, APIKey, APISecret)

    # --- 这里是清空逻辑 ---
    print(">>> 步骤0: 正在注销/清空旧的声纹库...")
    del_res = vpr.run("deleteGroup", {
        "groupId": GROUP_ID,
        "deleteGroupRes": {"encoding": "utf8", "compress": "raw", "format": "json"}
    })
    # 注意：如果之前库本身就不存在，可能会返回错误信息，这是正常的，不影响后面创建
    print(f"清空结果: {del_res.get('header', {}).get('message')}")
    # ---------------------------
    print(">>> 步骤1: 创建声纹特征库...")

    res = vpr.run("createGroup", {

        "groupId": GROUP_ID,

        "groupName": "MorpheusUsers",

        "createGroupRes": {"encoding": "utf8", "compress": "raw", "format": "json"}

    })

    print(f"结果: {res}")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    DATA_DIR = os.path.join(ROOT_DIR, "speaker_samples")

    if os.path.exists(DATA_DIR):

        for vip_id in os.listdir(DATA_DIR):

            user_path = os.path.join(DATA_DIR, vip_id)

            if os.path.isdir(user_path):

                print(f"\n>>> 步骤2: 注册 VIP ID {vip_id} 的声纹...")

                for wav in [f for f in os.listdir(user_path) if f.endswith('.wav')]:
                    print(f"  正在处理样本: {wav}...")

                    audio_data = force_convert_16k(os.path.join(user_path, wav))

                    reg_res = vpr.run("createFeature", {

                        "groupId": GROUP_ID,

                        "featureId": str(vip_id),

                        "createFeatureRes": {"encoding": "utf8", "compress": "raw", "format": "json"}

                    }, audio_bytes=audio_data)

                    print(f"  状态: {reg_res.get('header', {}).get('message')}")

    else:

        print(f"错误: 未找到 {DATA_DIR} 目录")