import requests
from utils import *
import os
import asyncio
from websockets.asyncio.client import connect
import json
import time
import shutil
import random


base_url = "0.0.0.0:12244"


###########################################
###########################################

res = requests.get(f"http://{base_url}/speakers")
print(res.json())


###########################################
###########################################


def gen_text(text, max_len=5):
    while len(text) > 0:
        l = random.randint(1, max_len)
        time.sleep(random.random() * 0.07)
        yield text[:l]
        text = text[l:]


async def tts(text, prompt_id, tts_index):
    async with connect(f"ws://{base_url}/tts") as websocket:

        async def send_msg():
            await websocket.send(
                json.dumps(
                    {
                        "req_params": {
                            "prompt_id": prompt_id,
                            "audio_format": "wav",
                            "sample_rate": 24000,
                            "instruct_text": "",
                        }
                    }
                )
            )

            for t in gen_text(text):
                await websocket.send(json.dumps({"text": t, "done": False}))
            await websocket.send(json.dumps({"text": "", "done": True}))

        asyncio.create_task(send_msg())

        whole_audio = []
        root = None
        resample_rate = 24000
        while True:
            s = time.time()
            message = await websocket.recv()
            message = json.loads(message)

            if message["error"]:
                print(f"{tts_index} {message}")

            elif not message["error"] and not message["is_end"]:
                print(f"{tts_index}-{message['index']} REQ TIME:", time.time() - s)

                chunk = any_format_to_ndarray(
                    message["data"],
                    message["audio_format"],
                    message["sample_rate"],
                    resample_rate,
                )
                whole_audio.append(chunk)

                root = f"results/{tts_index}_{prompt_id}_{message['id']}"
                os.makedirs(f"{root}/chunks", exist_ok=True)

                save_audio(
                    chunk,
                    f"{root}/chunks/{message['index']}.wav",
                    resample_rate,
                )

            elif message["is_end"]:
                print(f"{tts_index} finished...")
                break

        save_audio(np.concatenate(whole_audio), f"{root}/whole.wav", resample_rate)


texts = [
    """《404病房》
    护士站的值班表上并没有404号病房。但凌晨三点，我分明听见走廊尽头传来规律的滴水声。
    白大褂被冷汗浸透时，我握着手电推开了那扇漆皮剥落的铁门。
    生锈的轮椅在月光下空转，霉味里混着福尔马林之外的腥甜。镜面碎裂的洗手池滴答作响，每声都精准卡在心跳间隙。
    当我打开最里侧的储物柜，整排玻璃药瓶突然同时炸裂，飞溅的碎片却在半空诡异地凝滞成某种符号。
    镜中倒影忽然眨了眨眼——那不是我。苍白的手指从背后缠上脖颈时，我听见病历卡散落的声音。
    最后一张泛黄的纸片上，二十年前的潦草笔迹写着我的名字，诊断栏里爬满黑虫般的字迹：该患者坚称在404病房工作。
    晨光刺破窗棂时，巡逻保安在废弃仓库发现了十三支空镇静剂。监控录像里，我整夜都坐在布满灰尘的镜子前，对着空气微笑。""",
    ##############################
    """《午夜镜像》
    电梯在13楼停下时，显示屏分明跳动着"12"。走廊尽头的古董穿衣镜是房东特意叮嘱不能挪动的，此刻镜面却浮着层水雾，把月光滤成尸斑似的青灰。
    我第37次擦去雾气时，掌纹突然在镜中扭曲成陌生纹路。镜框雕花的藤蔓缠住手腕刹那，整栋楼的声控灯同时爆裂。
    黑暗中，镜里的我扬起嘴角，而真正的我正死死咬住嘴唇。
    保安第二天发现镜子碎成蛛网状，每块碎片都映着不同时间的我：03:15分惊恐后退，03:17分脖颈后仰成诡异角度，03:19分的碎片却空无一人。
    监控显示我整夜都紧贴镜面站立，指尖在玻璃上重复书写着1937年的日期。
    如今租客们总抱怨13楼有面擦不干净的镜子。偶尔有醉鬼看见穿真丝睡裙的女人在镜前梳头，发梢滴落的不知是水还是血——那件染血的睡裙，
    此刻正整整齐齐叠在我的衣柜底层。""",
]
prompt_ids = [
    # "teemo",
    "twitch",
]


async def main(task_count=1):
    shutil.rmtree("./results", ignore_errors=True)

    tasks = []
    for i in range(task_count):
        tasks.append(tts(random.choice(texts), random.choice(prompt_ids), i))
    await asyncio.gather(*tasks)


asyncio.run(main())
