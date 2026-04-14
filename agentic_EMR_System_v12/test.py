import requests

url = "https://u949345-bb39-00b1d78c.westd.seetacloud.com:8443/rerank"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer 8f6b1b7d2d8f4d0cb2f52b3d7dcb9f6e4a7c0d3e6f8a1b2c",
}
ls = [
    "反酸：胃内容物反流至食管或口腔，出现酸水上涌、口中发酸或胸口返酸感。",
    "吞咽困难：食物或液体通过咽喉或食管时出现梗阻、停滞、下行不畅或噎住感。",
    "恶心：上腹部不适并伴想吐的感觉，可发生于晨起、进食后或闻到气味时。",
    "食物反刍：进食后不久，未完全消化的食物不自主返入口腔，患者常会重新咀嚼再咽下，通常无明显恶心。"
]
payload = {
    "query": "吃完的东西感觉一会就又反上来了，我都会再咀嚼一下再咽下去",
    "documents": ls,
}

resp = requests.post(url, headers=headers, json=payload, timeout=30)
print(resp.status_code)

# 解析返回的JSON数据
result = resp.json()
scores = result["scores"]

# 找到最高分对应的索引
max_score_index = scores.index(max(scores))

# 获取最高分对应的文本
best_match = ls[max_score_index]

# 打印结果
print("最高得分：", max(scores))
print("最匹配的结果：", best_match)