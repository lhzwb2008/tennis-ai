# 网球挥拍测评

上传训练录像，生成评分和练习建议。

## 本地运行

```bash
cp .env.example .env
# 填入点评服务与对象存储配置
./run_web.sh
# 打开 http://127.0.0.1:27116
```

把样例视频放到 `samples/demo.mp4` 后，可点「分析 1 分钟样例」（只分析前 60 秒）。自己上传的视频按整段分析。

## 环境变量

点评服务：

- `CURSOR_API_KEY`
- `CURSOR_SANDBOX_REPO_URL`
- `CURSOR_MODEL_ID`（默认 `grok-4.6`）
- `CURSOR_MODEL_EFFORT`（默认 `xhigh`）

对象存储（与 english-test 相同 bucket / prefix）：

- `OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET`
- `OSS_BUCKET`（默认 `nba-dev-sh`）
- `OSS_PREFIX`（默认 `wenbo`，对象在 `{prefix}/tennis-ai/{job_id}/`）
- `OSS_REGION` / `OSS_ENDPOINT`
- `OSS_URL_MODE=signed`
- `OSS_SIGNED_URL_SECONDS`（默认 7 天）

未配置或调用失败时任务会直接报错，不会改用本地文案或本地文件地址。
