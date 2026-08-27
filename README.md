# 网球挥拍测评 2.0

上传训练录像，生成评分和练习建议。

## 线上

- HTTPS（Caddy + sslip.io，与 knx 同模式）：https://tennis.47.93.203.28.sslip.io/
- IP 入口（nginx）：http://47.93.203.28/tennis-ai/

服务器目录 `/opt/tennis-ai`。更新：在机器上执行 `deploy/pull.sh`（不覆盖 `.env`、样例视频、模型权重）。

## 本地运行

```bash
cp .env.example .env
# 填入点评服务与对象存储配置
./run_web.sh
# 打开 http://127.0.0.1:27116
```

把侧面样例视频放到 `samples/demo.mp4` 后，可点「分析 1 分钟样例」（只分析前 60 秒）。自己上传的视频按整段分析。能看到球或球拍时，击球画面会按球和拍的距离选取。

2.0 评分四维：重心（稳且低）、击球点（身体侧前方）、动力链（伤病相关核心）、击球效果（拍头速度与旋转轨迹）。

## 环境变量

点评服务：

- `CURSOR_API_KEY`
- `CURSOR_SANDBOX_REPO_URL`
- `CURSOR_MODEL_ID`（默认 `grok-4.6`）
- `CURSOR_MODEL_EFFORT`（默认 `high`；点评任务不需要 xhigh）
- `CURSOR_REUSE_AGENT`（默认 `1`，复用已启动的点评会话）
- `CURSOR_AGENT_TIMEOUT_MS`（默认 `180000`）

对象存储（与 english-test 相同 bucket / prefix）：

- `OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET`
- `OSS_BUCKET`（默认 `nba-dev-sh`）
- `OSS_PREFIX`（默认 `wenbo`，对象在 `{prefix}/tennis-ai/{job_id}/`）
- `OSS_REGION` / `OSS_ENDPOINT`
- `OSS_URL_MODE=signed`
- `OSS_SIGNED_URL_SECONDS`（默认 7 天）

未配置或调用失败时任务会直接报错，不会改用本地文案或本地文件地址。
