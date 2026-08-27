# 网球挥拍测评

上传训练视频：先做 2D 姿态与挥拍检测，再用 **Cursor Cloud Agent（Grok 4.6 Extra High）** 结合关键帧写评。

调用方式与 `../AIVideo` 的 `cursor_client.py` 相同：`POST /v1/agents`，`model.id = grok-4.6`，`params.effort = xhigh`。

## 本地运行

```bash
cp .env.example .env   # 填 CURSOR_API_KEY 与 CURSOR_SANDBOX_REPO_URL
./run_web.sh
# 打开 http://127.0.0.1:8766
```

把样例视频放到 `samples/demo.mp4` 后，可点「分析样例视频」。

## 环境变量

- `CURSOR_API_KEY`：Cursor Dashboard → Integrations
- `CURSOR_SANDBOX_REPO_URL`：Cloud Agent 挂载的 GitHub 仓库
- `CURSOR_MODEL_ID`：默认 `grok-4.6`
- `CURSOR_MODEL_EFFORT`：默认 `xhigh`（Extra High）

未配置密钥时，报告会回退到规则引擎文案。
