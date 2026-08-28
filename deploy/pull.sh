# 在服务器上：git pull 代码并重启（不碰 .env、样例视频、模型）
# 用法: ssh root@47.93.203.28 'bash -s' < deploy/pull.sh

set -euo pipefail
cd /opt/tennis-ai
git fetch origin
git pull --ff-only origin main
.venv/bin/pip install -q -r requirements.txt
systemctl restart tennis-ai
sleep 2
systemctl is-active tennis-ai
curl -sf http://127.0.0.1:27116/api/health
echo
