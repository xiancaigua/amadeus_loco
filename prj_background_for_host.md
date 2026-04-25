# 宿主机开发智能体环境说明（Isaac Lab 工程协调代理）

你当前运行在服务器宿主机上，而不是 Isaac Sim Docker 容器内。

## 核心原则
- Isaac 仿真必须在 Docker 内运行
- 代码编辑可以在宿主机完成
- 调试必须进入容器执行

---

## 1. 系统架构

### 宿主机
- 代码管理
- Docker 控制
- 网络代理

### Docker
- Isaac Sim
- Isaac Lab
- 训练与仿真

---

## 2. 项目路径

- IsaacLab: /workspace/IsaacLab
- 项目: /workspace/amadeus/amadeus

---

## 3. 进入容器调试（关键）

### 查看容器
docker ps

### 进入容器
docker exec -it isaac-zzh /bin/bash

### 进入项目
cd /workspace/amadeus/amadeus

### 设置环境
alias python=/isaac-sim/python.sh
alias pip="/isaac-sim/python.sh -m pip"

---

## 4. 调试命令

### 查看环境
python scripts/list_envs.py

### 测试导入
python -c "import isaaclab; print('ok')"

### 启动训练
python scripts/rl_games/train.py --task Template-Amadeus-v0

---

## 5. 行为准则

- 不要在宿主机运行 Isaac
- 不要修改 /isaac-sim
- 不要重装环境

---

## 总结

写代码在宿主机  
运行代码在 Docker  
调试必须 docker exec
