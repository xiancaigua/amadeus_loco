# Docker 内开发智能体环境说明（Isaac Sim / Isaac Lab / Amadeus）

你当前运行在一个用于 Isaac Sim / Isaac Lab 开发与训练的 Docker 容器中。请严格按照以下环境背景理解当前工作空间，并在执行任务时遵循这些约束。

## 1. 目标与角色

你的主要职责是：
- 在当前容器内帮助开发、修改、调试 Isaac Lab 相关代码
- 处理 Amadeus 项目的任务定义、环境配置、训练脚本、数据采集逻辑
- 执行与 Isaac Sim / Isaac Lab 强相关的命令
- 在修改代码时尽量保持最小改动、可运行、可验证

你不是一个通用系统维护代理。除非任务明确要求，否则不要大规模改系统配置，不要随意升级环境，不要破坏 Isaac Sim 运行时。

---

## 2. 当前环境结构

当前容器内关键路径如下：

- Isaac Sim 根目录：
  `/isaac-sim`

- Isaac Lab 根目录：
  `/workspace/IsaacLab`

- 用户自定义模板项目 Amadeus 根目录：
  `/workspace/amadeus/amadeus`

当前项目已经通过模板生成器创建，并已安装为 editable package。

---

## 3. Python / Pip 使用规范

这是最重要的规则之一：

### 不要使用系统 python
容器中普通 `python` 命令默认可能不存在，或者即使存在，也不应优先使用。

### 必须优先使用 Isaac Sim 自带 Python
标准调用方式是：

- Python：
  `/isaac-sim/python.sh`

- Pip：
  `/isaac-sim/python.sh -m pip`

如果当前 shell 已经配置了 alias：
- `python -> /isaac-sim/python.sh`
- `pip -> /isaac-sim/python.sh -m pip`

那么可以直接使用 `python` / `pip`，但你必须先确认 alias 已生效。

### 原因
Isaac Sim 依赖专用 runtime、扩展库、环境变量和 Kit 启动逻辑。使用系统 python 很可能导致：
- `ModuleNotFoundError`
- Isaac 扩展加载失败
- Task registry 丢失
- 仿真无法启动

---

## 4. 当前已确认成功的状态

以下状态已经确认正常，不要重复做无意义初始化：

- Isaac Lab 主体已安装成功
- `isaaclab_rl` 已安装成功
- `isaaclab_mimic` 已安装成功
- `isaaclab_tasks` 已安装成功
- `isaaclab_assets` 已安装成功
- `isaaclab_contrib` 已安装成功
- 自定义项目 `amadeus` 已成功安装：
  `amadeus==0.1.0`
- `python -c "import isaaclab; print('ok')"` 已通过
- `python scripts/list_envs.py` 已成功列出自定义环境

---

## 5. 当前已注册成功的任务

以下任务已经在 registry 中可见：

- `Template-Amadeus-Direct-v0`
- `Template-Amadeus-Marl-Direct-v0`
- `Template-Amadeus-v0`

这说明：
- 自定义 package 安装成功
- task entry-point 配置正确
- registry 加载正常

除非用户明确要求，不要再重复排查“为什么任务没注册”。

---

## 6. 网络与代理背景

当前容器内访问 GitHub 曾经遇到过严重网络问题，后已通过“本地电脑代理 + SSH 反向端口转发”解决。

当前已验证成功的方式是：
- 宿主机与容器均可访问 GitHub
- `git ls-remote https://github.com/isaac-sim/rl_games.git` 已成功

通常需要在当前 shell 中设置：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897