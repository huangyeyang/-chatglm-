1、准备工作
首先，确保您的开发环境中已经安装了Python3.9或更高版本。这可以通过Python的官方网站下载安装，或者使用包管理器进行安装。

1）显卡选择
24 GB显存的A10：建议使用至少这个规格的实例，或者更高规格的实例以满足可能更大的计算需求。
2）镜像选择：
PyTorch深度学习框架版本为2.1.2。Python 3.10、CUDA 11.2（cu121），CUDA是NVIDIA提供的用于通用并行计算的编程模型和API。、Ubuntu 22.04 LTS（长期支持版本）操作系统。

2、获取LLaMA-Factory
打开您的终端或命令行界面，然后执行以下命令来克隆LLaMA-Factory的代码仓库到本地：

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git

这将创建一个名为LLaMA-Factory的文件夹，包含所有必要的代码和文件。

3、安装依赖
在安装LLaMA-Factory之前，您需要确保安装了所有必要的依赖。进入克隆的仓库目录，然后执行以下命令来安装依赖：

cd LLaMA-Factory
pip install -e .[metrics]

这个命令将安装LLaMA-Factory及其所有必需的附加组件，用于模型的评估和分析。

4、卸载可能冲突的包
如果在安装过程中与其他库发生冲突，您可能需要先卸载这些库。例如，如果vllm库与LLaMA-Factory不兼容，可以使用以下命令卸载：

pip uninstall -y vllm

5、LLaMA-Factory版本检查
安装完成后，您可以通过运行以下命令来检查LLaMA-Factory是否正确安装以及其版本号：

llamafactory-cli version

如果安装成功，您将看到类似以下的输出，显示LLaMA Factory的版本信息

6、验证安装
为了确保LLaMA Factory能够正常工作，您可以运行一些基本的命令来测试其功能。例如，尝试运行LLaMA Factory提供的一些示例脚本，或者使用其命令行界面来查看帮助信息：

llamafactory-cli --help

模型微调
1启动Web UI
使用以下命令启动LLaMA-Factory的Web UI界面，以便进行交互式模型微调：

export USE_MODELSCOPE_HUB=1 && llamafactory-cli webui

这将启动一个本地Web服务器，您可以通过访问http://0.0.0.0:7860来使用Web UI。请注意，这是一个内网地址，只能在当前实例内部访问。

2、配置参数
在Web UI中，您需要配置以下关键参数以进行模型微调：
语言：选择模型支持的语言，例如zh。
模型名称：选择要微调的模型，例如LLaMA3-8B-Chat。
微调方法：选择微调技术，如lora。
数据集：选择用于训练的数据集。
学习率：设置模型训练的学习率。
计算类型：根据GPU类型选择计算精度，如bf16或fp16。
梯度累计：设置梯度累计的批次数。
LoRA+学习率比例：设置LoRA+的相对学习率。
LoRA作用模块：选择LoRA层挂载的模型部分。






