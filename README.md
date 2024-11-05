



本节我们简要介绍如何基于 transformers、peft 等框架，对 ChatGLM3-6B-chat 模型进行 Lora 微调。

本节所讲述的代码脚本在同级目录 [ChatGLM3-6B-chat Lora 微调](./06-ChatGLM3-6B-Lora微调.py) 下，运行该脚本来执行微调过程，但注意，本文代码未使用分布式框架，微调 ChatGLM3-6B-Chat 模型至少需要 21G 及以上的显存，且需要修改脚本文件中的模型路径和数据集路径。



## 环境配置

在完成基本环境配置和本地模型部署的情况下，你还需要安装一些第三方库，可以使用以下命令：

```bash
pip install transformers==4.37.2
pip install peft==0.4.0.dev0
pip install datasets==2.10.1
pip install accelerate==0.21.0
```



## 指令集构建

LLM 的微调一般指指令微调过程。所谓指令微调，是说我们使用的微调数据形如：

```json
    {
        "instruction": "这段时间去上厕所本来想小便的可是每次都会拉大便",
        "input": "",
        "output": "这可能是因为你的饮食习惯或者消化系统的问题导致的。建议你试着调整一下饮食，增加膳食纤维的摄入量，多喝水，避免过度依赖泻药。同时，如果问题持续存在，建议去医院检查一下，排除肠道疾病等可能性。"
    }
```

其中，`instruction` 是用户指令，告知模型其需要完成的任务；`input` 是用户输入，是完成用户指令所必须的输入内容；`output` 是模型应该给出的输出。

即我们的核心训练目标是让模型具有理解并遵循用户指令的能力。因此，在指令集构建时，我们应针对我们的目标任务，针对性构建任务指令集。我们构造的指令形如：

```json
    {
        "instruction": "医生呀！我刚被查出得了白癜风，不知道初期治疗需要多少钱？",
        "input": "",
        "output": "白癜风的治疗费用因个体差异和治疗方案的不同而有所差异。初期治疗主要以口服药物和外用药物为主，费用相对较低，一般几百元左右。但是，如果采用激光治疗、光疗等高端治疗方法，费用会更高。建议您咨询专业医生，根据自己的情况进行治疗方案的选择，同时了解相关的费用情况。"
    }
```
我们所构造的全部指令数据集在根目录下。

## QA和Instruction的区别与联系
QA是指一问一答的形式，通常是用户提问，模型给出回答。而Instruction则源自于Prompt Engineering，将问题拆分为两个部分：Instruction用于描述任务，Input用于描述待处理的对象。

问答（QA）格式的训练数据通常用于训练模型回答基于知识的问题，而指令（Instruction）格式的训练数据更适用于训练模型执行具体任务。例如，对于问题"请解释VC银翘片和双黄连口服液之间的区别"
- 问答（QA）格式:
    ```
    指令（Instruction）： 
    输入（Input）：VC银翘片和双黄连口服液之间的区别是什么？
    ```

- 指令（Instruction）格式:
    ```
    指令（Instruction）：请解释下面两个药品之间的区别。
    输入（Input）：VC银翘片和双黄连口服液。
    ```
指令的形式可能使模型具有更好的泛化能力，因为它强调了任务的性质，而不仅仅是特定的输入。通常指令格式和问答格式可以相互转化。

## 数据格式化

`Lora` 训练的数据是需要经过格式化、编码之后再输入给模型进行训练的，如果是熟悉 `Pytorch` 模型训练流程的同学会知道，我们一般需要将输入文本编码为 input_ids，将输出文本编码为 `labels`，编码之后的结果都是多维的向量。我们首先定义一个预处理函数，这个函数用于对每一个样本，编码其输入、输出文本并返回一个编码后的字典：

```python
def process_func(example):
    MAX_LENGTH = 512
    input_ids, labels = [], []
    instruction = tokenizer.encode(text="\n".join(["<|system|>", "现在你要扮演皇帝身边的女人--甄嬛", "<|user|>", 
                                    example["instruction"] + example["input"] + "<|assistant|>"]).strip() + "\n",
                                    add_special_tokens=True, truncation=True, max_length=MAX_LENGTH)

    response = tokenizer.encode(text=example["output"], add_special_tokens=False, truncation=True,
    max_length=MAX_LENGTH)

    input_ids = instruction + response + [tokenizer.eos_token_id]
    labels = [tokenizer.pad_token_id] * len(instruction) + response + [tokenizer.eos_token_id]
    pad_len = MAX_LENGTH - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad_len
    labels += [tokenizer.pad_token_id] * pad_len
    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

    return {
        "input_ids": input_ids,
        "labels": labels
    }
```



## 加载tokenizer和半精度模型

模型以半精度形式加载，如果你的显卡比较新的话，可以用`torch.bfolat`形式加载。对于自定义的模型一定要指定`trust_remote_code`参数为`True`。

```python
tokenizer = AutoTokenizer.from_pretrained('./model/chatglm3-6b', use_fast=False, trust_remote_code=True)

# 模型以半精度形式加载，如果你的显卡比较新的话，可以用torch.bfolat形式加载
model = AutoModelForCausalLM.from_pretrained('./model/chatglm3-6b', trust_remote_code=True, torch_dtype=torch.half, device_map="auto")
```

## 定义LoraConfig

`LoraConfig`这个类中可以设置很多参数，但主要的参数没多少，简单讲一讲，感兴趣的同学可以直接看源码。

- `task_type`：模型类型
- `target_modules`：需要训练的模型层的名字，主要就是`attention`部分的层，不同的模型对应的层的名字不同，可以传入数组，也可以字符串，也可以正则表达式。
- `r`：`lora`的秩，具体可以看`Lora`原理
- `lora_alpha`：`Lora alaph`，具体作用参见 `Lora` 原理 
- `modules_to_save`指定的是除了拆成lora的模块，其他的模块可以完整的指定训练。

`Lora`的缩放是啥嘞？当然不是`r`（秩），这个缩放就是`lora_alpha/r`, 在这个`LoraConfig`中缩放就是4倍。
这个缩放的本质并没有改变LoRa的参数量大小,本质在于将里面的参数数值做广播乘法,进行线性的缩放。

```python
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["query_key_value"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1# Dropout 比例
)
```

## 自定义 TrainingArguments 参数

`TrainingArguments`这个类的源码也介绍了每个参数的具体作用，当然大家可以来自行探索，这里就简单说几个常用的。

- `output_dir`：模型的输出路径
- `per_device_train_batch_size`：顾名思义 `batch_size`
- `gradient_accumulation_steps`: 梯度累加，如果你的显存比较小，那可以把 `batch_size` 设置小一点，梯度累加增大一些。
- `logging_steps`：多少步，输出一次`log`
- `num_train_epochs`：顾名思义 `epoch`
- `gradient_checkpointing`：梯度检查，这个一旦开启，模型就必须执行`model.enable_input_require_grads()`，这个原理大家可以自行探索，这里就不细说了。

```python
# Data collator GLM源仓库从新封装了自己的data_collator,在这里进行沿用。

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=None,
    padding=False
)

args = TrainingArguments(
    output_dir="./output/ChatGLM",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=10,
    num_train_epochs=3,
    gradient_checkpointing=True,
    save_steps=100,
    learning_rate=1e-4,
)
```

### 使用 Trainer 训练

把 model 放进去，把上面设置的参数放进去，数据集放进去，OK！开始训练！

```python
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=data_collator,
)
trainer.train()
```

## 模型推理

可以用这种比较经典的方式推理。

```python
model.eval()
model = model.cuda()
ipt = tokenizer("<|system|>\n现在你要扮演皇帝身边的女人--甄嬛\n<|user|>\n {}\n{}".format("你是谁？", "").strip() + "<|assistant|>\n", return_tensors="pt").to(model.device)
tokenizer.decode(model.generate(**ipt, max_length=128, do_sample=True)[0], skip_special_tokens=True)
```

## 加载
通过PEFT所微调的模型，都可以使用下面的方法进行重新加载，并推理:
- 加载源model与tokenizer；
- 使用`PeftModel`合并源model与PEFT微调后的参数。

```python
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("./model/chatglm3-6b", trust_remote_code=True, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("./model/chatglm3-6b", use_fast=False, trust_remote_code=True)

p_model = PeftModel.from_pretrained(model, model_id="./output/ChatGLM/checkpoint-1000/")  # 将训练所得的LoRa权重加载起来
## 得到的p_model是原来模型和微调后的Lora模型合并的结果

ipt = tokenizer("<|system|>\n现在你要扮演皇帝身边的女人--甄嬛\n<|user|>\n {}\n{}".format("你是谁？", "").strip() + "<|assistant|>\n", return_tensors="pt").to(model.device)
tokenizer.decode(p_model.generate(**ipt, max_length=128, do_sample=True)[0], skip_special_tokens=True)

```
