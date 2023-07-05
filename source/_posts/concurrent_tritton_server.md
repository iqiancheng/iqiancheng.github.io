---
title: 修改 Tritton Server 进行批量推理
date: 2023-07-03 20:33
---

### 配置修改

```json
max_batch_size: 1024
dynamic_batching {
    max_queue_delay_microseconds: 100000
}
```

### 修改input_ids的padding拼接

`web_server.py`  L111行修改`input_ids` padding 逻辑为`3 (</s>)`

```python
input_ids = tokenizer(self.prompt, add_special_tokens=True, return_tensors="pt", padding=False)
logging.info("input tokens cnt:" + str(len(input_ids[0])))
input_ids = input_ids["input_ids"]
```



### 客户端模拟多线程批量发起请求

`bench.py` 文件中修改多线程请求部分L152~159：

```python
start = time.time()
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for i in range(0, len(texts), worker_count):                
                futures = []
                for j in range(worker_count):
                    if i+j < len(texts):
                        futures.append(executor.submit(request, texts[i+j]))
                concurrent.futures.wait(futures)
                print("-----------------Finished Batch-----------------")
            executor.shutdown()
    end = time.time()
```



### Check 是否成功批量

```bash
# 用grep 过滤日志文件中存在 batch_size 的前后3行
grep -C 3 "batch_size" logfiles.log
```

如果你想要使用 `grep` 来查找包含特定模式的行，并且显示这些行的前后三行，你可以使用 `-A` (after)，`-B` (before)，和 `-C` (context，表示前后都要显示的行数) 这些选项。
