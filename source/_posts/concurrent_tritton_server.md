---
title: 修复脚本的在Windows 和Linux 系统下由于换行符导致的问题
date: 2023-07-02 12:31
---

执行某个脚本有如下报错

```shell
./hold_container.sh: line 2: $'\r': command not found
sleep: invalid time interval ‘infinity\r’
Try 'sleep --help' for more information.
./hold_container.sh: line 4: $'\r': command not found
```

这些错误通常是由于文本文件的换行符格式不正确引起的。您的脚本文件似乎包含了 Windows 风格的换行符（\r\n），而不是 Unix 风格的换行符（\n）。在 Unix 系统上运行时，这些额外的回车符会导致错误。

### sed 替换多个文件的换行符

要解决这个问题，您可以将脚本文件中的换行符格式从 Windows 风格转换为 Unix 风格。可以使用以下命令来进行转换：

```shell
sed -i 's/\r$//' *.sh &&\
sed -i 's/\r$//' web_server/*.sh
```

### Git 设置全局 autocrlf 换行配置

修复了上面的问题之后，如果代码在Git管理下，则会出现下面报错

```sh
git warning : The file will have its original line endings in your working directory
```

这个警告是由于在 Git 仓库中，文件的换行符格式与您当前工作目录中的默认换行符格式不匹配导致的。

要解决这个问题，可以尝试以下几个步骤：

确保您的 Git 配置中设置了正确的换行符配置。在终端中运行以下命令：

```sh
git config --global core.autocrlf
```

如果输出为 true，则将换行符自动转换为 Windows 风格的换行符（CRLF），如果输出为 input，则保持换行符不变。

尝试将文件的换行符转换为与您的工作目录中的换行符格式一致。在终端中运行以下命令：

```sh
git add --renormalize .

git commit -m "Fix line endings"
git push
```

执行上述步骤后，Git 应该会自动处理文件的换行符，并消除警告。但请注意，这些操作可能会影响到整个仓库，因此请确保在执行之前备份重要的文件。
