# 把本地研究仓库上传到GitHub

## 推荐步骤

1. 登录GitHub，创建名为`kforcing-adaptive-research`的空仓库。建议先设为Private，便于与合作者整理材料。若选择公开，仓库内容将公开可见。
2. 创建时不要初始化README、.gitignore或LICENSE，因为本地已经有文件和目录。
3. 在本目录打开PowerShell，用`git status`和`git log -1`检查本地状态。若已有初始提交，可以直接执行第5步。若尚未初始化，执行`git init -b main`。
4. 仅在尚无初始提交时，用你自己的Git作者身份提交文件：

```powershell
git add .
git commit -m "Add reproducible K-Forcing adaptive decoding research"
```

5. 配置刚建仓库的URL并推送：

```powershell
git remote add origin https://github.com/YOUR_USERNAME/kforcing-adaptive-research.git
git push -u origin main
```

把示例中的用户名及仓库名换成真实值。若Git提示作者身份缺失，在本仓库配置你自己的`user.name`和`user.email`，不要使用示例或他人身份。GitHub登录由本机Git认证流程完成，不把密码或token写入仓库。

如果使用已有且非空的仓库，应先检查其内容、默认分支和历史，再决定分支或合并方式；不要使用强制推送覆盖已有工作。

本目录初始化不代表已上传。以`git remote -v`、`git status`和GitHub页面确认发布结果。

## 后续更新

研究主工作区仍是原目录，发布目录是一个单独快照。后续新实验经过核查后，将相应源码、报告与结果同步到这里，再提交和推送。不要把原工作区的整个`work`目录或历史ZIP全部拖入。

官方说明：[把本地代码添加到GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)、[创建仓库](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository)。
