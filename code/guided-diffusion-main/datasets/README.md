下载数据集

此目录包含了用于在此代码库中下载ImageNet和LSUN卧室数据集的说明和脚本。

类别条件的ImageNet

对于我们的类别条件模型，我们使用官方的ILSVRC2012数据集，进行了手动中心裁剪和降采样。要获取此数据集，请访问image-net.org上的此页面并登录（如果您还没有帐户，请创建一个）。然后点击"Training images (Task 1 & 2)"的链接。这是一个大小为138GB的tar文件，包含1000个子tar文件，每个子文件对应一个类别。

下载完成后，请解压缩文件并查看其中的内容。您应该会看到1000个.tar文件。您需要逐个解压缩这些文件，但在您的操作系统上手动执行这个操作可能会不太实际。在基于Unix的系统上，您可以进入目录并运行以下简短的shell脚本来自动化这个过程：

```shell
for file in *.tar; do tar xf "$file"; rm "$file"; done
```

这将逐个解压缩并删除每个tar文件。

一旦所有图像都被解压缩，生成的目录应该可以用作数据目录（训练脚本的--data_dir参数）。文件名应该都以WNID（类别标识）开头，后面是下划线，如n01440764_2708.JPEG。方便的是（但不是偶然的），这正是自动数据加载程序期望发现类别标签的方式。

LSUN卧室

要下载和预处理LSUN卧室数据集，请克隆GitHub上的fyu/lsun，并运行他们的下载脚本python3 download.py bedroom。结果将是一个名为bedroom_train_lmdb的“lmdb”数据库。您可以将其传递给我们的lsun_bedroom.py脚本，如下所示：

```shell
python lsun_bedroom.py bedroom_train_lmdb lsun_train_output_dir
```

这将创建一个名为lsun_train_output_dir的目录。此目录可以通过--data_dir参数传递给训练脚本。

# Downloading datasets

This directory includes instructions and scripts for downloading ImageNet and LSUN bedrooms for use in this codebase.

## Class-conditional ImageNet

For our class-conditional models, we use the official ILSVRC2012 dataset with manual center cropping and downsampling. To obtain this dataset, navigate to [this page on image-net.org](http://www.image-net.org/challenges/LSVRC/2012/downloads) and sign in (or create an account if you do not already have one). Then click on the link reading "Training images (Task 1 & 2)". This is a 138GB tar file containing 1000 sub-tar files, one per class.

Once the file is downloaded, extract it and look inside. You should see 1000 `.tar` files. You need to extract each of these, which may be impractical to do by hand on your operating system. To automate the process on a Unix-based system, you can `cd` into the directory and run this short shell script:

```
for file in *.tar; do tar xf "$file"; rm "$file"; done
```

This will extract and remove each tar file in turn.

Once all of the images have been extracted, the resulting directory should be usable as a data directory (the `--data_dir` argument for the training script). The filenames should all start with WNID (class ids) followed by underscores, like `n01440764_2708.JPEG`. Conveniently (but not by accident) this is how the automated data-loader expects to discover class labels.

## LSUN bedroom

To download and pre-process LSUN bedroom, clone [fyu/lsun](https://github.com/fyu/lsun) on GitHub and run their download script `python3 download.py bedroom`. The result will be an "lmdb" database named like `bedroom_train_lmdb`. You can pass this to our [lsun_bedroom.py](lsun_bedroom.py) script like so:

```
python lsun_bedroom.py bedroom_train_lmdb lsun_train_output_dir
```

This creates a directory called `lsun_train_output_dir`. This directory can be passed to the training scripts via the `--data_dir` argument.
