#!/bin/bash
# 批量下载测试论文脚本
# 用途: 下载60篇经典论文用于测试语义搜索系统
# 特点: 所有论文混合存放，不预先分类，真实测试自动分类能力

# 注意：不使用 set -e，因为某些下载可能失败，我们希望继续下载其他文件

echo "=========================================="
echo "  文献管理系统 - 测试论文批量下载工具"
echo "=========================================="
echo ""
echo "📌 注意: 所有论文将混合存放在同一目录"
echo "         以便测试自动分类功能的准确性"
echo ""

# 创建单一目录（不分类）
DOWNLOAD_DIR=~/Downloads/test_papers
mkdir -p $DOWNLOAD_DIR
cd $DOWNLOAD_DIR

echo "📁 下载目录: $DOWNLOAD_DIR"
echo ""

# 定义下载函数
download_paper() {
    local url=$1
    local filename=$2
    local name=$3
    local topic=$4

    if [ -f "$filename" ]; then
        echo "  ⏭️  跳过 (已存在): [$topic] $name"
    else
        echo "  ⬇️  下载中: [$topic] $name"
        curl -L -o "$filename" "$url" 2>/dev/null || echo "  ❌ 下载失败: $name"
    fi
}

# ==================== CV 论文 ====================
echo "📘 开始下载计算机视觉CV论文 20篇..."
echo ""

download_paper "https://arxiv.org/pdf/1512.03385" "resnet.pdf" "ResNet" "CV"
download_paper "https://arxiv.org/pdf/1409.1556" "vgg.pdf" "VGG" "CV"
download_paper "https://arxiv.org/pdf/1409.4842" "googlenet.pdf" "GoogLeNet" "CV"
download_paper "https://arxiv.org/pdf/2010.11929" "vit.pdf" "Vision Transformer" "CV"
download_paper "https://arxiv.org/pdf/1506.02640" "yolo.pdf" "YOLO" "CV"
download_paper "https://arxiv.org/pdf/1506.01497" "faster_rcnn.pdf" "Faster R-CNN" "CV"
download_paper "https://arxiv.org/pdf/1703.06870" "mask_rcnn.pdf" "Mask R-CNN" "CV"
download_paper "https://arxiv.org/pdf/1411.4038" "fcn.pdf" "FCN" "CV"
download_paper "https://arxiv.org/pdf/1505.04597" "unet.pdf" "U-Net" "CV"
download_paper "https://arxiv.org/pdf/1406.2661" "gan.pdf" "GAN" "CV"
download_paper "https://arxiv.org/pdf/1411.1784" "cgan.pdf" "Conditional GAN" "CV"
download_paper "https://arxiv.org/pdf/1812.04948" "stylegan.pdf" "StyleGAN" "CV"
download_paper "https://arxiv.org/pdf/1501.00092" "srcnn.pdf" "SRCNN" "CV"
download_paper "https://arxiv.org/pdf/1406.2199" "two_stream.pdf" "Two-Stream CNN" "CV"
download_paper "https://arxiv.org/pdf/1911.05722" "moco.pdf" "MoCo" "CV"
download_paper "https://arxiv.org/pdf/2002.05709" "simclr.pdf" "SimCLR" "CV"
download_paper "https://arxiv.org/pdf/1612.00593" "pointnet.pdf" "PointNet" "CV"
download_paper "https://arxiv.org/pdf/1905.11946" "efficientnet.pdf" "EfficientNet" "CV"
download_paper "https://arxiv.org/pdf/1709.01507" "senet.pdf" "SENet" "CV"
download_paper "https://arxiv.org/pdf/1704.04861" "mobilenet.pdf" "MobileNet" "CV"

echo ""

# ==================== NLP 论文 ====================
echo "📗 开始下载自然语言处理NLP论文 20篇..."
echo ""

download_paper "https://arxiv.org/pdf/1706.03762" "transformer.pdf" "Transformer" "NLP"
download_paper "https://arxiv.org/pdf/1810.04805" "bert.pdf" "BERT" "NLP"
download_paper "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf" "gpt.pdf" "GPT" "NLP"
download_paper "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf" "gpt2.pdf" "GPT-2" "NLP"
download_paper "https://arxiv.org/pdf/1907.11692" "roberta.pdf" "RoBERTa" "NLP"
download_paper "https://arxiv.org/pdf/2003.10555" "electra.pdf" "ELECTRA" "NLP"
download_paper "https://arxiv.org/pdf/1910.10683" "t5.pdf" "T5" "NLP"
download_paper "https://arxiv.org/pdf/1301.3781" "word2vec.pdf" "Word2Vec" "NLP"
download_paper "https://nlp.stanford.edu/pubs/glove.pdf" "glove.pdf" "GloVe" "NLP"
download_paper "https://arxiv.org/pdf/1409.3215" "seq2seq.pdf" "Seq2Seq" "NLP"
download_paper "https://arxiv.org/pdf/1409.0473" "nmt_attention.pdf" "NMT with Attention" "NLP"
download_paper "https://arxiv.org/pdf/1606.05250" "squad.pdf" "SQuAD" "NLP"
download_paper "https://arxiv.org/pdf/1511.08308" "ner_lstm.pdf" "NER with LSTM" "NLP"
download_paper "https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf" "rntn.pdf" "RNTN" "NLP"
download_paper "https://arxiv.org/pdf/1910.13461" "bart.pdf" "BART" "NLP"
download_paper "https://arxiv.org/pdf/1911.00536" "dialogpt.pdf" "DialoGPT" "NLP"
download_paper "https://arxiv.org/pdf/1906.08237" "xlnet.pdf" "XLNet" "NLP"
download_paper "https://arxiv.org/pdf/1911.02116" "xlm.pdf" "XLM" "NLP"
download_paper "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf" "dssm.pdf" "DSSM" "NLP"
download_paper "https://arxiv.org/pdf/2004.05150" "longformer.pdf" "Longformer" "NLP"

echo ""

# ==================== RL 论文 ====================
echo "📕 开始下载强化学习RL论文 20篇..."
echo ""

download_paper "https://arxiv.org/pdf/1312.5602" "dqn.pdf" "DQN" "RL"
download_paper "https://arxiv.org/pdf/1511.06581" "dueling_dqn.pdf" "Dueling DQN" "RL"
download_paper "https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf" "policy_gradient.pdf" "Policy Gradient" "RL"
download_paper "https://arxiv.org/pdf/1602.01783" "a3c.pdf" "A3C" "RL"
download_paper "https://arxiv.org/pdf/1707.06347" "ppo.pdf" "PPO" "RL"
download_paper "https://arxiv.org/pdf/1509.02971" "ddpg.pdf" "DDPG" "RL"
download_paper "https://arxiv.org/pdf/1801.01290" "sac.pdf" "SAC" "RL"
download_paper "https://arxiv.org/pdf/1802.09477" "td3.pdf" "TD3" "RL"
download_paper "https://arxiv.org/pdf/1803.10122" "world_models.pdf" "World Models" "RL"
download_paper "https://arxiv.org/pdf/1809.05214" "mbpo.pdf" "MBPO" "RL"
download_paper "https://arxiv.org/pdf/1706.02275" "maddpg.pdf" "MADDPG" "RL"
download_paper "https://arxiv.org/pdf/1705.05363" "curiosity.pdf" "Curiosity-driven" "RL"
download_paper "https://arxiv.org/pdf/1810.12894" "rnd.pdf" "RND" "RL"
download_paper "https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf" "irl.pdf" "Inverse RL" "RL"
download_paper "https://arxiv.org/pdf/1606.03476" "gail.pdf" "GAIL" "RL"
download_paper "https://arxiv.org/pdf/1609.05140" "option_critic.pdf" "Option-Critic" "RL"
download_paper "https://arxiv.org/pdf/1707.01495" "hindsight_er.pdf" "Hindsight ER" "RL"
download_paper "https://arxiv.org/pdf/1712.01815" "alphazero.pdf" "AlphaZero" "RL"
download_paper "https://arxiv.org/pdf/2006.04779" "cql.pdf" "CQL" "RL"
download_paper "https://arxiv.org/pdf/1706.03741" "rlhf.pdf" "RLHF" "RL"

echo ""
echo "✅ 所有论文下载完成"
echo ""

# 统计下载结果
echo "=========================================="
echo "📊 下载统计"
echo "=========================================="
TOTAL=$(find . -maxdepth 1 -name "*.pdf" | wc -l | tr -d ' ')

echo "总计下载: $TOTAL / 60 篇论文"
echo ""
echo "📁 下载位置: $DOWNLOAD_DIR"
echo ""
