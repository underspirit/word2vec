//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http:// www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100	// 一个词的最大长度
#define EXP_TABLE_SIZE 1000	// exp查找表的元素个数,(-6,6)进行1000等分
#define MAX_EXP 6	// exp查找表的阈值
#define MAX_SENTENCE_LENGTH 1000	// 定义最大的句子长度
#define MAX_CODE_LENGTH 40	// 定义最长的霍夫曼编码长度

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;	// 词频
  int *point;	// 编码路径
  char *word,	// 词
  *code,	// 编码
  codelen;	// 编码长度
};

char train_file[MAX_STRING],	// 保存 训练文本 文件名
	output_file[MAX_STRING];	// 保存 输出词向量文件名
char save_vocab_file[MAX_STRING],	// 保存词典到文件
	read_vocab_file[MAX_STRING];	// 读取词典文件,而不是从训练数据生成词典
struct vocab_word *vocab;	// 词典
int binary = 0,	// 表示输出的结果文件是否采用二进制存储，0表示不使用（即普通的文本存储，可以打开查看），1表示使用，即vectors.bin的存储类型
	cbow = 1,	// 是否使用cbow模型，0表示使用skip-gram模型，1表示使用cbow模型，默认情况下是skip-gram模型
	debug_mode = 2,	// 大于0，加载完后，输出汇总信息；大于1，加载训练词汇的时候输出信息，训练过程中输出信息
	window = 5,	// 训练的窗口大小
	min_count = 5,	// 表示设置最低频率，默认为5，如果一个词语在文档中出现的次数小于该阈值，那么该词就会被舍弃
	num_threads = 12,	// 线程个数
	min_reduce = 1;	// 构建词典时,词频小于该值则舍弃
int *vocab_hash;	// 词典哈希表，下标是词的hash，值是词在vocab中的索引
long long vocab_max_size = 1000,//词典的最大长度，可以扩增，每次扩1000
		vocab_size = 0,	// 词典大小,当前处理的词的序号-----------词典的当前大小，当接近vocab_max_size的时候会扩充
		layer1_size = 100;	// 词向量纬度--------隐藏层节点数
long long train_words = 0,	// 训练文本中的单词总数（词频累加）
		word_count_actual = 0,	// 已经训练完的word个数
		iter = 5,	// 训练迭代次数
		file_size = 0,	// 训练文件大小，ftell得到
		classes = 0;	// 输出词类别而不是词向量,默认输出词向量
real alpha = 0.025,	// 学习率，训练过程自动调整
		starting_alpha,	// 初始学习率,从alpha开始自适应变化
		sample = 1e-3;	// 采样的阈值，如果一个词在训练样本中出现的频率越大，那么就越会被采样
real *syn0,	// 输入词向量,所有词都存在里面,可看出[vacab_size][layer1_size]的二维数组
	*syn1,	// binary tree非叶节点的参数向量
	*syn1neg,	// NEG时,负样本对应的词向量,为辅助参数向量
	*expTable;	// 自然指数查找表
clock_t start;//算法运行的开始时间，会用于计算平均每秒钟处理多少词

int hs = 0,	// 是否使用HS方法，0表示不使用，1表示使用
	negative = 5;	// 表示是否使用NEG方法,指定负采样数量
const int table_size = 1e8;	// table_size 静态负采样表的规模
int *table;	// 负采样表

// 根据词频生成负采样表用于negative sampling
void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;//训练文本每个词的词频的0.75次方的累加和
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);	// 求出总的计数
  i = 0;//词在词典中的索引
  d1 = pow(vocab[i].cn, power) / train_words_pow;//第一个词的概率长度，后面用于对每个词的概率进行累加，最后为1.有点像轮盘赌算法
  for (a = 0; a < table_size; a++) {	// 得出映射关系-----进行采样等到采样表
    table[a] = i;
    if (a / (double)table_size > d1) {//若等分点，大于当前词的概率
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
//每次从fin中读取一个单词
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {	// 文件未结束
    ch = fgetc(fin);
    if (ch == 13) continue;	// 13是回车,\r
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");	// 是换行符,则用</s>替换
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;//标识字符数组的结束‘\0’
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
// 返回一个词在词典中的位置,如果不存在则返回-1--------采用线性探测法解决hash冲突
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];	// 相等则返回所在位置
    hash = (hash + 1) % vocab_hash_size;	// 不相等则线性探测
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
// 读取一个词,返回它在词典中的索引
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
// 添加一个词到词典中
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;//在调用函数中赋值为1。
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {//词典大小接近vocab_max_size时，扩容1000
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);	// 获得hash
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;	// 线性探测开放定制法
  vocab_hash[hash] = vocab_size - 1;	// 指向处理的词的索引
  return vocab_size - 1;//返回添加的单词在词典中的位置
}

// Used later for sorting by word counts 比较函数，按词频进行排序
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
// 将词典按照词频降序排序
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  // 将</s>放在第一个位置------</s>表示回车
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);//词典进行快速排序，第一个是回车不参与排序
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;//词典已经重排，需重建vocab_hash
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
	// 词频小于min_count的将被删除
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual,重新计算hash
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;	// 重新计算训练词数量
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));//分配的多余空间收回
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));//给哈弗曼树编码和路径分配空间
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
// 通过删除低频词缩小词典
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
// 通过词频构建Huffman树
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));	// 存储树中节点的计数值
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));	// 存储树节点的编码
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));	// 存储树节点的父节点
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;	// 初始化有vacab_size棵树的森林
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;	// 后半部分用来存储合成两个节点后的父节点.先初始化为一个大数值
  pos1 = vocab_size - 1;	// 指针pos1从中间往前走
  pos2 = vocab_size;	// pos2从中间往后走
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];	// 合成新节点
    parent_node[min1i] = vocab_size + a;	// 设置父节点
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;	// 两个最小的节点中,大的那个编码为1,小的为0
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];	// 编码复制
      point[i] = b;	// 路径赋值
      i++;	// 编码个数
      b = parent_node[b];	// 找到父节点
      if (b == vocab_size * 2 - 2) break;	// 到达根节点,跳出
    }
    vocab[a].codelen = i;	// 编码长度赋值，少1，没有算根节点
    vocab[a].point[0] = vocab_size - 2;	// TODO?? 逆序，把第一个赋值为root（即2*vocab_size - 2 - vocab_size）
    for (b = 0; b < i; b++) {	// 逆序处理
      vocab[a].code[i - b - 1] = code[b];	// 编码逆序，没有根节点，左子树0，右子树1
      vocab[a].point[i - b] = point[b] - vocab_size;	// TODO?? 其实point数组最后一个是负的，用不到，point的长度是编码的真正长度，比code长1
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

/**
 * 从训练数据中构造词典
 */
void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;	// 初始化hash表
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {	// 当前词典不存在该词则添加
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;	// 词频加1
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();	// 词典过大,则缩减低频词
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);	// 指针到文件末尾
  file_size = ftell(fin);	// 返回当前指针位置,即文件大小
  fclose(fin);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  // 申请词向量空间
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (hs) {	// 使用层次softmax,申请binary tree的参数空间
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)	// 0初始化
     syn1[a * layer1_size + b] = 0;
  }
  if (negative>0) {	// 使用negative sampling
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;
  }
  // 随机初始化词向量
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }
  CreateBinaryTree();
}

// 训练线程
void *TrainModelThread(void *id) {
  // word 向sen中添加单词用，句子完成后表示句子中的当前单词
  // last_word 上一个单词，辅助扫描窗口
  // sentence_length 当前句子的长度（单词数）
  // sentence_position 当前单词在当前句子中的index
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  // word_count 已训练语料总长度
  // last_word_count 保存值，以便在新训练语料长度超过某个值时输出信息
  // sen 单词数组，表示句子
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  // l1 ns中表示word在concatenated word vectors中的起始位置，之后layer1_size是对应的word vector，因为把矩阵拉成长向量了
  // l2 cbow或ns中权重向量的起始位置，之后layer1_size是对应的syn1或syn1neg，因为把矩阵拉成长向量了
  // c 循环中的计数作用
  // target ns中当前的sample
  // label ns中当前sample的label
  long long l1, l2, c, target, label, local_iter = iter;
  // id 线程创建的时候传入，辅助随机数生成
  unsigned long long next_random = (long long)id;
  // f e^x / (1/e^x)，fs中指当前编码为是0（父亲的左子节点为0，右为1）的概率，ns中指label是1的概率
  // g 误差(f与真实值的偏离)与学习速率的乘积
  real f, g;
  clock_t now;	// 当前时间，和start比较计算算法效率
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));	// 输入词向量累加均值
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));	// 误差累计项，其实对应的是Gneu1
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);	// 将文件内容平均分配给各个线程,每个线程读取文件的开始位置不同
  while (1) {
    if (word_count - last_word_count > 10000) {	// 每处理玩10000个词调整学习率
      word_count_actual += word_count - last_word_count;	// TODO
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,	// 计算训练进度
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));	// 计算训练速度
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));	// 调整学习率
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;	// 学习率过小则固定一个值
    }
    if (sentence_length == 0) {	// 重新读取一个句子
      while (1) {	// 读取一个句子
        word = ReadWordIndex(fi);
        if (feof(fi)) break;	// 文件末尾
        if (word == -1) continue;	// 词不在词典中,则跳过该词
        word_count++;
        if (word == 0) break;	// 如果是换行符,则跳出
        // The subsampling randomly discards frequent words while keeping the ranking same
        // 对高频词进行下采样,低频词被丢弃概率低，高频词被丢弃概率高
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;	// 以一定概率舍弃该词
        }
        sen[sentence_length] = word;	// 将该词存入句子
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;	// 句子过长,舍弃后面的.
      }
      sentence_position = 0;	// 句子指针重置
    }
    if (feof(fi) || (word_count > train_words / num_threads)) {	// 文件结束,或者该线程工作已经完成
      word_count_actual += word_count - last_word_count;	// TODO
      local_iter--;	// 迭代次数减一
      if (local_iter == 0) break;	// 迭代次数已达到,则退出,该线程结束
      word_count = 0;	// 重新开始一次迭代
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);	// 重置文件开始位置
      continue;
    }
    word = sen[sentence_position];	// 取句子中的一个单词，开始训练
    if (word == -1) continue;	// TODO 怎么还会不存在
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;	// 0初始化参数
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;	// b是个随机数，0到window-1，指定了本次算法操作实际的窗口大小
    if (cbow) {  // 训练cbow模型,输入是平均窗口词向量,预测中心词
      // in -> hidden
      cw = 0;
      // 将窗口内的word vectors累加
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;	// c超界则直接跳过
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];	// 累加到neu1
        cw++;
      }
      if (cw) {	// 经过了窗口累加,即neu1不为0
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;	// 累加词向量取平均
        // hierarchy softmax 方式
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;	// 该节点在syn1中第一维的偏移量
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];	// 向量的积,这里输入是平均窗口词向量
          if (f <= -MAX_EXP) continue;	// 不在expTable内的舍弃掉,TODO 或者改为改成太小的都是0，太大的都是1
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];	// 计算σ,查表
          // 'g' is the gradient multiplied by the learning rate
          // g是梯度与学习率的乘积,辅助量
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];	// 计算该节点的误差项(关于neu1的梯度),最后每一层树节点都会累加起来
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];	// 更新该层非叶节点的参数,g*neu1[c]为关于syn1的梯度
        }
        // NEGATIVE SAMPLING 方式
        if (negative > 0) for (d = 0; d < negative + 1; d++) {	// 遍历每一个负采样的样本
          if (d == 0) {	// 第一个词是一个正样本,为当前词
            target = word;
            label = 1;
          } else {	// 进行一次负采样
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];	// 得到负样本的索引
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;	// 跳过自身
            label = 0;
          }
          l2 = target * layer1_size;	// target在syn1neg中第一维的偏移量
          f = 0;
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];	// 向量的积
          if (f > MAX_EXP) g = (label - 1) * alpha;	// σ得出正类
          else if (f < -MAX_EXP) g = (label - 0) * alpha;	// σ得出负类
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;	// 查表计算σ,再求g
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];	// 计算该样本的误差项(关于neu1的梯度),最后每个负样本的都会累加起来
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];		// 更新该负样本的参数,g*neu1[c]为关于syn1neg的梯度
        }
        // hidden -> in
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {	// 更新窗口词向量,正确
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];	// 更新词向量
        }
      }
    } else {  // train skip-gram 模型, 输入是窗口词的词向量,预测中心词
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;	// 当前窗口词在syn0中第一维的偏移量
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;	// 0初始化
        // HIERARCHICAL SOFTMAX
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;	// 该节点在syn1中第一维的偏移量
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];	// 向量的积,输入syn0[c + l1]为该窗口词的词向量
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];	// 计算σ,查表
          // 'g' is the gradient multiplied by the learning rate
          // g是梯度与学习率的乘积,辅助量
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];	// 计算该样本的误差项(关于syn0的梯度),最后每个负样本的都会累加起来
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];	// 更新非叶节点参数向量
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {	// 遍历每一个负采样的样本
          if (d == 0) {	// 第一个负样本为一个本身,即正类
            target = word;
            label = 1;
          } else {	// 进行一次负采样
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;	// 负采样采到自身则跳过
            label = 0;
          }
          l2 = target * layer1_size;	// l2为负采样样本的索引
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];	// 向量的积,输入syn0[c + l1]为该窗口词的词向量
          if (f > MAX_EXP) g = (label - 1) * alpha;	// σ得出正类
          else if (f < -MAX_EXP) g = (label - 0) * alpha;	// σ得出负类
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;	// 查表计算σ,再得出g
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];	// 计算该样本的误差项(关于syn0的梯度),最后每个负样本的都会累加起来
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];	// 更新该负样本的参数,g*syn0[c]为关于syn1neg的梯度
        }
        // Learn weights input -> hidden
        // 每处理完一个窗口词
        // 更新窗口词,而不是中心词,因为l1是窗口词的一个维度的索引,博客错了
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
      }
    }
    sentence_position++;	// 指向句子中的下一个词
    if (sentence_position >= sentence_length) {	// 该句子训练完成
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  // 创建多线程
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();	// 得到词典, 优先从词汇表文件中加载，否则从训练文件中加载
  if (save_vocab_file[0] != 0) SaveVocab();	// 输出词汇表文件，词+词频
  if (output_file[0] == 0) return;	// 不训练词向量则跳出
  InitNet();	// 网络结构初始化
  if (negative > 0) InitUnigramTable();	// 构建负采样表
  start = clock();	// 开始计时
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);	// 创建线程
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);	// 等待所有线程结束
  fo = fopen(output_file, "wb");	// 打开输出文件
  if (classes == 0) {	// 输出词向量
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);	// 输出词数量、向量维度
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);	// 输出该词
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);	// 二进制形式输出每一个维度
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);	// 字符形式输出
      fprintf(fo, "\n");	// 输出换行
    }
  } else {	// 进行K-means聚类
    // Run K-means on the word vectors
	// clcn 类别数
	// iter 聚类次数
    int clcn = classes, iter = 10, closeid;
    // centcn 存储每一类的样本个数
    int *centcn = (int *)malloc(classes * sizeof(int));
    // cl 存储每个样本属于哪个类
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    real closev, x;
    // 最终用来存储每一类聚类中心点
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;	// 将样本平均分到各个类别中
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;	// 0初始化聚类中心
      for (b = 0; b < clcn; b++) centcn[b] = 1;	// 1初始化每个类中的样本数
      for (c = 0; c < vocab_size; c++) {	// 累加每一类的样本向量
        for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++;	// 该类样本数加1
      }
      for (b = 0; b < clcn; b++) {	// 求出每一类样本的中心点,并单位化
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];	// 求该类的均值中心向量
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];	// 算平方和
        }
        closev = sqrt(closev);	// 求长度
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;	// 单位化该类的中心向量
      }
      for (c = 0; c < vocab_size; c++) {	// 对每一个样本重新分类
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {	// 求出该样本与哪个中心点最近,则归为该类
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];	// 计算样本中心与当前样本的内积
          if (x > closev) {	// TODO是余弦距离吗?
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    // 保存聚类结果
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
}

// 根据参数名称确定参数的序号
int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

void test(){
	char word[MAX_STRING];
	FILE *fin;
	long long a, i = 0;
	for (a = 0; a < vocab_hash_size; a++)
		vocab_hash[a] = -1;	// 初始化hash表

	fin = fopen(train_file, "rb");
	while(i < 3){
		ReadWord(word, fin);
		printf("%s %d\n", word, i);
		i++;
	}

	fclose(fin);
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word)); // 初始化存储空间
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {	// 生成sigmoid查找表
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
//  test();
  TrainModel();
  return 0;
}
