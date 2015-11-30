//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>

const long long max_size = 2000;         // max length of strings,
const long long N = 40;                  // number of closest words that will be shown,显示前N个最近的词
const long long max_w = 50;              // max length of vocabulary entries,词的最大长度

int main(int argc, char **argv) {
  FILE *f;
  // 用户输入的词
  char st1[max_size];
  char *bestw[N];	// 存储最近的N个词
  // st 存储用户输入的词(词组)
  char file_name[max_size], st[100][max_size];
  // bestd 存储每个相似词的相似度
  // vec 存储用户输入词的词向量,词组则是累加向量
  float dist, len, bestd[N], vec[max_size];
  // word,词典大小
  // size,向量维度
  // cn 用户本次输入的词的个数(拆分词组)
  // bi 存储用户输入词的在词典中的索引
  long long words, size, a, b, c, d, cn, bi[100];
  char ch;
  float *M;	// 存储所有词向量
  char *vocab;	// 存储词向量对应的词
  if (argc < 2) {
    printf("Usage: ./distance <FILE>\nwhere FILE contains word projections in the BINARY FORMAT\n");
    return 0;
  }
  strcpy(file_name, argv[1]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  for (a = 0; a < N; a++) bestw[a] = (char *)malloc(max_size * sizeof(char));
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
  for (b = 0; b < words; b++) {
    a = 0;
    while (1) {	// 读取一个词
      vocab[b * max_w + a] = fgetc(f);	// 读一个字符
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;	// 读下一个字符
    }
    vocab[b * max_w + a] = 0;	// 每个词以0结尾
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);	// 读取该词对应的词向量
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];	// 平方和
    len = sqrt(len);	// 得向量长度
    for (a = 0; a < size; a++) M[a + b * size] /= len;	// 单位化词向量
  }
  fclose(f);
  while (1) {
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    printf("Enter word or sentence (EXIT to break): ");
    a = 0;
    while (1) {	// 读取用户的输入
      st1[a] = fgetc(stdin);
      if ((st1[a] == '\n') || (a >= max_size - 1)) {
        st1[a] = 0;	// 以0结尾
        break;
      }
      a++;
    }
    if (!strcmp(st1, "EXIT")) break;	// 退出
    cn = 0;
    b = 0;
    c = 0;
    while (1) {	// 将用户输入的词存入st,若是一个词组则拆成多个词
      st[cn][b] = st1[c];
      b++;
      c++;
      st[cn][b] = 0;	// 词以0结尾
      if (st1[c] == 0) break;
      if (st1[c] == ' ') {	// 拆分词组
        cn++;
        b = 0;
        c++;
      }
    }
    cn++;
    for (a = 0; a < cn; a++) {
      for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st[a])) break;	// 在词典中寻找相等的词
      if (b == words) b = -1;	// 没找到
      bi[a] = b;
      printf("\nWord: %s  Position in vocabulary: %lld\n", st[a], bi[a]);
      if (b == -1) {
        printf("Out of dictionary word!\n");
        break;
      }
    }
    if (b == -1) continue;	// 不存在,重新输入词
    printf("\n                                              Word       Cosine distance\n------------------------------------------------------------------------\n");
    for (a = 0; a < size; a++) vec[a] = 0;
    for (b = 0; b < cn; b++) {	// 累加用户输入的词(词组)对应的词向量到vec
      if (bi[b] == -1) continue;	// 应该不会再等于-1吧
      for (a = 0; a < size; a++) vec[a] += M[a + bi[b] * size];
    }
    len = 0;
    for (a = 0; a < size; a++) len += vec[a] * vec[a];	// 平方和
    len = sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= len;	// 单位化
    for (a = 0; a < N; a++) bestd[a] = -1;	// 初始化相似度为0
    for (a = 0; a < N; a++) bestw[a][0] = 0;	// 初始化相似词为空(因为以0开头)
    for (c = 0; c < words; c++) {
      a = 0;
      for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;	// a = 1 表示当前词即为用户输入的词
      if (a == 1) continue;	// 跳过
      dist = 0;
      for (a = 0; a < size; a++) dist += vec[a] * M[a + c * size];	// 计算余弦相似度
      for (a = 0; a < N; a++) {	// 相似度在前N个,则加入到bestd中
        if (dist > bestd[a]) {
          for (d = N - 1; d > a; d--) {	// 后移
            bestd[d] = bestd[d - 1];
            strcpy(bestw[d], bestw[d - 1]);
          }
          bestd[a] = dist;
          strcpy(bestw[a], &vocab[c * max_w]);
          break;
        }
      }
    }
    for (a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);	// 输出
  }
  return 0;
}
