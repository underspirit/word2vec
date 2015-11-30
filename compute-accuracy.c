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
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <ctype.h>

const long long max_size = 2000;         // max length of strings
const long long N = 1;                   // number of closest words,因为只有最相似的那一个正确才算正确匹配
const long long max_w = 50;              // max length of vocabulary entries

int main(int argc, char **argv)
{
  FILE *f;
  // st1,2,3,4 用户输入的词
  // bestd 存储每个相似词的相似度
  // bestw 存储最近的N个词
  char st1[max_size], st2[max_size], st3[max_size], st4[max_size], bestw[N][max_size], file_name[max_size], ch;
  float dist, len, bestd[N], vec[max_size];
  // word,词典大小
  // size,向量维度
  long long words, size, a, b, c, d, b1, b2, b3, threshold = 0;
  float *M;	// 存储所有词向量
  char *vocab;	// 存储词向量对应的词
  // TCN  某类别处理个数
  // CNN  某类别正确个数
  // TACN 总处理个数
  // CACN 总正确个数
  // SECN 总semantic处理个数
  // SYCN 总syntatic处理个数
  // SEAC 总semantic正确个数
  // SYAC 总syntatic正确个数
  // QID  当前处理的类别个数
  // TQ   总的问题个数
  // TQS  总的处理的问题数
  int TCN, CCN = 0, TACN = 0, CACN = 0, SECN = 0, SYCN = 0, SEAC = 0, SYAC = 0, QID = 0, TQ = 0, TQS = 0;
  if (argc < 2) {
    printf("Usage: ./compute-accuracy <FILE> <threshold>\nwhere FILE contains word projections, and threshold is used to reduce vocabulary of the model for fast approximate evaluation (0 = off, otherwise typical value is 30000)\n");
    return 0;
  }
  strcpy(file_name, argv[1]);
  if (argc > 2) threshold = atoi(argv[2]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  if (threshold) if (words > threshold) words = threshold;	// 缩小词典
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc(words * max_w * sizeof(char));
  M = (float *)malloc(words * size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(float) / 1048576);
    return -1;
  }
  for (b = 0; b < words; b++) {
    a = 0;
    while (1) {		// 读取一个词
      vocab[b * max_w + a] = fgetc(f);
      if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
      if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
    }
    vocab[b * max_w + a] = 0;	// 每个词以0结尾
    for (a = 0; a < max_w; a++) vocab[b * max_w + a] = toupper(vocab[b * max_w + a]);	// 转换为大写字符
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);	// 读取该词对应的词向量
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;	// 单位化词向量
  }
  fclose(f);
  TCN = 0;
  while (1) {
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    scanf("%s", st1);	// 读取word1
    for (a = 0; a < strlen(st1); a++) st1[a] = toupper(st1[a]);
    if ((!strcmp(st1, ":")) || (!strcmp(st1, "EXIT")) || feof(stdin)) {
      if (TCN == 0) TCN = 1;
      if (QID != 0) {	// 输出这个类别的统计信息
        printf("ACCURACY TOP1: %.2f %%  (%d / %d)\n", CCN / (float)TCN * 100, CCN, TCN);
        printf("Total accuracy: %.2f %%   Semantic accuracy: %.2f %%   Syntactic accuracy: %.2f %% \n", CACN / (float)TACN * 100, SEAC / (float)SECN * 100, SYAC / (float)SYCN * 100);
      }
      QID++;
      scanf("%s", st1);
      if (feof(stdin)) break;
      printf("%s:\n", st1);
      TCN = 0;	// 该类别统计量重置
      CCN = 0;
      continue;
    }
    if (!strcmp(st1, "EXIT")) break;
    scanf("%s", st2);	// 读取word2
    for (a = 0; a < strlen(st2); a++) st2[a] = toupper(st2[a]);
    scanf("%s", st3);	// 读取word3
    for (a = 0; a<strlen(st3); a++) st3[a] = toupper(st3[a]);
    scanf("%s", st4);	// 读取word4
    for (a = 0; a < strlen(st4); a++) st4[a] = toupper(st4[a]);
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st1)) break;	// 找到st1,2,3的索引
    b1 = b;
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st2)) break;
    b2 = b;
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st3)) break;
    b3 = b;
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    TQ++;	// 总问题数加1
    if (b1 == words) continue;	// 输入词不存在词典中,则跳过
    if (b2 == words) continue;
    if (b3 == words) continue;
    for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st4)) break;	// 找到st4的索引
    if (b == words) continue;	// 目标词不在词典中,跳过
    for (a = 0; a < size; a++) vec[a] = (M[a + b2 * size] - M[a + b1 * size]) + M[a + b3 * size];	// 计算word4 = word2 - word1 + word3
    TQS++;	// 处理的总个数加1
    for (c = 0; c < words; c++) {
      if (c == b1) continue;	// 跳过自身的三个词
      if (c == b2) continue;
      if (c == b3) continue;
      dist = 0;
      for (a = 0; a < size; a++) dist += vec[a] * M[a + c * size];	// 计算余弦相似度
      for (a = 0; a < N; a++) {	// 相似度在前N个,则加入到bestd中,这里N=1,所以是选最大
        if (dist > bestd[a]) {
          for (d = N - 1; d > a; d--) {
            bestd[d] = bestd[d - 1];
            strcpy(bestw[d], bestw[d - 1]);
          }
          bestd[a] = dist;
          strcpy(bestw[a], &vocab[c * max_w]);	// 记录相似词
          break;
        }
      }
    }
    if (!strcmp(st4, bestw[0])) {	// 如果结果正确
      CCN++; // 正确个数加一
      CACN++;	// 总正确个数加1
      if (QID <= 5) SEAC++; 		// 前5类是semantic类别,semantic正确数加1
      else SYAC++;	// 后面的类都是syntatic类,加1
    }
    if (QID <= 5) SECN++;
    else SYCN++;
    TCN++;	// 该类别处理个数加1
    TACN++; // 总处理个数加1
  }
  printf("Questions seen / total: %d %d   %.2f %% \n", TQS, TQ, TQS/(float)TQ*100);
  return 0;
}
