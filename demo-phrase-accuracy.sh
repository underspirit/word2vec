make
if [ ! -e news.2012.en.shuffled ]; then
  wget http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.en.shuffled.gz
  gzip -d news.2012.en.shuffled.gz -f
fi
# 替换’为'，′为'，'为'                                                  替换A-Za-z'_ \n之外的字符为空格
sed -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" < news.2012.en.shuffled | tr -c "A-Za-z'_ \n" " " > news.2012.en.shuffled-norm0
# threshold 200训练词组
time ./word2phrase -train news.2012.en.shuffled-norm0 -output news.2012.en.shuffled-norm0-phrase0 -threshold 200 -debug 2
# threshold 100训练词组
time ./word2phrase -train news.2012.en.shuffled-norm0-phrase0 -output news.2012.en.shuffled-norm0-phrase1 -threshold 100 -debug 2
# 全部转换为小写
tr A-Z a-z < news.2012.en.shuffled-norm0-phrase1 > news.2012.en.shuffled-norm1-phrase1
# 训练词向量
time ./word2vec -train news.2012.en.shuffled-norm1-phrase1 -output vectors-phrase.bin -cbow 1 -size 200 -window 10 -negative 25 -hs 0 -sample 1e-5 -threads 20 -binary 1 -iter 15
# 计算准确率
./compute-accuracy vectors-phrase.bin < questions-phrases.txt
