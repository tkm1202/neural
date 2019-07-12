コンパイル
gcc -o neural neural.c -lm
実行方法
./neural

iris データセット (iris train.dat) を学習用データとして読み込み, 修正後の重 み w1(入力層→中間層),w2(中間層→出力層) の値をそれぞれ iris w1.dat , iris w2.dat へ出力する.
