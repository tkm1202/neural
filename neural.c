#include <stdio.h>
#include <math.h>

#define P 140
#define C 3
#define N 4
#define H 4

int main(){
  FILE *fp1,*fp2,*fp3;
  int i,j,count=0,flag=0;
  double labels[P+1][C+1];
  double rho = 0.1,total,class;
  double h1[H+1];
  double h2[H+1];
  double g1[H+1];
  double g2[H+1];
  double e[C+1],E = 0,sigma=0;
  int p,k,n;
  if((fp1=fopen("iris_train.dat","r"))==NULL){
    printf("error\n");
  }
  if((fp2=fopen("iris_w1.dat","w"))==NULL){
    printf("error\n");
  }
  if((fp3=fopen("iris_w2.dat","w"))==NULL){
    printf("error\n");
  }
  for(i=1;i<=P;i++){
    for(j=0;j<=C;j++){
      labels[i][j]=0;
    }
  }
  double pattern[P+1][N+1];
  for(p=1;p<=P;p++){
    fscanf(fp1,"%lf %lf %lf %lf %lf",&pattern[p][0],&pattern[p][1],&pattern[p][2],&pattern[p][3],&class);
     printf("%lf %lf %lf %lf %lf\n",pattern[p][0],pattern[p][1],pattern[p][2],pattern[p][3],class);
     if(class==0)labels[p][0]=1;
     else if(class==1)labels[p][1]=1;
     else labels[p][2]=1;
	}

  
  /* 中間層の重みの初期化 */
  double w1[H+1][N+1];
  w1[0][0] = 0.2; w1[0][1] = 0.8; w1[0][2] = 0.2; w1[0][3] = 0.5;
  w1[1][0] = 0.8; w1[1][1] = 0.6; w1[1][2] = 0.4; w1[1][3] = 0.5;
  w1[2][0] = 0.1; w1[2][1] = 0.8; w1[2][2] = 0.7; w1[2][3] = 0.5;
  w1[3][0] = 0.3; w1[3][1] = 0.4; w1[3][2] = 0.2; w1[3][3] = 0.5;
  /* 出力層の重みの初期化  */
  double w2[C+1][H+1];
  w2[0][0] = 0.2; w2[0][1] = 0.6; w2[0][2] = 0.5; w2[0][3] = 0.9;
  w2[1][0] = 0.6; w2[1][1] = 0.8; w2[1][2] = 0.7; w2[1][3] = 0.003;
  w2[2][0] = 0.5; w2[2][1] = 0.6; w2[2][2] = 0.1; w2[2][3] = 0.784;
  //printf("1\n");
  p=1;count=0;
  while(flag<C-1/* 学習の終了判定 */){count++;printf("ループ%d回目 ",count);
    for(i=0;i<=N;i++){
      h1[i]=0;
    }
    for(i=0;i<=C;i++){
      h2[i]=0;
    }
    /* 順伝播の計算 */
     if(p==P+1)p=1;
	/* g1の計算 */
    for(j=0; j<H; j++){
      for(i=0; i<N; i++){
	h1[j] = h1[j] + w1[j][i] * pattern[p][i];
	  }
      g1[j] = 1/(1+exp((-1) * h1[j]));/* 重みごとに足して活性化関数を通す */
      // printf("h1[%d]=%lf,g1[%d]=%lf\n",j,h1[j],j,g1[j]);
	}
	/* g2の計算 */
    for(k=0; k<C; k++){
      for(i=0; i<H; i++){
	h2[k] = h2[k] + w2[k][i] * g1[i];
      }
      g2[k] = 1/(1+exp((-1) * h2[k]));/* 重みごとに足して活性化関数を通す */
      // printf("h2[%d]=%lf,g2[%d]=%lf\n",k,h2[k],k,g2[k]);
     
	}
   
    /* 逆伝播の計算 */
    /* w2の計算 */
    
    for(k=0; k<C; k++){
      e[k] = (g2[k] - labels[p][k]) * g2[k] * (1 - g2[k]);
  
      for(j = 0; j < H; j++){
	w2[k][j] = w2[k][j] - rho * e[k] * g1[j];
      }
    } 
;
    /* w1の計算 */
    for(j=0; j<H; j++){
      for(i=0; i<C; i++){
	sigma = sigma + (e[i] * w2[i][j]);
      }
      E = sigma *  g1[j] * (1 - g1[j]);
      sigma = 0;
      for(i=0; i<=N; i++){
	w1[j][i] = w1[j][i] - rho * E * pattern[p][i];
      }
    
      E = 0;
    }
     printf("pattern%d,g2[0]=%lf,g2[1]=%lf,g2[2]=%lf\n",p,g2[0],g2[1],g2[2]);
    total=0;
    for(i=0;i<C;i++){
      total+=(pow((labels[k][i]-g2[i]),2))/2;
    }//printf("%lf\n",total);

    if(total<=0.002){
      flag++;//printf("flag=%d\n",flag);
    }
    else {flag=0;}
    p++;
  }

 
  for(i=0;i<H;i++){
    for(j=0;j<N;j++){
      fprintf(fp2,"%lf ",w1[i][j]);
    }fprintf(fp2,"\n");
  }

  for(i=0;i<C;i++){
    for(j=0;j<H;j++){
      fprintf(fp3,"%lf ",w2[i][j]);
    }fprintf(fp3,"\n");
  }

 fclose(fp1);
 fclose(fp2);
 fclose(fp3);
 

}
