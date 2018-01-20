1 # include <time .h>
2 # include <stdlib .h>
3 # include <stdio .h>
4 # include <string .h>
5 # include <math .h>
6 # include <cuda .h>
7 # include <cutil .h>
8 # include <ctime >
9
10 unsigned int width , height ;
11 int mask [3][3] = {1 ,2 ,1 ,
12 2 ,3 ,2 ,
13 1 ,2 ,1 ,
14 };
15
16 int getPixel ( unsigned char * arr , int col , int row) {
17
18 int sum = 0;
19
20 for (int j= -1; j <=1; j++) {
21 for (int i= -1; i <=1; i++) {
22 int color = arr [( row + j) * width + (col + i) ];
23 sum += color * mask [i +1][ j +1];
24 }
25 }
26
27 return sum /15;
28 }
29
34
30 void h_blur ( unsigned char * arr , unsigned char * result ) {
31 int offset = 2 * width ;
32 for (int row =2; row < height -3; row ++) {
33 for (int col =2; col <width -3; col ++) {
34 result [ offset + col ] = getPixel (arr , col , row ) ;
35 }
36 offset += width ;
37 }
38 }
39
40
41 __global__ void d_blur ( unsigned char * arr , unsigned char * result ,
int width , int height ) {
42 int col = blockIdx .x * blockDim .x + threadIdx .x;
43 int row = blockIdx .y * blockDim .y + threadIdx .y;
44
45 if (row < 2 || col < 2 || row >= height -3 || col >= width -3 )
46 return ;
47
48 int mask [3][3] = {1 ,2 ,1 , 2 ,3 ,2 , 1 ,2 ,1};
49
50 int sum = 0;
51 for (int j= -1; j <=1; j++) {
52 for (int i= -1; i <=1; i++) {
53 int color = arr [( row + j) * width + (col + i) ];
54 sum += color * mask [i +1][ j +1];
55 }
56 }
57
58 result [row * width + col ] = sum /15;
59
60 }
61
62
63 int main ( int argc , char ** argv )
64 {
65 /* ******************** setup work ***************************
*/
66 unsigned char * d_resultPixels ;
35
67 unsigned char * h_resultPixels ;
68 unsigned char * h_pixels = NULL ;
69 unsigned char * d_pixels = NULL ;
70
71
72 char * srcPath = "/ Developer /GPU Computing /C/src / GaussianBlur /
image / wallpaper2 .pgm";
73 char * h_ResultPath = "/ Developer /GPU Computing /C/src /
GaussianBlur / output / h_wallpaper2 .pgm ";
74 char * d_ResultPath = "/ Developer /GPU Computing /C/src /
GaussianBlur / output / d_wallpaper2 .pgm ";
75
76
77 cutLoadPGMub ( srcPath , & h_pixels , &width , & height ) ;
78
79 int ImageSize = sizeof ( unsigned char ) * width * height ;
80
81 h_resultPixels = ( unsigned char *) malloc ( ImageSize ) ;
82 cudaMalloc (( void **) & d_pixels , ImageSize ) ;
83 cudaMalloc (( void **) & d_resultPixels , ImageSize ) ;
84 cudaMemcpy ( d_pixels , h_pixels , ImageSize , cudaMemcpyHostToDevice
) ;
85
86 /* ******************** END setup work
*************************** */
87
88 /* ************************ Host processing
************************* */
89
90 clock_t starttime , endtime , difference ;
91 starttime = clock () ;
92
93 // apply gaussian blur
94 h_blur ( h_pixels , h_resultPixels ) ;
95
96 endtime = clock () ;
97 difference = ( endtime - starttime ) ;
98 double interval = difference / ( double ) CLOCKS_PER_SEC ;
99 printf ("CPU execution time = %f ms\n", interval * 1000) ;
36
100 cutSavePGMub ( h_ResultPath , h_resultPixels , width , height ) ;
101
102 /* ************************ END Host processing
************************* */
103
104
105 /* ************************ Device processing
************************* */
106 dim3 block (16 ,16) ;
107 dim3 grid ( width /16 , height /16) ;
108 unsigned int timer = 0;
109 cutCreateTimer (& timer ) ;
 cutStartTimer ( timer ) ;

 /* CUDA method */
 d_blur <<< grid , block > > >( d_pixels , d_resultPixels , width ,
height ) ;

 cudaThreadSynchronize () ;
 cutStopTimer ( timer ) ;
 printf (" CUDA execution time = %f ms\n", cutGetTimerValue ( timer ) ) ;

 cudaMemcpy ( h_resultPixels , d_resultPixels , ImageSize ,
cudaMemcpyDeviceToHost ) ;
 cutSavePGMub ( d_ResultPath , h_resultPixels , width , height ) ;

/* ************************ END Device processing
************************* */
 printf (" Press enter to exit ...\ n") ;
 getchar () ;
 }
