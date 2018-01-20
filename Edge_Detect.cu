2 # include <time .h>
3 # include <stdlib .h>
4 # include <stdio .h>
5 # include <string .h>
6 # include <math .h>
7 # include <cuda .h>
8 # include <cutil .h>
9 # include <ctime >
39
10
11 unsigned int width , height ;
12
13 int Gx [3][3] = { -1 , 0 , 1 ,
14 -2 , 0 , 2 ,
15 -1 , 0 , 1};
16
17 int Gy [3][3] = {1 ,2 ,1 ,
18 0 ,0 ,0 ,
19 -1 , -2 , -1};
20
21 int getPixel ( unsigned char * org , int col , int row) {
22
23 int sumX , sumY ;
24 sumX = sumY = 0;
25
26 for (int i= -1; i <= 1; i++) {
27 for (int j= -1; j <=1; j++) {
28 int curPixel = org [( row + j) * width + (col + i) ];
29 sumX += curPixel * Gx[i +1][ j +1];
30 sumY += curPixel * Gy[i +1][ j +1];
31 }
32 }
33 int sum = abs( sumY ) + abs( sumX ) ;
34 if (sum > 255) sum = 255;
35 if (sum < 0) sum = 0;
36 return sum ;
37 }
38
39 void h_EdgeDetect ( unsigned char * org , unsigned char * result ) {
40 int offset = 1 * width ;
41 for (int row =1; row < height -2; row ++) {
42 for (int col =1; col <width -2; col ++) {
43 result [ offset + col ] = getPixel (org , col , row ) ;
44 }
45 offset += width ;
46 }
47 }
48
40
49 __global__ void d_EdgeDetect ( unsigned char *org , unsigned char *
result , int width , int height ) {
50 int col = blockIdx .x * blockDim .x + threadIdx .x;
51 int row = blockIdx .y * blockDim .y + threadIdx .y;
52
53 if (row < 2 || col < 2 || row >= height -3 || col >= width -3 )
54 return ;
55
56 int Gx [3][3] = { -1 , 0 , 1 ,
57 -2 , 0 , 2 ,
58 -1 , 0 , 1};
59
60 int Gy [3][3] = {1 ,2 ,1 ,
61 0 ,0 ,0 ,
62 -1 , -2 , -1};
63
64 int sumX , sumY ;
65 sumX = sumY = 0;
66
67 for (int i= -1; i <= 1; i++) {
68 for (int j= -1; j <=1; j++) {
69 int curPixel = org [( row + j) * width + (col + i) ];
70 sumX += curPixel * Gx[i +1][ j +1];
71 sumY += curPixel * Gy[i +1][ j +1];
72 }
73 }
74
75 int sum = abs( sumY ) + abs( sumX ) ;
76 if (sum > 255) sum = 255;
77 if (sum < 0) sum = 0;
78
79 result [row * width + col ] = sum ;
80
81 }
82
83 int main ( int argc , char ** argv )
84 {
85 printf (" Starting program \n") ;
86
41
87 /* ******************** setup work ***************************
*/
88
89 unsigned char * d_resultPixels ;
90 unsigned char * h_resultPixels ;
91 unsigned char * h_pixels = NULL ;
92 unsigned char * d_pixels = NULL ;
93
94 char * srcPath = "/ Developer /GPU Computing /C/src / EdgeDetection /
image / cartoon .pgm";
95 char * h_ResultPath = "/ Developer /GPU Computing /C/src /
EdgeDetection / output / h_cartoon .pgm";
96 char * d_ResultPath = "/ Developer /GPU Computing /C/src /
EdgeDetection / output / d_cartoon .pgm";
97
98 cutLoadPGMub ( srcPath , & h_pixels , &width , & height ) ;
99
100 int ImageSize = sizeof ( unsigned char ) * width * height ;
101
102 h_resultPixels = ( unsigned char *) malloc ( ImageSize ) ;
103 cudaMalloc (( void **) & d_pixels , ImageSize ) ;
104 cudaMalloc (( void **) & d_resultPixels , ImageSize ) ;
105 cudaMemcpy ( d_pixels , h_pixels , ImageSize , cudaMemcpyHostToDevice
) ;
106
107 /* ******************** END setup work
*************************** */
108
109 /* ************************ Host processing
************************* */
110 clock_t starttime , endtime , difference ;
111
112 printf (" Starting host processing \n") ;
113 starttime = clock () ;
114 h_EdgeDetect ( h_pixels , h_resultPixels ) ;
115 endtime = clock () ;
116 printf (" Completed host processing \n") ;
117
118 difference = ( endtime - starttime ) ;
42
119 double interval = difference / ( double ) CLOCKS_PER_SEC ;
120 printf ("CPU execution time = %f ms\n", interval * 1000) ;
121 cutSavePGMub ( h_ResultPath , h_resultPixels , width , height ) ;
122 /* ************************ END Host processing
************************* */
123
124 /* ************************ Device processing
************************* */
125 dim3 block (16 ,16) ;
126 dim3 grid ( width /16 , height /16) ;
127 unsigned int timer = 0;
128 cutCreateTimer (& timer ) ;
129
130 printf (" Invoking Kernel \n") ;
131 cutStartTimer ( timer ) ;
132 /* CUDA method */
133 d_EdgeDetect <<< grid , block > > >( d_pixels , d_resultPixels , width
, height ) ;
134 cudaThreadSynchronize () ;
135 cutStopTimer ( timer ) ;
136 printf (" Completed Kernel \n") ;
137
138 printf (" CUDA execution time = %f ms\n", cutGetTimerValue ( timer ) )
;
139
140 cudaMemcpy ( h_resultPixels , d_resultPixels , ImageSize ,
cudaMemcpyDeviceToHost ) ;
141 cutSavePGMub ( d_ResultPath , h_resultPixels , width , height ) ;
142
143 /* ************************ END Device processing
************************* */
144
145
146
147 printf (" Press enter to exit ...\ n") ;
148 getchar () ;
149 }
