#include <iostream>
using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

#include <opencv2\core\cuda.hpp>
#include <opencv2\cuda\cudaimgproc.hpp>
using namespace cuda;

#define GPU 

int main()
{
	int n = 10000;

#ifdef GPU
	int num_devices = getCudaEnabledDeviceCount();
	cout << "可使用的设备个数为：" << num_devices << endl;
	setDevice(0);
#endif

	Mat img = imread("G:/test3.jpg");
	//Mat img = imread("RightCode.bmp",0);
	Mat t = imread("G:/8.jpg", 0);
	Mat dst;	


#ifdef GPU
	GpuMat gpu_img;
	GpuMat gpu_t;
	gpu_img.upload(img);
	gpu_t.upload(t);

	uchar d = *(gpu_img.data);	

	Ptr<TemplateMatching> alg = createTemplateMatching(CV_8U, CV_TM_CCOEFF_NORMED);

	GpuMat gpu_dst;
#endif

	TickMeter tm;
	tm.start();
	
#ifdef GPU
	for (int i = 0; i < n; i++)
		cuda::cvtColor(gpu_img, gpu_dst, CV_BGR2GRAY);
		//alg->match(gpu_img, gpu_t, gpu_dst);		
#else
	for (int i = 0; i < n; i++)
		cv::cvtColor(img, dst, CV_BGR2GRAY);
		//matchTemplate(img, t, dst, CV_TM_CCOEFF_NORMED);		
#endif

	tm.stop();
	cout << tm.getTimeMilli()/n << endl;

#ifdef GPU	
	gpu_dst.download(dst);	
#endif

	imshow("dst", dst);
	waitKey();
}