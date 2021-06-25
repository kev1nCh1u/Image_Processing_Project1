#include <iostream>
#include <omp.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <eigen3/Eigen/Dense>
#include <string>

#define DRAW_MATCHES

#define FOCAL_LENGTH 10.0
#define FRAME_WIDTH 13.2
#define FRAME_HEIGHT 8.8
#define PRINCIPALPOINT_X 0.0
#define PRINCIPALPOINT_Y 0.0
#define ACCURACY 1.0

#define INIT_L_X 0.0
#define INIT_L_Y 0.0
#define INIT_L_Z 0.0
#define INIT_L_OMEGA 0.0
#define INIT_L_PHI 0.0
#define INIT_L_KAPPA 0.0

#define INIT_R_X 1.0f
#define INIT_R_Y 0.0518f
#define INIT_R_Z 0.015f
#define INIT_R_OMEGA 0.127f
#define INIT_R_PHI -0.250f
#define INIT_R_KAPPA 0.188f

using namespace cv;
using namespace cv::xfeatures2d;

void calForRotationMatrix(const Eigen::Matrix<float, 3, 1> &params, Eigen::Matrix3f &opt);
void pixel2photo(const std::vector<DMatch> &matches, const std::pair<uint, uint> &pixelSize, const std::vector<KeyPoint> &kp1, const std::vector<KeyPoint> &kp2, std::vector<Eigen::Matrix<float, 3, 1>> &opt1, std::vector<Eigen::Matrix<float, 3, 1>> &opt2);

int main()
{
	Mat img1 = imread("../img/DJI_1.JPG", IMREAD_GRAYSCALE);
	Mat img2 = imread("../img/DJI_2.JPG", IMREAD_GRAYSCALE);
	if (img1.empty() || img2.empty())
	{
		std::cout << "can't open the images!\n";
		return -1;
	}
	std::cout << "open image finish\n";

	// std::string detectorType = "surf";
	// int numFeatures = 600; //https://stackoverflow.com/questions/17613723/what-is-the-meaning-of-minhessian-surffeaturedetector
	// const float ratio_thresh = 0.4f;
	// Ptr<SURF> detector = SURF::create(numFeatures);
	// std::vector<KeyPoint> keypoints1, keypoints2;
	// Mat descriptors1, descriptors2;
	// detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
	// detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

	// std::string detectorType = "sift";
	// int numFeatures = 500;
	// const float ratio_thresh = 0.3f;
	// Ptr<SIFT> detector = SIFT::create();
	// std::vector<KeyPoint> keypoints1, keypoints2;
	// Mat descriptors1, descriptors2;
	// detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
	// detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

	std::string detectorType = "orb";
	int numFeatures = 5000;
	const float ratio_thresh = 0.8f;
	Ptr<FeatureDetector> detector = ORB::create(numFeatures);
	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;
	detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
	detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

	std::cout << "detector: " << detectorType << "\n";
	std::cout << "numFeatures: "  << numFeatures << "\n";
	std::cout << "left image key points: " << keypoints1.size() << "\n";
	std::cout << "right image key points: " << keypoints2.size() << "\n";

	if(descriptors1.type()!=CV_32F) {
		descriptors1.convertTo(descriptors1, CV_32F);
	}

	if(descriptors2.type()!=CV_32F) {
		descriptors2.convertTo(descriptors2, CV_32F);
	}

	Mat img12(cv::Size(img1.cols + img2.cols, img1.rows), CV_8UC3);
	drawKeypoints(img1, keypoints1, img12(cv::Rect(0, 0, img1.cols, img1.rows)), Scalar(0, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(img2, keypoints2, img12(cv::Rect(img1.cols, 0, img2.cols, img2.rows)), Scalar(0, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	namedWindow("left_image_keypoints", WINDOW_NORMAL);
	imwrite("../img/"+ detectorType + "_left_image_keypoints.jpg", img12);
	// imshow("left_image_keypoints", img12);
	// waitKey();

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED); //https://docs.opencv.org/3.4/db/d39/classcv_1_1DescriptorMatcher.html#a179cbdf6c8de32f44ae7d5593996e77eaf73d671c6860c24f44b2880a77fadcdc
	std::vector<std::vector<DMatch>> knn_matches;
	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

	std::cout << "distance ratio: " << ratio_thresh << "\n";
	std::vector<DMatch> good_matches;
	for (auto &match : knn_matches)
	{
		if (match[0].distance < ratio_thresh * match[1].distance)
			good_matches.emplace_back(match[0]);
	}
	std::cout << "good matches num: " << good_matches.size() << "\n";

#ifdef DRAW_MATCHES
	Mat img_matches;
	drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
				Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	namedWindow("good_matches", WINDOW_NORMAL);
	imwrite("../img/"+ detectorType + "_good_matches.jpg", img_matches);
	// imshow("good_matches", img_matches);
	// waitKey();
#endif

	// cv::BFMatcher matcher(cv::NORM_L2, true);
	// std::vector<cv::DMatch> matches;
	// matcher.match(descriptors1, descriptors2, matches);

	// std::cout<<"match=";
	// //the number of matched features between the two images
	// std::cout<<matches.size()<<std::endl;
	// cv::Mat imageMatches;
	// cv::drawMatches(img2,keypoints1,img1,descriptors1,
	//    matches,imageMatches,cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
	// cv::namedWindow("matches");
	// cv::imshow("matches",imageMatches);
	// cv::waitKey(0);

	size_t matchSize = good_matches.size();
	Eigen::Matrix<float, 3, 1> b;
	std::vector<Eigen::Matrix<float, 3, 1>> a1(matchSize), a2(matchSize);
	Eigen::Matrix3f rot_matrix_l, rot_matrix_r;
	Eigen::Matrix<float, 3, 1> principalPoint{PRINCIPALPOINT_X, PRINCIPALPOINT_Y, FOCAL_LENGTH};
	Eigen::Matrix<float, 3, 1> pos_l{INIT_L_X, INIT_L_Y, INIT_L_Z}, rot_l{INIT_L_OMEGA, INIT_L_PHI, INIT_L_KAPPA};
	Eigen::Matrix<float, 3, 1> pos_r{INIT_R_X, INIT_R_Y, INIT_R_Z}, rot_r{INIT_R_OMEGA, INIT_R_PHI, INIT_R_KAPPA};
	std::vector<Eigen::Matrix<float, 3, 1>> kp1, kp2;
	std::pair<uint, uint> pixelSize{img1.cols, img1.rows};
	pixel2photo(good_matches, pixelSize, keypoints1, keypoints2, kp1, kp2);

	Eigen::MatrixXf A = Eigen::MatrixXf::Zero(matchSize, 4 * matchSize);
	Eigen::MatrixXf B(matchSize, 5);
	//Eigen::MatrixXf P = Eigen::MatrixXf::Zero(4 * matchSize, 4 * matchSize);
	Eigen::MatrixXf L = Eigen::MatrixXf::Zero(4 * matchSize, 1);
	Eigen::MatrixXf Na(matchSize, matchSize);
	Eigen::MatrixXf W(matchSize, 1);
	Eigen::MatrixXf xbar(5, 1);

	double stopVal = 1e-6;
	std::cout << "if > " << stopVal << " stop" << std::endl;

	uint protectNumber = 0;
	bool endCondition;
	do
	{
		b = pos_r - pos_l;
		calForRotationMatrix(rot_l, rot_matrix_l);
		calForRotationMatrix(rot_r, rot_matrix_r);
		float mlx0 = rot_matrix_l(0, 0), mlx1 = rot_matrix_l(0, 1), mlx2 = rot_matrix_l(0, 2);
		float mly0 = rot_matrix_l(1, 0), mly1 = rot_matrix_l(1, 1), mly2 = rot_matrix_l(1, 2);
		float mrx0 = rot_matrix_r(0, 0), mrx1 = rot_matrix_r(0, 1), mrx2 = rot_matrix_r(0, 2);
		float mry0 = rot_matrix_r(1, 0), mry1 = rot_matrix_r(1, 1), mry2 = rot_matrix_r(1, 2);

		//pre-calculations for sin & cos value of rot_r
		//	  sin(Omega)			 sin(Phi)				sin(Kappa)
		float sO = sin(rot_r(0, 0)), sP = sin(rot_r(1, 0)), sK = sin(rot_r(2, 0));
		//	  cos(Omega)			 cos(Phi)				cos(Kappa)
		float cO = cos(rot_r(0, 0)), cP = cos(rot_r(1, 0)), cK = cos(rot_r(2, 0));

#pragma omp parallel for num_threads(4)
		for (int i = 0; i < matchSize; ++i)
		{
			a1[i] = rot_matrix_l.transpose() * (kp1[i] - principalPoint);
			a2[i] = rot_matrix_r.transpose() * (kp2[i] - principalPoint);
		}

//A & B & P & L matrix
#pragma omp parallel for num_threads(4)
		for (int i = 0; i < matchSize; ++i)
		{
			float u1 = a1[i](0, 0), v1 = a1[i](1, 0), w1 = a1[i](2, 0);
			float u2 = a2[i](0, 0), v2 = a2[i](1, 0), w2 = a2[i](2, 0);
			A(i, 4 * i + 0) = b(0, 0) * (w2 * mlx1 - v2 * mlx2) + b(1, 0) * (-w2 * mlx0 + u2 * mlx2) + b(2, 0) * (v2 * mlx0 - u2 * mlx1);
			A(i, 4 * i + 1) = b(0, 0) * (w2 * mly1 - v2 * mly2) + b(1, 0) * (-w2 * mly0 + u2 * mly2) + b(2, 0) * (v2 * mly0 - u2 * mly1);
			A(i, 4 * i + 2) = b(0, 0) * (v1 * mrx2 - w1 * mrx1) + b(1, 0) * (-u1 * mrx2 + w1 * mrx0) + b(2, 0) * (u1 * mrx1 - v1 * mrx0);
			A(i, 4 * i + 3) = b(0, 0) * (v1 * mry2 - w1 * mry1) + b(1, 0) * (-u1 * mry2 + w1 * mry0) + b(2, 0) * (u1 * mry1 - v1 * mry0);

			L(4 * i + 0, 0) = kp1[i](0, 0);
			L(4 * i + 1, 0) = kp1[i](1, 0);
			L(4 * i + 2, 0) = kp2[i](0, 0);
			L(4 * i + 3, 0) = kp2[i](1, 0);

			B(i, 0) = -(u1 * w2 - u2 * w1);
			//(m1x13*x1 - f*m1w33 + m1y23*y1)*(m2x11*x2 - f*m2w31 + m2y21*y2) - (m1x11*x1 - f*m1w31 + m1y21*y1)*(m2x13*x2 - f*m2w33 + m2y23*y2)
			B(i, 1) = -(u1 * v2 - u2 * v1);

			//orpc : object coordinate in right-photo coordinate system
			auto orpc = kp2[i] - principalPoint;

			float dU2dOmega = 0;
			float dV2dOmega = orpc(0, 0) * (-sO * sK + cO * sP * cK) + orpc(1, 0) * (-sO * cK - cO * sP * sK) + orpc(2, 0) * (-cO * cP);
			float dW2dOmega = orpc(0, 0) * (cO * sK + sO * sP * cK) + orpc(1, 0) * (cO * cK - sO * sP * sK) + orpc(2, 0) * (-sO * cP);
			B(i, 2) = b(0, 0) * (v1 * dW2dOmega - w1 * dV2dOmega) - b(1, 0) * (u1 * dW2dOmega - w1 * dU2dOmega) + b(2, 0) * (u1 * dV2dOmega - v1 * dU2dOmega);

			float dU2dPhi = orpc(0, 0) * (-sP * cK) + orpc(1, 0) * (sP * sK) + orpc(2, 0) * (cP);
			float dV2dPhi = orpc(0, 0) * (sO * cP * cK) + orpc(1, 0) * (-sO * cP * sK) + orpc(2, 0) * (sO * sP);
			float dW2dPhi = orpc(0, 0) * (-cO * cP * cK) + orpc(1, 0) * (cO * cP * sK) + orpc(2, 0) * (-cO * sP);
			B(i, 3) = b(0, 0) * (v1 * dW2dPhi - w1 * dV2dPhi) - b(1, 0) * (u1 * dW2dPhi - w1 * dU2dPhi) + b(2, 0) * (u1 * dV2dPhi - v1 * dU2dPhi);

			float dU2dKappa = orpc(0, 0) * (-cP * sK) + orpc(1, 0) * (-cP * cK);
			float dV2dKappa = orpc(0, 0) * (cO * cK - sO * sP * sK) + orpc(1, 0) * (-cO * sK - sO * sP * sK);
			float dW2dKappa = orpc(0, 0) * (sO * cK + cO * sP * sK) + orpc(1, 0) * (-sO * sK + cO * sP * cK);
			B(i, 4) = b(0, 0) * (v1 * dW2dKappa - w1 * dV2dKappa) - b(1, 0) * (u1 * dW2dKappa - w1 * dU2dKappa) + b(2, 0) * (u1 * dV2dKappa - v1 * dU2dKappa);

			//P(2 * i, 2 * i) = 1.0f / (ACCURACY * ACCURACY);
			//P(2 * i + 1, 2 * i + 1) = 1.0f / (ACCURACY * ACCURACY);
		}

		Na = A * A.transpose();
		auto Nb = B.transpose() * B;
		W = -(A * L);
		xbar = Nb.inverse() * B.transpose() * Na.inverse() * W;
		pos_r(1, 0) += xbar(0, 0);
		pos_r(2, 0) += xbar(1, 0);
		rot_r(0, 0) += xbar(2, 0);
		rot_r(1, 0) += xbar(3, 0);
		rot_r(2, 0) += xbar(4, 0);

		protectNumber++;
		std::cout << "\r"  << " protect number: " << protectNumber << " xbar: " << xbar(0, 0) << std::flush;
		endCondition = 1;
		for (size_t i = 0; i < xbar.rows(); ++i)
		{
			if (xbar(i, 0) > stopVal)
			{
				endCondition = 0;
				break;
			}
		}
	} while (protectNumber < 500 && !endCondition);
	std::cout << std::endl;

	auto V = A.transpose() * Na.inverse() * (W - B * xbar);
	// std::cout << "\n==========pos==========\n"
	// 		  << pos_r << "\n==========rot==========\n"
	// 		  << rot_r << "\n";

	std::cout << "x:" << pos_r(0, 0) << "\t y:" << pos_r(1, 0) << "\t z:" << pos_r(2, 0) 
			  << "\t o:" << rot_r(0, 0) << "\t p:" << rot_r(1, 0) << "\t k:" << rot_r(2, 0) << "\n";

	return 0;
}

void calForRotationMatrix(const Eigen::Matrix<float, 3, 1> &params, Eigen::Matrix3f &opt)
{
	float omega = params(0, 0), phi = params(1, 0), kappa = params(2, 0);
	opt(0, 0) = cos(phi) * cos(kappa);
	opt(0, 1) = cos(omega) * sin(kappa) + sin(omega) * sin(phi) * cos(kappa);
	opt(0, 2) = sin(omega) * sin(kappa) - cos(omega) * sin(phi) * cos(kappa);
	opt(1, 0) = -cos(phi) * sin(kappa);
	opt(1, 1) = cos(omega) * cos(kappa) - sin(omega) * sin(phi) * sin(kappa);
	opt(1, 2) = sin(omega) * cos(kappa) + cos(omega) * sin(phi) * sin(kappa);
	opt(2, 0) = sin(phi);
	opt(2, 1) = -sin(omega) * cos(phi);
	opt(2, 2) = cos(omega) * cos(phi);
}

void pixel2photo(const std::vector<DMatch> &matches, const std::pair<uint, uint> &pixelSize, const std::vector<KeyPoint> &kp1, const std::vector<KeyPoint> &kp2, std::vector<Eigen::Matrix<float, 3, 1>> &opt1, std::vector<Eigen::Matrix<float, 3, 1>> &opt2)
{
	opt1.resize(matches.size());
	opt2.resize(matches.size());
	float x_ratio = FRAME_WIDTH / (float)pixelSize.first;
	float y_ratio = FRAME_HEIGHT / (float)pixelSize.second;

	for (size_t i = 0; i < matches.size(); ++i)
	{
		opt1[i](0, 0) = kp1[matches[i].queryIdx].pt.x * x_ratio - PRINCIPALPOINT_X;
		opt1[i](1, 0) = kp1[matches[i].queryIdx].pt.y * y_ratio - PRINCIPALPOINT_Y;
		opt1[i](2, 0) = 0.0f;

		opt2[i](0, 0) = kp2[matches[i].trainIdx].pt.x * x_ratio - PRINCIPALPOINT_X;
		opt2[i](1, 0) = kp2[matches[i].trainIdx].pt.y * y_ratio - PRINCIPALPOINT_Y;
		opt2[i](2, 0) = 0.0f;
	}
}