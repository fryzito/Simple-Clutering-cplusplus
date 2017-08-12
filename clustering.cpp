#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <utility>
#include <vector>
#include <sys/types.h>
#include <dirent.h> // libreria de linux se descargo i pego en include.
#include <errno.h>
#include <string>

#define dbg(x)std::cout<< #x <<" = "<<x<<std::endl
#define dbg2(x,y)std::cout<<#x<<"="<<x<<" "<<#y<<"="<<y<<std::endl
#define dbg3(x,y,z)std::cout<<#x<<"="<<x<<" "<<#y<<"="<<y<<" "<<#z<<"="<<z<<std::endl

cv::Scalar colorTab[] =
{
	cv::Scalar(0, 0, 255),
	cv::Scalar(0,255,0),
	cv::Scalar(255,100,100),
	cv::Scalar(255,0,255),
	cv::Scalar(0,255,255)
}; // Esto es para dar color.
   
/*Lee un archivo y dos puntos*/
std::vector<cv::Point2f> ReadFile(std::string cadena) {
	std::string line;
	std::ifstream myfile(cadena);
	
	std::vector<cv::Point2f> dataPoint;

	// Ojo lee hasta fin de archivo
	
	while (std::getline(myfile, line)) {

		line[(int)line.find(',')]=' ';

		std::istringstream iss(line);
		int nroFrame;
		double x, y, w, h;
		iss >> nroFrame >> x >> y >> w >> h;
		dbg(line);
		dbg3(nroFrame,x,y);
		dbg2(w,h);

		//cv::Point2f pnt = cv::Point2f(x,y);
		//dataPoint.push_back(pnt);

	}
	myfile.close();

	return dataPoint;
}

// Funcion Para leer files de ReadFiles
int getdir(std::string dir, std::vector<std::string> &files)
{
	DIR *dp;
	struct dirent *dirp;
	if ((dp = opendir(dir.c_str())) == NULL) {
		std::cout << "Error(" << errno << ") opening " << dir << std::endl;
		return errno;
	}

	while ((dirp = readdir(dp)) != NULL) {
		files.push_back(std::string(dirp->d_name));
	}
	closedir(dp);
	return 0;
}

/*Recibe una direccion y lista todos los archivos dentro de esa carpeta*/
std::vector<std::string> ReadFiles(std::string str) {
	std::string dir = std::string(str); // Direccion desde la cual vamos a leer todos los archivos
	std::vector<std::string> files = std::vector<std::string>();
	getdir(dir, files);
	std::vector<std::string> ans;
	for (unsigned int i = 0; i < files.size(); i++) {
		if (files[i].size() != 1 && files[i].size() != 2)
			ans.push_back(files[i]);
	}
	return ans;
}

// Retorna la lista de los archivos .txt ya para leer
std::vector<std::string> readDir(std::string str) {
	std::vector<std::string> listFiles = ReadFiles(str);

	std::vector<std::string> ans;
	for (int i = 0; i < listFiles.size(); i++) {
		if (listFiles[i].find(".txt") != -1) {
			//std::cout << listFiles[i] << std::endl;
			ans.push_back(listFiles[i]);
		}
	}
	return ans;
}

// Pintar matriz para affinity
void printfMatrix(std::vector<std::vector<double>> B) {
	for (int i = 0; i<B.size(); i++)
		for (int j = 0; j<B[i].size(); j++) {
			printf("%.3lf ", B[i][j]);
		}
	putchar('\n');
}

// Modulo Affinity
void Affinity(std::vector<cv::Point2f> Puntos, std::vector<int> &idx) {
 
	int N = Puntos.size();
	std::vector<double> aux(2);
	std::vector<std::vector<double>> dataPoint(N,aux);
	for (int i = 0; i < Puntos.size();i++) { // Pasando parametros a dataPoint
		dataPoint[i][0] = Puntos[i].x;
		dataPoint[i][1] = Puntos[i].y;
	}
	// Inicializando matrices
	std::vector<double> aux3(N,0.0);
	std::vector<std::vector<double>> S(N, aux3);
	std::vector<std::vector<double>> R(N, aux3);
	std::vector<std::vector<double>> A(N, aux3);
	
	int iter = 50; // Numero de iteraciones
	double lambda = 0.9;

	int size = N*(N - 1) / 2;
	std::vector<double> tmpS;
	//compute similarity between data point i and j (i is not equal to j)
	for (int i = 0; i<N - 1; i++) {
		for (int j = i + 1; j<N; j++) {
			S[i][j] = -((dataPoint[i][0] - dataPoint[j][0])*(dataPoint[i][0] - dataPoint[j][0]) + (dataPoint[i][1] - dataPoint[j][1])*(dataPoint[i][1] - dataPoint[j][1]));
			S[j][i] = S[i][j];
			tmpS.push_back(S[i][j]);
		}
	}
	//compute preferences for all data points: median 
	sort(tmpS.begin(), tmpS.end());
	double median = 0;

	if (size % 2 == 0)
		median = (tmpS[size / 2] + tmpS[size / 2 - 1]) / 2;
	else
		median = tmpS[size / 2];
	for (int i = 0; i<N; i++) S[i][i] = median;

	//// Segunda Parte
	//printfMatrix(A,N);

	for (int m = 0; m<iter; m++) {
		//update responsibility
		for (int i = 0; i<N; i++) {
			for (int k = 0; k<N; k++) {
				double max = -1e100;
				// Sacamos la maximo suma dos a dos de cada subconjunto menores a k
				for (int kk = 0; kk<k; kk++) {
					if (S[i][kk] + A[i][kk]>max)
						max = S[i][kk] + A[i][kk];
				}
				// sacamos la maxima suma dos a dos de cada subconjunto mayores a k
				for (int kk = k + 1; kk<N; kk++) {
					if (S[i][kk] + A[i][kk]>max)
						max = S[i][kk] + A[i][kk];
				}
				// lambda=0.9
				R[i][k] = (1 - lambda)*(S[i][k] - max) + lambda*R[i][k];
			}
		}
		//update availability
		for (int i = 0; i<N; i++) {
			for (int k = 0; k<N; k++) {
				if (i == k) {
					double sum = 0.0;
					for (int ii = 0; ii<i; ii++) {
						sum += std::max<double>(0.0, R[ii][k]);
					}
					for (int ii = i + 1; ii<N; ii++) {
						sum += std::max<double>(0.0, R[ii][k]);
					}
					A[i][k] = (1 - lambda)*sum + lambda*A[i][k];
				}
				else {
					double sum = 0.0;
					int maxik = std::max<double>(i, k);
					int minik = std::min<double>(i, k);
					for (int ii = 0; ii<minik; ii++) {
						sum += std::max<double>(0.0, R[ii][k]);
					}
					for (int ii = minik + 1; ii<maxik; ii++) {
						sum += std::max<double>(0.0, R[ii][k]);
					}
					for (int ii = maxik + 1; ii<N; ii++) {
						sum += std::max<double>(0.0, R[ii][k]);
					}
					A[i][k] = (1 - lambda)*std::min<double>(0.0, R[k][k] + sum) + lambda*A[i][k];
				}
			}
		}
		//printfMatrix(A,N);
	}

	//find the exemplar
	std::vector<std::vector<double>> E(N,aux3);
	std::vector<int> center;
	for (int i = 0; i<N; i++) {
		E[i][i] = R[i][i] + A[i][i];
		if (E[i][i]>0) {
			center.push_back(i);
		}
	}
	//data point assignment, idx[i] is the exemplar for data point i
	for (int i = 0; i<N; i++) {
		int idxForI = 0;
		double maxSim = -1e100;
		for (int j = 0; j<center.size(); j++) {
			int c = center[j];
			if (S[i][c]>maxSim) {
				maxSim = S[i][c];
				idxForI = c;
			}
		}
		idx[i] = idxForI+1;
	}
}


void processFile(std::string inputFile) {
	
	std::vector<cv::Point2f> Read_Pnts = ReadFile(inputFile); // Lee el file de arrriba
	dbg(Read_Pnts.size());
	/*
	const int MAX_CLUSTERS = 3; // obsiona por el momento no se esta utilizando
	cv::Mat img(500, 500, CV_8UC3);  // Definicion de tamaño y tipo

	int i, sampleCount = Read_Pnts.size();

	// CV::Mat (int row,int col,int type)
	cv::Mat points(sampleCount, 1, CV_32FC2); // Labels guarda los indices de los clusters

	cv::Mat centers; // Para pintar los puntos de los clusters

	std::cout << sampleCount << std::endl;

	// set points
	int k;
	for (k = 0; k < sampleCount; k++) {
		points.at<cv::Vec2f>(k, 0) = Read_Pnts[k];
	}

	// get points
	for (k = 0; k < sampleCount; k++) {
		std::cout << points.at<cv::Vec2f>(k, 0) << std::endl;
	}

	// Aplica Affinity Propagation, se tiene que hablar del criterio de compraracion (en este caso distancia euclidea).
	std::vector<int> labels(Read_Pnts.size());
	Affinity(Read_Pnts,labels);

	img = cv::Scalar::all(0);

	// Modulo para pintar en el recuadro, falta mapear los valores para aveces se sale del rango
	for (i = 0; i < sampleCount; i++)
	{
		int clusterIdx = labels[i];
		cv::Point ipt = points.at<cv::Point2f>(i);
		circle(img, ipt, 2, colorTab[clusterIdx], cv::FILLED, cv::LINE_AA);
		std::cout << clusterIdx << std::endl;
	}

	imshow("clusters", img);
	*/
}

int main(int /*argc*/, char** /*argv*/)
{

	// Leer todos los files de esta dir
	std::string Endereco = "D:\\DataSet Laboratory\\people\\1";
	std::vector<std::string> listDir =  ReadFiles(Endereco);

	for (int i = 0; i < listDir.size(); i++) {

		// Listando los directorios
		if (listDir[i].find(".")==-1){
			// Entrando a cada carpeta y listando,lisDir tiene el nombre de las carpetas
			std::cout << listDir[i] << std::endl;
			
			// readDir Regresa una lista con los archivos .txt para leer de lisDir_i
			std::vector<std::string> archivos = readDir(Endereco+"\\"+listDir[i]);
			// Pasta tiene lista de archivos de  una carpeta
			std::string enderecoSum = "";
			enderecoSum += Endereco;
			enderecoSum += "\\";
			enderecoSum += listDir[i];

			for (int j = 0; j < archivos.size(); j++) {

				// Aqui tenemos el archivo a ser procesado para lectora de puntos de cada tracklet
				std::cout << archivos[j] << std::endl;
				processFile(enderecoSum+"\\"+archivos[j]); 

				// Sugerencia hacer un wait para pasar a la siguiente iteracion


			}
		}
	}

	return 0;
}
