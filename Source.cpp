#include <iostream>
#include <vector>
#include <string>
#include <complex>
#include <math.h>
#include <cmath>
#include <cassert>
#include <fstream>

#include <cstdlib>   // for rand(), srand()
#include <ctime>     // for time()
// Udated 2/3/2022 Added Fully connected layer computation and softmax
// Modified by Prof. Murali Subbarao, ESE 344
using namespace std;
// allocate memory for a 1d vector
void alloc1d(vector<double>& tn1d, int s1) {
	// allocate memory for a 1d tensor tn1d of size s1
	tn1d.resize(s1);
}
// read data for a 1d tensor
void read1d(ifstream& f, vector<double>& tn1d, int s1) {
	for (int i1 = 0; i1 < s1; i1++)
	{
		f >> tn1d[i1];
	}
}
// allocate memory for a 2d tensor
void alloc2d(vector<vector<double>>& tn2d, int s1, int s2) {
	// allocate memory for a 2d tensor tn2d of size s1, s2
	tn2d.resize(s1);
	for (int i1 = 0; i1 < s1; i1++) {
		tn2d[i1].resize(s2);
	}
}
// read data for a 2d tensor
void read2d(ifstream& f, vector<vector<double>>& tn2d, int s1, int s2) {

	for (int i2 = 0; i2 < s2; i2++) {
		for (int i1 = 0; i1 < s1; i1++)
		{
			f >> tn2d[i1][i2];
		}
	}
}
// allocate memory for a 3d tensor
void alloc3d(vector<vector<vector<double>>>& tn3d, int s1, int s2, int s3) {
	// allocate memory for a 3d tensor tn3d of size s1, s2, s3
	tn3d.resize(s1);
	for (int i1 = 0; i1 < s1; i1++) {
		tn3d[i1].resize(s2);
		for (int i2 = 0; i2 < s2; i2++) {
			tn3d[i1][i2].resize(s3);
		}
	}
}
// read data for a 3d tensor
void read3d(ifstream& f, vector<vector<vector<double>>>& tn3d, int s1, int s2, int s3) {
	for (int i3 = 0; i3 < s3; i3++) {
		for (int i2 = 0; i2 < s2; i2++) {
			for (int i1 = 0; i1 < s1; i1++)
			{
				f >> tn3d[i1][i2][i3];
			}
		}
	}
}
// allocate memory for a 4d tensor tn4d of size s1, s2, s3, s4
void alloc4d(vector<vector<vector<vector<double>>>>& tn4d, int s1, int s2, int s3, int s4) {

	tn4d.resize(s1);

	for (int i1 = 0; i1 < s1; i1++) {
		tn4d[i1].resize(s2);
		for (int i2 = 0; i2 < s2; i2++) {
			tn4d[i1][i2].resize(s3);
			for (int i3 = 0; i3 < s3; i3++)
			{
				tn4d[i1][i2][i3].resize(s4);
			}
		}
	}
}
// read a 4d tensor tn4d of size s1, s2, s3, s4
void read4d(ifstream& f, vector<vector<vector<vector<double>>>>& tn4d, int s1, int s2, int s3, int s4) {

	for (int i4 = 0; i4 < s4; i4++) {
		for (int i3 = 0; i3 < s3; i3++) {
			for (int i2 = 0; i2 < s2; i2++) {
				for (int i1 = 0; i1 < s1; i1++) {
					f >> tn4d[i1][i2][i3][i4];
				}
			}
		}
	}
}

class ConvNet {
public:
	vector<vector<vector<double>>> D1;
	// Stage 1
	// Layer L1
	void M1(ifstream& f) {
		alloc3d(D1, 32, 32, 3);
		read3d(f, D1, 32, 32, 3);
	}
	//Stage 2
	//Layer L2
	vector<vector<vector<double>>> D2;
	vector<vector<vector<vector<double>>>> D3;
	vector<double> D4;
	void M2(ifstream& f) {
		alloc4d(D3, 5,5,3,16);
		read4d(f, D3, 5,5,3,16);
	}
	void M3(ifstream& f) {
		alloc1d(D4, 16);
		read1d(f, D4, 16);
	}
	void M4() {
		int stride = 1;

		alloc3d(D2, 32, 32, 16);
		//tn1 = 3d (d1)
		//tn2 = 4d (d3)
		//tn3 = 3d (d2)
		//tn4 = n/a

		int tn1s1 = 32, tn1s2 = 32, tn1s3 = 3;
		int tn2s1 = 5, tn2s2 = 5, tn2s3 = tn1s3, tn2s4 = 16;
		int tn2s1by2 = tn2s1 / 2;   // 2
		int tn2s2by2 = tn2s2 / 2;   // 2
		int tn3s1 = tn1s1 / stride, tn3s2 = tn1s2 / stride, tn3s3 = tn2s4;
		int tn4s1 = tn1s1, tn4s2 = tn1s2, tn4s3 = tn1s3, tn4s4 = 10;
		cout << "Output of Convolution layer: tn3" << endl;
		for (int tn2i4 = 0, tn3i3 = 0; tn2i4 < tn2s4; tn2i4++, tn3i3++) {
			for (int tn1i1 = 0, tn3i1 = 0; tn1i1 < tn1s1; tn1i1 += stride, tn3i1++) {
				//cout << "test M4" << endl;
					for (int tn1i2 = 0, tn3i2 = 0; tn1i2 < tn1s2; tn1i2 += stride, tn3i2++) {
						double tmpsum = 0.0;
						for (int tn2i3 = 0; tn2i3 < tn2s3; tn2i3++) {
                            // note tn1s3=tn2s3
								for (int tn2i1 = -tn2s1by2; tn2i1 <= tn2s1by2; tn2i1++) {
									for (int tn2i2 = -tn2s2by2; tn2i2 <= tn2s2by2; tn2i2++) {
										if (((tn1i1 + tn2i1) >= 0) && ((tn1i1 + tn2i1) < tn1s1) && ((tn1i2 + tn2i2) >= 0) && ((tn1i2 + tn2i2) <tn1s1)) { // zero padding of tn1
										
												tmpsum += D3[tn2i1 + tn2s1by2][tn2i2 + tn2s2by2][tn2i3][tn2i4] * D1[tn1i1 + tn2i1][tn1i2 + tn2i2][tn2i3];
										}
									}
								}
						}
						D2[tn3i1][tn3i2][tn3i3] = tmpsum;
						D2[tn3i1][tn3i2][tn3i3] += D4[tn3i3];
						cout << tmpsum << "  ";
					}
					cout << endl;
			}
			cout << endl << endl;
		}
	}
	//Layer L3
	vector<vector<vector<double>>> D5;
	void M5() {
		alloc3d(D5, 32, 32, 16);
		int stride = 1;
		for (int k = 0; k < 16; k++) {  // for each cross section k
			for (int m = 0, i = 0; m < 32; m += stride, i++) { //for row m
				for (int n = 0, j = 0; n < 32; n += stride, j++) { // for column n
					D5[m][n][k] = max(D2[m][n][k], 0.0);
				}
			}
		}
	}
	// Layer L4
	vector<vector<vector<double>>> D6;
	void M6() {
		alloc3d(D6, 16, 16, 16);
		int stride = 2;
		for (int k = 0; k < 16; k++) {  // for each cross section k
			for (int m = 0, i = 0; m < 16; m += stride, i++) { //for row m
				for (int n = 0, j = 0; n < 16; n += stride, j++) { // for column n
					D6[i][j][k] = max(max(D5[m][n][k], D5[m + 1][n][k]), max(D5[m][n + 1][k], D5[m + 1][n + 1][k]));
				}
			}
		}
	}
	// Layer L5
	vector<vector<vector<double>>> D7;
	vector<vector<vector<vector<double>>>> D8;
	vector<double> D9;
	void M7(ifstream& f) {
		alloc4d(D8, 5, 5, 16, 20);
		read4d(f, D8, 5, 5, 16, 20);
	}
	void M8(ifstream& f) {
		alloc1d(D9, 20);
		read1d(f, D9, 20);
	}
	void M9() {
		alloc3d(D7, 16, 16, 20);
		int stride = 1;

		int tn1s1 = 16, tn1s2 = 16, tn1s3 = 16;
		int tn2s1 = 5, tn2s2 = 5, tn2s3 = tn1s3, tn2s4 = 20;
		int tn2s1by2 = tn2s1 / 2; // 2
		int tn2s2by2 = tn2s2 / 2; // 2
		int tn3s1 = tn1s1 / stride, tn3s2 = tn1s2 / stride, tn3s3 = tn2s4;

		for (int tn2i4 = 0, tn3i3 = 0; tn2i4 < tn2s4; tn2i4++, tn3i3++) {
			//cout << "Test M9";
			for (int tn1i1 = 0, tn3i1 = 0; tn1i1 < tn1s1; tn1i1 += stride, tn3i1++) {
				for (int tn1i2 = 0, tn3i2 = 0; tn1i2 < tn1s2; tn1i2 += stride, tn3i2++) {
					double tmpsum = 0.0;
					for (int tn2i3 = 0; tn2i3 < tn2s3; tn2i3++) {
						// note tn1s3=tn2s3
						for (int tn2i1 = -tn2s1by2; tn2i1 <= tn2s1by2; tn2i1++) {
							for (int tn2i2 = -tn2s2by2; tn2i2 <= tn2s2by2; tn2i2++) {
								if (((tn1i1 + tn2i1) >= 0) && ((tn1i1 + tn2i1) < tn1s1) && ((tn1i2 + tn2i2) >= 0) && ((tn1i2 + tn2i2) < tn1s1)) { // zero padding of tn1
									tmpsum += D8[tn2i1 + tn2s1by2][tn2i2 + tn2s2by2][tn2i3][tn2i4] * D6[tn1i1 + tn2i1][tn1i2 + tn2i2][tn2i3];
								}
							}
						}
					}
					D7[tn3i1][tn3i2][tn3i3] = tmpsum;
					D7[tn3i1][tn3i2][tn3i3] += D9[tn3i3];
				}
			}
		}
	}
	// Layer L6
	vector<vector<vector<double>>> D10;
	void M10() {
		alloc3d(D10, 16, 16, 20);
		int stride = 2;
		for (int k = 0; k < 20; k++) {  // for each cross section k
			for (int m = 0, i = 0; m < 16; m += stride, i++) { //for row m
				for (int n = 0, j = 0; n < 16; n += stride, j++) { // for column n
					D10[m][n][k] = max(0.0, D7[m][n][k]);
				}
			}
		}
	}
	// Layer L7
	vector<vector<vector<double>>> D11;
	void M11() {
		alloc3d(D11, 8, 8, 20);
		int stride = 2;
		for (int k = 0; k < 20; k++) {  // for each cross section k
			for (int m = 0, i = 0; m < 8; m += stride, i++) { //for row m
				for (int n = 0, j = 0; n < 8; n += stride, j++) { // for column n
					D11[i][j][k] = max(max(D10[m][n][k], D10[m + 1][n][k]), max(D10[m][n + 1][k], D10[m + 1][n + 1][k]));
				}
			}
		}
	}
	// Layer L8
	vector<vector<vector<double>>> D12;
	vector<vector<vector<vector<double>>>> D13;
	vector<double> D14;
	void M12(ifstream& f) {
		alloc4d(D13, 5, 5, 20, 20);
		read4d(f, D13, 5, 5, 20, 20);
	}
	void M13(ifstream& f) {
		alloc1d(D14, 20);
		read1d(f, D14, 20);
	}
	void M14() {
		//tn1 = 3d (D11)
		//tn2 = 4d (D13)
		//tn3 = 3d (D12)
		//tn4 = n/a

		alloc3d(D12, 8, 8, 20);

		int stride = 1;

		int tn1s1 = 8, tn1s2 = 8, tn1s3 = 20;
		int tn2s1 = 5, tn2s2 = 5, tn2s3 = 20, tn2s4 = 20;
		int tn2s1by2 = tn2s1 / 2;   // 2
		int tn2s2by2 = tn2s2 / 2;   // 2
		int tn3s1 = tn1s1 / stride, tn3s2 = tn1s2 / stride, tn3s3 = tn2s4;
		int tn4s1 = tn1s1, tn4s2 = tn1s2, tn4s3 = tn1s3, tn4s4 = 10;
		cout << "Output of Convolution layer: tn3" << endl;
		for (int tn2i4 = 0, tn3i3 = 0; tn2i4 < tn2s4; tn2i4++, tn3i3++) {
			for (int tn1i1 = 0, tn3i1 = 0; tn1i1 < tn1s1; tn1i1 += stride, tn3i1++) {
				for (int tn1i2 = 0, tn3i2 = 0; tn1i2 < tn1s2; tn1i2 += stride, tn3i2++) {
					double tmpsum = 0.0;
					for (int tn2i3 = 0; tn2i3 < tn2s3; tn2i3++) {
						// note tn1s3=tn2s3
						for (int tn2i1 = -tn2s1by2; tn2i1 <= tn2s1by2; tn2i1++) {
							for (int tn2i2 = -tn2s2by2; tn2i2 <= tn2s2by2; tn2i2++) {
								if (((tn1i1 + tn2i1) >= 0) && ((tn1i1 + tn2i1) < tn1s1) && ((tn1i2 + tn2i2) >= 0) && ((tn1i2 + tn2i2) < tn1s1)) { // zero padding of tn1
									//cout << "test m14" << endl;
									tmpsum += D13[tn2i1 + tn2s1by2][tn2i2 + tn2s2by2][tn2i3][tn2i4] * D11[tn1i1 + tn2i1][tn1i2 + tn2i2][tn2i3];
								}
							}
						}
					}
					
					D12[tn3i1][tn3i2][tn3i3] = tmpsum;
					D12[tn3i1][tn3i2][tn3i3] += D14[tn3i3];
					cout << tmpsum << "  ";
				}
				cout << endl;
			}
			cout << endl << endl;
		}
	}
	// Layer L9
	vector<vector<vector<double>>> D15;
	void M15() {
		alloc3d(D15, 8, 8, 20);
		int stride = 2;
		for (int k = 0; k < 20; k++) {  // for each cross section k
			for (int m = 0, i = 0; m < 8; m += stride, i++) { //for row m
				for (int n = 0, j = 0; n < 8; n += stride, j++) { // for column n
					D15[m][n][k] = max(0.0, D12[m][n][k]);
				}
			}
		}
	}
	// Layer L10
	vector<vector<vector<double>>> D16;
	void M16() {
		alloc3d(D16, 8, 8, 20);
		int stride = 2;
		for (int k = 0; k < 20; k++) {  // for each cross section k
			for (int m = 0, i = 0; m < 8; m += stride, i++) { //for row m
				for (int n = 0, j = 0; n < 8; n += stride, j++) { // for column n
					D16[i][j][k] = max(max(D15[m][n][k], D15[m + 1][n][k]), max(D15[m][n + 1][k], D15[m + 1][n + 1][k]));
				}
			}
		}
	}
	// Layer L11
	vector<double> D17;
	vector<vector<vector<vector<double>>>> D18;
	vector<double> D19;
	void M17(ifstream& f) {
		alloc4d(D18, 4, 4, 20, 10);
		read4d(f, D18, 4, 4, 20, 10);
	}
	void M18(ifstream& f) {
		alloc1d(D19, 10);
		read1d(f, D19, 10);
	}
	void M19() {
		alloc1d(D17, 10);

		for (int tn4i4 = 0; tn4i4 < 10; tn4i4++) {
			// note tn1s1=tn4s1 tn1s2=tn4s2, tn1s3=tn4s3
			double tmpsum = 0.0;
			for (int tn4i3 = 0; tn4i3 < 20; tn4i3++) {
				for (int tn4i1 = 0; tn4i1 < 4; tn4i1++) {
					for (int tn4i2 = 0; tn4i2 < 4; tn4i2++) {
						tmpsum += D18[tn4i1][tn4i2][tn4i3][tn4i4] * D16[tn4i1][tn4i2][tn4i3];
					}
				}
			}
			D17[tn4i4] = tmpsum + D19[tn4i4];
		}
	}
	// Layer L12
	vector<double> D20;
	void M20() {
		double tempsum = 0;
		int i;
		for (i = 0; i < 10; i++) {
			tempsum += (D17[i] * D17[i]);
		}
		tempsum = sqrt(tempsum);
		for (i = 0; i < 10; i++) {
			D17[i] /= tempsum;
		}
	}
	void M21() {
		alloc1d(D20, 10);
		int i;
		double tempsum = 0;
		for (i = 0; i < 10; i++) {
			tempsum += (exp(D17[i]));
		}
		for (i = 0; i < 10; i++) {
			D20[i] = (exp(D17[i])) / tempsum;
		}
	}
	void M22(ostream& output_file) {
		output_file << endl;
		int i;
		for (i = 0; i < 10; i++)
		{
			output_file << D20[i] << "   ";
		}
		output_file << endl;
	}

private:


};

int main() {
	ifstream input_file;
	ofstream output_file;
	input_file.open("project1inputdata.txt");
	output_file.open("proj1outputdata.txt");

	ConvNet ConvNet_File;
	ConvNet_File.M1(input_file);
	ConvNet_File.M2(input_file);
	ConvNet_File.M3(input_file);
	ConvNet_File.M4();
	ConvNet_File.M5();
	ConvNet_File.M6();
	ConvNet_File.M7(input_file);
	ConvNet_File.M8(input_file);
	ConvNet_File.M9();
	ConvNet_File.M10();
	ConvNet_File.M11();
	ConvNet_File.M12(input_file);
	ConvNet_File.M13(input_file);
	ConvNet_File.M14();
	ConvNet_File.M15();
	ConvNet_File.M16();
	ConvNet_File.M17(input_file);
	ConvNet_File.M18(input_file);
	ConvNet_File.M19();
	ConvNet_File.M20();
	ConvNet_File.M21();
	ConvNet_File.M22(output_file);

	return 1;
}


