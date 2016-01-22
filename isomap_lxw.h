//isomap_lxw.h
//Created by Xinwei Li. All rights reserved.
#ifndef isomap_lxw_h
#define isomap_lxw_h

#include <vnl/vnl_matrix.h>
#include <vnl/vnl_vector.h>

typedef vnl_matrix<double> MatrixType;
typedef vnl_vector<double> VectorType;

template<typename T> T L2_distance(T &AM,T &BM,int df);
template<typename T> 
T run_isomap(T DValM,int K,int dims,VectorType & R);
template<typename T>
T min2(const T& AM,const T& BM);//min matrix of two matrix

VectorType SumM(MatrixType M);//sum of a matrix,return a vector

template<typename T> T square_function (T a);//a^2

MatrixType repmatRow(VectorType v,int N);//row vector repeat N times
MatrixType repmatColumn(VectorType v,int N);//column vector repeat N times

VectorType MtoV(MatrixType D);//matrix to a vector
double computePearsonCorrelation( VectorType x, VectorType y );
#endif
