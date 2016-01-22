#include "isomap_lxw.h"
#include <vnl/vnl_index_sort.h>
#include <vnl/algo/vnl_symmetric_eigensystem.h>
#include <algorithm>
template<typename T>
T min2(const T& AM,const T& BM){
	int N=AM.rows();
	T CM(N,N);
	for (unsigned int rx = 0; rx <N ; ++rx)
		for (unsigned int cx = 0; cx <rx; ++cx)
		{
			double	minval	=std::min(AM(rx,cx),BM(rx,cx));
			CM(rx,cx)=minval;
			CM(cx,rx)=minval;
		}
	return CM;
}
VectorType MtoV(MatrixType D){
	VectorType v(D.size());
	VectorType::iterator iter1=v.begin();
	for (MatrixType::iterator iter = D.begin(); iter!=D.end() ; ++iter,++iter1)
		*iter1=*iter;
	return v;
}

template<typename T> T square_function (T a)
{
	return a*a;
}

MatrixType repmatRow(VectorType v,int N){
	int M=v.size();
	MatrixType temp(N,M);
	for (unsigned int i = 0; i <N ; ++i)
	{	
		temp.set_row(i,v);
	}
	return temp;
}
MatrixType repmatColumn(VectorType v,int N){
	int M=v.size();
	MatrixType temp(M,N);
	for (unsigned int i = 0; i <N ; ++i)
	{	
		temp.set_column(i,v);
	}
	return temp;
}

VectorType SumM(MatrixType M){
	int N=M.columns();
	VectorType v(N);
	for (unsigned int i = 0; i <N ; ++i)
		v(i)=M.get_column(i).sum();
	return v;
}

template<typename T> T L2_distance(T &AM,T &BM,int df){
	int M=AM.columns();
	int N=BM.columns();
	T CM(M,N);
	T temp1,temp2;
	temp1=AM.apply(square_function);
	temp2=BM.apply(square_function);
	temp1=repmatColumn(SumM(temp1),N);
	temp2=repmatRow(SumM(temp2),M);
	CM=temp1+temp2-2.0*AM.transpose()*BM;
	CM=CM.apply(sqrt);
	if (df==1)
		CM.fill_diagonal(0.0);
	return CM;
}

double computePearsonCorrelation( VectorType x, VectorType y )
{
	unsigned int numSubjects = x.size();
	// first compute the means
	double dXM = 0;
	double dYM = 0;
	for ( unsigned int sub=0; sub<numSubjects; sub++ )
	{
		dXM += x[sub];
		dYM += y[sub];
	}
	dXM/=numSubjects;
	dYM/=numSubjects;
	// now compute the variances
	double dXV = 0;
	double dYV = 0;
	for ( unsigned int sub=0; sub<numSubjects; sub++ )
	{
		dXV+=(x[sub]-dXM)*(x[sub]-dXM);
		dYV+=(y[sub]-dYM)*(y[sub]-dYM);
	}
	dXV/=(numSubjects-1);
	dYV/=(numSubjects-1);
	// which gives the standard deviations
	double dXS = sqrt( dXV );
	double dYS = sqrt( dYV );
	// from which we can compute the correlation coefficient
	double dR = 0;
	for ( unsigned int sub=0; sub<numSubjects; sub++ )
	{
		dR+= (x[sub]-dXM)*(y[sub]-dYM);
	}
	dR/=(numSubjects-1)*dXS*dYS;
	return dR;
}

template<typename T> 
T run_isomap(T DValM,int K,int dims,VectorType &R){
	//construct neighborhood graph
	int	N=	DValM.rows();
	T Ycoords(dims,N);
	int INF=1000*DValM.max_value()*N;
	typedef vnl_matrix<int> IndexMatrixType;
	typedef vnl_index_sort<double,int> IndexSortType;
	IndexMatrixType sortedIndicesM;
	T sortedValM;
	IndexSortType indexSort;
	indexSort.matrix_sort(
			IndexSortType::ByColumn, DValM, sortedValM, sortedIndicesM);
	for (unsigned int rx = 0; rx <N ; ++rx)
		for (unsigned int cx = 1+K; cx < N; ++cx)
		{
			DValM(rx, sortedIndicesM(cx,rx)) =INF; 
		}
	DValM=min2(DValM,DValM.transpose());//对称
	//compute shortest paths
	for(int k=0;k<N;++k)
	{	
		MatrixType temp1,temp2;
		temp1=repmatColumn(DValM.get_column(k),N);
		temp2=repmatRow(DValM.get_row(k),N);
		DValM=min2(DValM,temp1+temp2);
	}
	DValM.fill_diagonal(0.0);
	MatrixType DValM2;
	DValM2=DValM.apply(square_function);
	MatrixType tM(N,N);
	tM.fill(-1.0/N).fill_diagonal(1.0-1.0/N);
	tM=-tM*DValM2*tM/2.0;
	vnl_symmetric_eigensystem<double> eig(tM);
	vnl_diag_matrix<double> val=eig.D;
	MatrixType vec=eig.V;
	for (unsigned int i=0;i<dims;++i)
	{	unsigned int j=N-i-1;
		VectorType temp = sqrt(abs(val(j)))*vec.get_column(j);
		Ycoords.set_row(i,temp);
	}
	for (unsigned int i=0;i<dims;++i)
	{	MatrixType Yc=Ycoords.extract(i+1,N);
		MatrixType DY=L2_distance(Yc,Yc,0);
		R(i)=computePearsonCorrelation(MtoV(DY),MtoV(DValM));
		R(i)=1.0-R(i)*R(i);
	}
	return Ycoords;
}

