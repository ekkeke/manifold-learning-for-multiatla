// C/C++ File
// AUTHOR:   Xinwei Li -lixinwei9046@gmail.com 
// FILE:     ManifoldlearningLocalWeightImageFilter.txx
// ROLE:     TODO (some explanation)
// CREATED:  2014-05-03 16:07:11
// MODIFIED: 2014-05-12 15:46:15
#include <itkNeighborhoodIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <vnl/vnl_matrix.h>
#include "isomap_lxw.h"
#include "isomap_lxw.cxx"

#include <set>
#include <map>
#include <vector>
using namespace std;

	template <class TInputImage, class TOutputImage>
void ManifoldlearningLocalWeightImageFilter<TInputImage, TOutputImage>::UpdateInputs()
{
	// Set all the inputs
	this->SetNumberOfIndexedInputs(1+2*m_Atlases.size());//targetimage +atlases brain +atlases seg

	size_t kInput=0;
	this->SetNthInput(kInput++,m_Target);
	for(size_t i=0; i<m_Atlases.size();i++)
	{	
		this->SetNthInput(kInput++,m_Atlases[i]);
	}
	for (size_t i=0;i<m_AtlasSegs.size();i++)
	{
		this->SetNthInput(kInput++,m_AtlasSegs[i]);
	}
}

	template <class TInputImage, class TOutputImage>
void ManifoldlearningLocalWeightImageFilter<TInputImage,TOutputImage>::GenerateInputRequestedRegion()
{
	Superclass::GenerateInputRequestedRegion();

	//Get the output requested region
	RegionType outRegion =this->GetOutput()->GetRequestedRegion();

	//Pad this region by the search window and patch size
	outRegion.PadByRadius(m_SearchRadius);
	outRegion.PadByRadius(m_PatchRadius);

	//Iterate over all the inputs to this filter
	for(size_t i =0;i<this->GetNumberOfInputs();i++)
	{
		InputImageType *input=const_cast<InputImageType *>(this->GetInput(i));
		RegionType region=outRegion;
		region.Crop(input->GetLargestPossibleRegion());
		input->SetRequestedRegion(region);
	}
}

	template <class TInputImage, class TOutputImage>
void ManifoldlearningLocalWeightImageFilter<TInputImage,TOutputImage>::ComputeOffsetTable(
		const InputImageType *image,
		const SizeType &radius,
		int **offset,
		size_t &nPatch,
		int **manhattan)
{
	//Use iterators to construct offset tables
	RegionType r=image->GetBufferedRegion();
	NIter itTempPatch(radius,image,r);

	//Position the iterator in the middle to avoid probelems with boundary conditions
	IndexType iCenter;
	for(size_t i=0;i<InputImageDimension;i++)
		iCenter[i]=r.GetIndex(i)+r.GetSize(i)/2;
	itTempPatch.SetLocation(iCenter);

	//Compute the offsets
	nPatch=itTempPatch.Size();
	(*offset)=new int[nPatch];
	if(manhattan)
		(*manhattan)=new int[nPatch];
	for(size_t i=0;i<nPatch;i++)
	{
		(*offset)[i]=itTempPatch[i]-itTempPatch.GetCenterPointer();
		if (manhattan)
		{
			typename NIter::OffsetType off=itTempPatch.GetOffset(i);
			for (int d=0;d<InputImageDimension;d++)
				(*manhattan)[i]+=abs(off[d]);
		}
	}
}

	template <class TInputImage,class TOutputImage>
void ManifoldlearningLocalWeightImageFilter<TInputImage,TOutputImage>::GenerateData()
{
	//Allocate the output
	this->GetOutput()->SetBufferedRegion(this->GetOutput()->GetRequestedRegion());
	this->GetOutput()->Allocate();

	//Get the number of atlases
	int n=m_AtlasSegs.size();

	NIter itTarget(m_PatchRadius,m_Target,m_Target->GetRequestedRegion());

	//Construct offset tabels for all the images
	size_t nPatch,nSearch;
	int *offPatchTarget,
			**offPatchAtlas=new int *[n],
			**offPatchSeg=new int *[n],
			**offSearchAtlas=new int *[n],
			*manhattan;

	//Compute the offset table for the target image
	ComputeOffsetTable(m_Target,m_PatchRadius,&offPatchTarget,nPatch);

	//Collect search statistics
	std::vector<int> searchHisto(100,0);

	//Find all unique labels in the requested region
	std::set<InputImagePixelType> labelset;
	for(int i=0;i<n;i++)
	{
		//Find all the labels
		const InputImageType *seg=m_AtlasSegs[i];
		itk::ImageRegionConstIteratorWithIndex<InputImageType> it(seg, seg->GetRequestedRegion());
		for (;!it.IsAtEnd();++it)
		{
			InputImagePixelType label=it.Get();
			labelset.insert(label);
		}

		//Compute the offset table for that atlas
		ComputeOffsetTable(m_AtlasSegs[i],m_PatchRadius,offPatchSeg+i,nPatch);
		ComputeOffsetTable(m_Atlases[i],m_PatchRadius,offPatchAtlas+i,nPatch);
		ComputeOffsetTable(m_Atlases[i],m_SearchRadius,offSearchAtlas+i,nSearch,&manhattan);
	}

	//Initialize the posterior maps
	m_PosteriorMap.clear();

	//Allocate posterior images for the different labels
	for(typename std::set<InputImagePixelType>::iterator sit=labelset.begin();sit!=labelset.end();++sit)
	{
		m_PosteriorMap[*sit]=PosteriorImage::New();
		m_PosteriorMap[*sit]->SetLargestPossibleRegion(m_Target->GetLargestPossibleRegion());
		m_PosteriorMap[*sit]->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());
		m_PosteriorMap[*sit]->SetBufferedRegion(this->GetOutput()->GetBufferedRegion());
		m_PosteriorMap[*sit]->Allocate();
		m_PosteriorMap[*sit]->FillBuffer(0.0f);
	}

	//a image for sum of weights of posterior images ,used for normalized the weight in posterior images
	PosteriorImagePtr countermap=PosteriorImage::New();
	countermap->SetLargestPossibleRegion(m_Target->GetLargestPossibleRegion());
	countermap->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());
	countermap->SetBufferedRegion(this->GetOutput()->GetBufferedRegion());
	countermap->Allocate();
	countermap->FillBuffer(0.0f);

	//Create array for storing the normalized patch
	InputImagePixelType *xNormTargetPatch=new InputImagePixelType[nPatch];

	//most similar patch in each atlas
	InputImagePixelType **xNormAtlasesPatch=new InputImagePixelType*[n];
	for(int i=0;i<n;i++)
		xNormAtlasesPatch[i]=new InputImagePixelType[nPatch];

	//similar patch label
	const InputImagePixelType **patchSeg = new const InputImagePixelType*[n];

	int iter=0;
	//Iterate over voxels in the output region
	typedef itk::ImageRegionIteratorWithIndex<TOutputImage> OutIter;
	for(OutIter it(this->GetOutput(),this->GetOutput()->GetBufferedRegion());!it.IsAtEnd();++it)
	{
		//Point the target iterator to the output location
		itTarget.SetLocation(it.GetIndex());

		// If all atlases have no label(NULL?) at this location, jump out this loop
		int flag=0;
		for(int i=0;i<n;i++)
		{
			const InputImageType *seg=m_AtlasSegs[i];
			if (seg->GetPixel(it.GetIndex()))
				flag++;
		}
		if(flag==0)
			continue;

		//Compute stats for the target patch
		InputImagePixelType mu,sigma;
		InputImagePixelType *pTargetCurrent=m_Target->GetBufferPointer()+m_Target->ComputeOffset(it.GetIndex());
		PatchStats(pTargetCurrent,nPatch,offPatchTarget,mu,sigma);
		//normalize the patch
		for (unsigned int i=0;i<nPatch;i++)
			xNormTargetPatch[i]=(*(pTargetCurrent+offPatchTarget[i])-mu)/sigma;
		//In each atlas, search for a patch that matches our patch
		for(int i=0;i<n;i++)
		{
      const InputImageType *seg = m_AtlasSegs[i];
			int *offPatch=offPatchAtlas[i],
					*offSearch=offSearchAtlas[i];
			const InputImageType *atlas=m_Atlases[i];

			//Get the requested region for the tatlas
			RegionType rr=atlas->GetRequestedRegion();

			//Define the search region
			RegionType rSearch;
			for (int j=0;j<InputImageDimension;j++)
			{
			//requested region became small
				rr.SetIndex(j,rr.GetIndex(j)+m_SearchRadius[j]);
				rr.SetSize(j,rr.GetSize(j)-2*m_SearchRadius[j]);

				rSearch.SetIndex(j,it.GetIndex()[j]-m_SearchRadius[j]);
				rSearch.SetSize(j,2*m_SearchRadius[j]+1);
			}
			rSearch.Crop(rr);
			
			//Search over neighborhood
			double bestMatch=1e100;
			const InputImagePixelType *bestMatchPtr;

			InputImagePixelType bestMatchSum,bestMatchSSQ,bestMatchMean,bestMatchVar,bestMatchStd;

			const InputImagePixelType *pAtlasCurrent=atlas->GetBufferPointer()+atlas->ComputeOffset(it.GetIndex());
			
			int bestK=0;
			for(unsigned int k=0;k<nSearch;k++)
			{
				const InputImagePixelType *pSearchCenter=pAtlasCurrent+offSearch[k];
				InputImagePixelType matchSum=0,matchSSQ=0;
				double match=this->PatchSimilarity(pSearchCenter,xNormTargetPatch, nPatch,offPatch,matchSum,matchSSQ);
				if(match<bestMatch)
				{
					bestMatch=match;
					bestK=k;
					bestMatchSum=matchSum;
					bestMatchSSQ=matchSSQ;
					bestMatchPtr=pSearchCenter;
				}
			}
			//Update the manhattan distance histogram
			searchHisto[manhattan[bestK]]++;
			//Normalized the best patch
			bestMatchMean=bestMatchSum/nPatch;
			bestMatchVar=(bestMatchSSQ-nPatch*bestMatchMean*bestMatchMean)/(nPatch-1);
			if (bestMatchVar<1.0e-12)
				bestMatchVar=1.0e-12;
			bestMatchStd=sqrt(bestMatchVar);
			for(unsigned int m=0;m<nPatch;m++)
			{
				xNormAtlasesPatch[i][m]=(*(bestMatchPtr+offPatch[m])-bestMatchMean)/bestMatchStd;
			}
			
			//Store the best found neighborhood
			patchSeg[i]=(bestMatchPtr-m_Atlases[i]->GetBufferPointer()) +seg->GetBufferPointer();
		}
		//Compute L2 distance
		typedef vnl_matrix<double> MatrixType;
		MatrixType Dis(n+1,n+1);

		for (int j=0;j<n;j++)
		{
			for (int k=0;k<=j;k++)
			{
				InputImagePixelType dval=0.0;
				for (unsigned int m=0;m<nPatch;m++)
				{
					dval += pow(xNormAtlasesPatch[j][m]-xNormAtlasesPatch[k][m],2);
				}
				dval=sqrt(dval);
				Dis(j,k)=Dis(k,j)=dval;
			}
		}
		for (int j=0;j<n;j++)
		{
			InputImagePixelType tval=0.0;
			for (unsigned int m=0;m<nPatch;m++)
			{
				tval += pow(xNormAtlasesPatch[j][m]-xNormTargetPatch[m],2);
			}
			tval=sqrt(tval);

			Dis(j,n)=Dis(n,j)=tval;
		}
		Dis.fill_diagonal(0.0);
		VectorType R(m_dims);
		MatrixType Ycoords=run_isomap(Dis,m_K,m_dims,R);

		//distance in low dimension 
		double beta=2;
		VectorType lowDis(n),W(n),ones(n,1.0);
		for (int j=0;j<n;j++)
		{
			InputImagePixelType tval=0.0;
			for (unsigned int m=0;m<m_dims;m++)
				tval+=pow(Ycoords[m][j]-Ycoords[m][n],2);
			//tval=sqrt(tval);
			W(j)=pow(tval,-float(beta));
		}
		//normalized the weights
		W*=1.0/dot_product(W,ones);
		//Perform voting using averaging scheme.
	for(unsigned int ni = 0; ni < nPatch; ni++)
      {
      IndexType idx = itTarget.GetIndex(ni);
      if(this->GetOutput()->GetRequestedRegion().IsInside(idx))
        {
        for(int i = 0; i < n; i++)
          {
          // The segmentation at the corresponding patch location in atlas i
						if(W[i]>0.001){
          InputImagePixelType label = *(patchSeg[i] + offPatchSeg[i][ni]);

          // Add that weight the posterior map for voxel at idx
          m_PosteriorMap[label]->SetPixel(idx, m_PosteriorMap[label]->GetPixel(idx) + W[i]);
          countermap->SetPixel(idx, countermap->GetPixel(idx) + W[i]);
						}
          }
        }
      }

		if(++iter % 1000 ==0)
			cout<<"."<<flush;
	}
	
	cout<<endl<<"Search Manhattan Distance Histogram "<<endl;
	for (size_t i=0;i<searchHisto.size() &&searchHisto[i]>0;i++)
		cout<<"    "<<i <<"\t"<<searchHisto[i]<<endl;

	cout<<endl<<"VOTING "<<endl;

	//Perform voting at each voxel
	for(OutIter it(this->GetOutput(),this->GetOutput()->GetBufferedRegion());!it.IsAtEnd();++it)
	{
		double wmax=0;
		InputImagePixelType winner =0;

		for(typename std::set<InputImagePixelType>::iterator sit=labelset.begin();
				sit!=labelset.end();++sit)
		{
			double posterior=m_PosteriorMap[*sit]->GetPixel(it.GetIndex());

			//Vote!
			if (wmax<posterior)
			{
				wmax=posterior;
				winner=*sit;
			}
		}
		it.Set(winner);
	}
	cout<<endl<<"VOTING finished"<<endl;

	// posterior maps
	if(!m_RetainPosteriorMaps)
		m_PosteriorMap.clear();
	else
	{
		for (OutIter it(this->GetOutput(),this->GetOutput()->GetBufferedRegion());!it.IsAtEnd();++it)
		{
			IndexType idx=it.GetIndex();
			for (typename std::set<InputImagePixelType>::iterator sit=labelset.begin();
					sit!=labelset.end();++sit)
			{
				if(countermap->GetPixel(idx)>0)
					m_PosteriorMap[*sit]->SetPixel(idx,m_PosteriorMap[*sit]->GetPixel(idx)/countermap->GetPixel(idx));
			}
		}
	}

}

	template <class TInputImage,class TOutputImage>
		void ManifoldlearningLocalWeightImageFilter<TInputImage,TOutputImage>::PatchStats(const InputImagePixelType *p,
				size_t n, int *offsets,
				InputImagePixelType &mean,
				InputImagePixelType &std)
		{
			InputImagePixelType sum=0,ssq=0;
			for(unsigned int i=0;i<n;i++)
			{
				InputImagePixelType v=*(p+offsets[i]);
				sum+=v;
				ssq+=v*v;
			}
			mean =sum/n;
			std=(ssq-n*mean*mean)/(n-1);
			if (std<1e-6)
				std=1e-6;
			std=sqrt(std);
		}

	//a simplify function of sum of squared difference between two patches
	template <class TInputImage,class TOutputImage>
		double ManifoldlearningLocalWeightImageFilter<TInputImage,TOutputImage>::PatchSimilarity(
				const InputImagePixelType *psearch,
				const InputImagePixelType *normtrg,
				size_t n,
				int *offsets,//offset of psearch
				InputImagePixelType &sum_psearch,
				InputImagePixelType &ssq_psearch)
		{
			InputImagePixelType sum_uv=0;
			for (unsigned int i=0; i<n;i++)
			{
				InputImagePixelType u=*(psearch+offsets[i]);
				InputImagePixelType v=normtrg[i];
				sum_psearch +=u;
				ssq_psearch +=u*u;
				sum_uv +=u*v;
			}

			InputImagePixelType var_u_unnorm =ssq_psearch-sum_psearch*sum_psearch/n;
			if (var_u_unnorm<1.0e-6)
				var_u_unnorm=1.0e-6;

			if (sum_uv>0)
				return -(sum_uv*sum_uv)/var_u_unnorm;
			else
				return (sum_uv*sum_uv)/var_u_unnorm;
		}
