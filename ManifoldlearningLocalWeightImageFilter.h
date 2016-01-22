// C/C++ File
// AUTHOR:   Xinwei Li -lixinwei9046@gmail.com 
// FILE:     ManifoldlearningLocalWeightImageFilter.h
// ROLE:     TODO (some explanation)
// CREATED:  2014-05-03 15:53:08
// MODIFIED: 2014-05-05 19:24:00

#ifndef __ManifoldlearningLocalWeightImageFilter_h_
#define __ManifoldlearningLocalWeightImageFilter_h_
#include "itkImageToImageFilter.h"
#include "itkConstNeighborhoodIterator.h"

template <class TInputImage, class TOutputImage>
class ManifoldlearningLocalWeightImageFilter : public itk::ImageToImageFilter <TInputImage, TOutputImage>
{
	public:
		/** Standard class typedefs. */
		typedef ManifoldlearningLocalWeightImageFilter Self;
		typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
		typedef itk::SmartPointer<Self> Pointer;
		typedef itk::SmartPointer<const Self> ConstPointer;

		/** Run-time type information (and related methods). */
		itkTypeMacro(ManifoldlearningLocalWeightImageFilter,ImageToImageFilter);

		itkNewMacro(Self);

		/** Superclass typedefs. */
		typedef typename Superclass::OutputImageRegionType OutputImageRegionType;
		typedef typename Superclass::OutputImagePixelType OutputImagePixelType;

		/** Some convenient typedefs. */
		typedef TInputImage InputImageType;
		typedef typename InputImageType::Pointer InputImagePointer;
		typedef typename InputImageType::ConstPointer InputImageConstPointer;
		typedef typename InputImageType::PixelType InputImagePixelType;
		typedef typename InputImageType::RegionType RegionType;
		typedef typename InputImageType::SizeType SizeType;
		typedef typename InputImageType::IndexType IndexType;

		typedef std::vector<int> LabelSetType;
		typedef std::vector<InputImagePointer> InputImageList;

		/** ImageDimension constants */
		itkStaticConstMacro(InputImageDimension,unsigned int,TInputImage::ImageDimension);
		itkStaticConstMacro(OutputImageDimension,unsigned int,TOutputImage::ImageDimension);

		/** Set target image */
		void SetTargetImage(InputImageType *image)
		{m_Target=image;UpdateInputs();}

		/** Add an atlas */
		void AddAtlas(InputImageType *grey, InputImageType *seg)
		{
			m_Atlases.push_back(grey);
			m_AtlasSegs.push_back(seg);
			UpdateInputs();
		}
		
		/**Set the parameters */
		itkSetMacro(SearchRadius, SizeType);
		itkGetMacro(SearchRadius, SizeType);

		itkSetMacro(PatchRadius,SizeType);
		itkGetMacro(PatchRadius,SizeType);

		itkSetMacro(K,int);
		itkGetMacro(K,int);

		itkSetMacro(dims,int);
		itkGetMacro(dims,int);

		/** Set the requested region */
		void GenerateInputRequestedRegion();

		/** Set posterior maps */
		itkSetMacro(RetainPosteriorMaps,bool);
		itkGetMacro(RetainPosteriorMaps,bool);

		typedef itk::Image<float,InputImageDimension> PosteriorImage;
		typedef typename PosteriorImage::Pointer PosteriorImagePtr;
		typedef typename std::map<InputImagePixelType,PosteriorImagePtr> PosteriorMap;
		/** Get the posterior maps if they have been retained */
		const PosteriorMap &GetPosteriorMaps()
		{	return m_PosteriorMap;}

		void GenerateData();

	protected:
		ManifoldlearningLocalWeightImageFilter(){
			m_K=7;
			m_dims=2;
			m_RetainPosteriorMaps=false;
		}
		~ManifoldlearningLocalWeightImageFilter(){};

	private:
		typedef itk::ConstNeighborhoodIterator<InputImageType> NIter;
		void ComputeOffsetTable(const InputImageType *Image, const SizeType &radius, int **offset, size_t &nPatch, int **manhattan=NULL);

		void UpdateInputs();

		void PatchStats(const InputImagePixelType *p, size_t n, int *offsets,InputImagePixelType &mean,InputImagePixelType &std);

		double PatchSimilarity(const InputImagePixelType *psearch,const InputImagePixelType *pnormtrg,size_t n, int *offsets, InputImagePixelType &psearchSum, InputImagePixelType &psearchSSQ);

		SizeType m_SearchRadius,m_PatchRadius;
		int m_K,m_dims;
		//posterior maps
		PosteriorMap m_PosteriorMap;

		bool m_RetainPosteriorMaps;
		
		//inputs
		InputImagePointer m_Target;
		InputImageList m_Atlases;
		InputImageList m_AtlasSegs;

};

#endif
