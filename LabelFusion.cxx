// C/C++ File
// AUTHOR:   Xinwei Li -lixinwei9046@gmail.com 
// FILE:     LabelFusion.cxx
// ROLE:     TODO (some explanation)
// CREATED:  2014-05-05 10:25:46
// MODIFIED: 2014-05-09 15:23:40

#include "ManifoldlearningLocalWeightImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include <iostream>
#include "ManifoldlearningLocalWeightImageFilter.txx"
#include <time.h>
using namespace std;

int usage()
{
	cout << "Manifold Learning Label Fusion: " << endl;
	cout << "usage: " << endl;
	cout << " mllf dim [options] output_image" << endl;
	cout << endl;
	cout << "required options:" << endl;
	cout << "  dim                             Image dimension (2 or 3)" << endl;
	cout << "  -g atlas1.nii ...atlasN.nii ... "<<endl;
	cout << "                                  Warped atlas images" << endl;
	cout << "  -tg target.nii "<<endl;
	cout << "                                  Target image(s)" << endl;
	cout << "  -l label1.nii ... labelN.nii    Warped atlas segmentation" << endl;
	cout << "  -m [parameters]                 Isomap parameters K, dims in brackets"<<endl; 
	cout << "other options: " << endl;
	cout << "  -rp radius                      Patch radius for similarity measures, scalar or vector (AxBxC) " << endl;
	cout << "                                  Default: 2x2x2" << endl;
	cout << "  -rs radius                      Local search radius." << endl;
	cout << "                                  Default: 3x3x3" << endl;
	cout << "  -p filenamePattern              Save the posterior maps (probability that each voxel belongs to each label) as images."<<endl;
	cout << "                                  The number of images saved equals the number of labels."<<endl;
	cout << "                                  The filename pattern must be in C printf format, e.g. posterior%04d.nii.gz" << endl;

	return -1;
}
template <unsigned int VDim>
struct LFParam
{
	vector<string> fnAtlas;
	vector<string> fnLabel;
	string fnTargetImage;
	string fnOutput;
	string fnPosterior;

	int K, dims;
	itk::Size<VDim> r_patch, r_search;

	LFParam()
	{
		K=7;
		dims=2;
		r_patch.Fill(2);
		r_search.Fill(3);
	}

	void Print(std::ostream &oss)
	{
		oss<<"Target image: "<<fnTargetImage<<endl;
		oss<<"Output image: "<<fnOutput<<endl;
		oss<<"Atlas images: "<<fnAtlas.size()<<endl;
		for (size_t i=0;i<fnAtlas.size();i++)
		{
			oss<<i<< " : "<<fnAtlas[i]<<" | "<<fnLabel[i]<<endl;
		}
		oss << "Method:   Isomap "<<endl;
		oss <<"K = "<<K<<endl;
		oss <<"dims= "<<dims<<endl;
		oss <<"Search Radius: "<<r_search<<endl;
		oss <<"Patch Radius: "<<r_patch<<endl;
		if(fnPosterior.size())
			oss<<"Posterior Filename Pattern: "<<fnPosterior<<endl;
	}
};

	template <unsigned int VDim>
void ExpandRegion(itk::ImageRegion<VDim> &r, bool &isinit, const itk::Index<VDim> &idx)
{
	if(!isinit)
	{
		for(size_t d = 0; d < VDim; d++)
		{
			r.SetIndex(d, idx[d]);
			r.SetSize(d,1);
		}
		isinit = true;
	}
	else
	{
		for(size_t d = 0; d < VDim; d++)
		{
			int x = r.GetIndex(d), s = r.GetSize(d);
			if(idx[d] < x)
			{
				r.SetSize(d, s + x - idx[d]);
				r.SetIndex(d, idx[d]);
			}
			else if(idx[d] >= x + s)
			{
				r.SetSize(d, 1 + idx[d] - x);
			} 
		}
	}
}


template <unsigned int VDim>
	bool
parse_vector(char *text, itk::Size<VDim> &s)
{
	char *t=strtok(text,"x");
	size_t i=0;
	while (t && i <VDim)
	{
		s[i++]=atoi(t);
		t=strtok(NULL,"x");
	}

	if(i==VDim)
		return true;
	else if (i==1)
	{
		s.Fill(s[0]);
		return true;
	}
	return false;
}

	template <unsigned int VDim>
int lfapp(int argc,char *argv[])
{
	//Parameter vector
	LFParam<VDim> p;

	//Read the parameters from command line
	p.fnOutput=argv[argc-1];

	int argend=argc-2;

	for(int j=2;j<argend;j++)
	{
		string arg=argv[j];
		if (arg=="-g")
		{//read atlases images
			while(argv[j+1][0]!='-' && j<argend)
				p.fnAtlas.push_back(argv[++j]);
		}
		else if (arg=="-tg")
		{
			//Read target image
			p.fnTargetImage=(argv[++j]);
		}
		else if(arg=="-l")
		{
			//Read label images
			while (argv[j+1][0]!='-' && j<argend)
				p.fnLabel.push_back(argv[++j]);
		}
		else if(arg=="-p")
		{
			p.fnPosterior=argv[++j];
		}
		else if(arg=="-rs" &&j<argend)
		{
			//Search radius
			if(!parse_vector<VDim>(argv[++j],p.r_search))
			{
				cerr << "Bad vector spec " << argv[j] <<endl;
				return -1;
			}
		}
		else if(arg=="-rp" &&j<argend)
		{
			//Patch radius
			if(!parse_vector<VDim>(argv[++j],p.r_patch))
			{
				cerr << "Bad vector spec " << argv[j] <<endl;
				return -1;
			}
		}
		else if(arg=="-m" && j<argend)
		{
			int K,dims;
			char *parm =argv[++j];
			switch(sscanf(parm,"[%d,%d]",&K,&dims))
			{
				case 2:
					p.K=K; p.dims=dims;break;
				case 1:
					p.K=K;break;
			}
		}
		else
		{
			cerr << "Unkown option "<<arg<<endl;
			return -1;
		}
	}

	//Check the posterior filename pattern
	if(p.fnPosterior.size())
	{
		char buffer[4096];
		sprintf(buffer,p.fnPosterior.c_str(),100);
		if (strcmp(buffer,p.fnPosterior.c_str())==0)
		{
			cerr<<"Invalid filename pattern "<<p.fnPosterior<<endl;
			return -1;
		}
	}

	// Print parameters
	cout<< "PARAMETERS: "<<endl;
	p.Print(cout);

	// Configuire the filter
	typedef itk::Image<float,VDim> ImageType;
	typedef ManifoldlearningLocalWeightImageFilter<ImageType,ImageType> VoterType;
	typename VoterType::Pointer voter = VoterType::New();

	//Set inputs
	typedef itk::ImageFileReader<ImageType> ReaderType;
	typedef itk::ImageFileWriter<ImageType> WriterType;

	//set Target
	typename ReaderType::Pointer rTarget=ReaderType::New();
	rTarget->SetFileName(p.fnTargetImage.c_str());
	rTarget->Update();
	typename ImageType::Pointer target=rTarget->GetOutput();
	voter->SetTargetImage(target);

	//Compute the output region-- rMask
	itk::ImageRegion<VDim> rMask;
	bool isMaskInit =false;

	for(size_t i=0;i<p.fnLabel.size();i++)
	{
		typename ReaderType::Pointer ra,rl;

		ra=ReaderType::New();
		ra->SetFileName(p.fnAtlas[i].c_str());
		ra->Update();

		rl=ReaderType::New();
		rl->SetFileName(p.fnLabel[i].c_str());
		rl->Update();

		voter->AddAtlas(ra->GetOutput(),rl->GetOutput());
		for(itk::ImageRegionIteratorWithIndex<ImageType> it(rl->GetOutput(),rl->GetOutput()->GetRequestedRegion());!it.IsAtEnd();++it)
		{
			if (1)
			{
				ExpandRegion<VDim>(rMask, isMaskInit, it.GetIndex());
			}
		}
	}

	//Make sure the region is inside bounds
	itk::ImageRegion<VDim> rOut=target->GetLargestPossibleRegion();
	for(int d=0;d<VDim;d++)
	{
		rOut.SetIndex(d,p.r_patch[d]+p.r_search[d]+rOut.GetIndex(d));
		rOut.SetSize(d,rOut.GetSize(d)-2*(p.r_patch[d]+p.r_search[d]));
	}
	rMask.Crop(rOut);

	voter->SetPatchRadius(p.r_patch);
	voter->SetSearchRadius(p.r_search);
	voter->SetK(p.K);
	voter->Setdims(p.dims);

	if(p.fnPosterior.size())
		voter->SetRetainPosteriorMaps(true);

	cout<<"Output Requested Region: "<<rMask.GetIndex()<<", "<<rMask.GetSize()<<" ("<<rMask.GetNumberOfPixels()<<" pixels)"<<std::endl;

	voter->GetOutput()->SetRequestedRegion(rMask);

	voter->GenerateData();

	cout<<"saving segmentation ..."<<endl;

	//Converter to an output image
	target->FillBuffer(0.0);
	for(itk::ImageRegionIteratorWithIndex<ImageType> it(voter->GetOutput(),rMask);!it.IsAtEnd();++it)
	{
		target->SetPixel(it.GetIndex(),it.Get());
	}

	//Creat writer
	typename WriterType::Pointer writer=WriterType::New();
	writer->SetInput(target);
	writer->SetFileName(p.fnOutput.c_str());
	writer->Update();

	//store the posterior maps
	if(p.fnPosterior.size())
	{
		cout<<"saving label posteriors ..."<<endl;
		const typename VoterType::PosteriorMap &pm =voter->GetPosteriorMaps();

		typename VoterType::PosteriorImagePtr pout=VoterType::PosteriorImage::New();
		pout->SetRegions(target->GetBufferedRegion());
		pout->CopyInformation(target);
		pout->Allocate();

		//Iterate over the labels
		typename VoterType::PosteriorMap::const_iterator it;
		for (it=pm.begin();it!=pm.end();it++)
		{
			//Get the filename
			char buffer[4096];
			sprintf(buffer, p.fnPosterior.c_str(),(int) it->first);

			//Initialize to zeros
			pout->FillBuffer(0.0);

			//Replace the portion affected by the filter
			for (itk::ImageRegionIteratorWithIndex<typename VoterType::PosteriorImage> qt(it->second,rMask);
					!qt.IsAtEnd();++qt)
			{
				pout->SetPixel(qt.GetIndex(),qt.Get());
			}

			//Create writer
			typename WriterType::Pointer writer=WriterType::New();
			writer->SetInput(pout);
			writer->SetFileName(buffer);
			writer->Update();
		}
	}
}


int main(int argc,char *argv[])
{
	clock_t start_time=clock();
	if (argc<4)
		return usage();
	int dim = atoi(argv[1]);

	if(dim==2)
		lfapp<2>(argc,argv);
	else if(dim==3)
		lfapp<3>(argc,argv);
	else
	{
		cerr<<"Dimension "<<argv[1]<<" is not supported"<<endl;
		return -1;
	}
	clock_t end_time=clock();
	cout<<"Running time is: "<<static_cast<double>(end_time-start_time)<<"minutes"<<endl;
		return 0;

}	

