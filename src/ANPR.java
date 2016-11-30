
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.CvANN_MLP;
import org.opencv.ml.CvSVM;
import org.opencv.ml.CvSVMParams;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.utils.Converters;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.NodeList;
import org.w3c.dom.Node;
import org.w3c.dom.Element;
import org.xml.sax.SAXException;

import java.io.File;

//
// Detects faces in an image, draws boxes around them, and writes the results
// to "faceDetection.png".
//
class DetectFaceDemoDTC {
	
  private static Mat cropLicensePlates(Mat image, String outputDir, String fileName){
	  	Mat image_gray = new Mat(image.width(), image.height(), CvType.CV_8U);
	  	Mat output = null;
	    //image.convertTo(image, Imgproc.colo);
	    System.out.println(image.channels());
	    Imgproc.cvtColor(image, image_gray, Imgproc.COLOR_BGR2GRAY);
 
	    Imgproc.blur(image_gray, image_gray, new Size(5,5));
	    
	    String filename = fileName + "_GRAY.png";
	    //System.out.println(String.format("Writing %s", filename));
	    Highgui.imwrite(filename, image_gray);
	    
	    Mat image_sobel = new Mat(image.width(), image.height(), CvType.CV_8U);
	    Imgproc.Sobel(image_gray, image_sobel, CvType.CV_8U, 1, 0, 3, 1, 0);
	    
	    filename = fileName + "_SOBEL.png";
	    //System.out.println(String.format("Writing %s", filename));
	    Highgui.imwrite(filename, image_sobel);
	    
	    Mat img_thres = new Mat(image.width(), image.height(), CvType.CV_8U);
	    Imgproc.threshold(image_sobel, img_thres, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);

	    filename = fileName + "_Threshold.png";
	    //System.out.println(String.format("Writing %s", filename));
	    Highgui.imwrite(filename, img_thres);
	    
	    Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(17,8)); 
	    Imgproc.morphologyEx(img_thres, img_thres, Imgproc.MORPH_CLOSE, element);
	    
	    filename = fileName + "_MorphEX.png";
	    System.out.println(String.format("Writing faceDetectionMorphEX %s", fileName));
	    Highgui.imwrite(filename+".png", img_thres);
	    
	    List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		Imgproc.findContours(img_thres, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
		
		List<RotatedRect> rects = new ArrayList<RotatedRect>();
		for(MatOfPoint mop: contours)
		{
			RotatedRect mr = Imgproc.minAreaRect(new MatOfPoint2f(mop.toArray()));
			if(!verifySizes(mr))
			{
				contours.remove(mr);
			}
			else
			{
				rects.add(mr);
			}
			
			//System.out.println(verifySizes(mr));		
		}
		
		Random randomGen = new Random();
		//System.out.println("rect sixe" + rects.size());
		Mat result = image.clone();
		int i = 0;
		for(RotatedRect recto: rects)
		{
			Core.circle(result, recto.center, 3, new Scalar(0,255,0), -1);
			float minSize = (float) ((recto.size.width < recto.size.height) ? recto.size.width : recto.size.height);
			//System.out.println("min sixe " + minSize);
			minSize -= minSize * 0.5;
			//System.out.println("min sixe " + minSize);
			Mat mask = new Mat(image.rows() + 2, image.cols() + 2, CvType.CV_8UC1);
			mask.setTo(Scalar.all(0));
			//System.out.println("mask size " + mask.size());
			
			int loDiff = 30;
			int upDiff = 30;
			
			int connectivity = 4;
			int newMaskVal = 255;
			int NumSeeds = 10;
			Rect ccomp = new Rect();
			int flags = connectivity + (newMaskVal << 8) +
						Imgproc.FLOODFILL_FIXED_RANGE + Imgproc.FLOODFILL_MASK_ONLY;
			
			for(int j = 0; j < NumSeeds; ++j)
			{
				Point seed = new Point();
				seed.x = recto.center.x + randomGen.nextDouble()*1000%(int)minSize - minSize/2;
				seed.y = recto.center.y + randomGen.nextDouble()*1000%(int)minSize - minSize/2;
			
				Core.circle(result, seed, 1, new Scalar(0,255,255), -1);			
				int area = Imgproc.floodFill(image, mask, seed, new Scalar(255,0,0), ccomp, new Scalar(loDiff, loDiff, loDiff), new Scalar(upDiff, upDiff, upDiff), flags);	
			}
			++i;
			
		    filename = "faceDetectionMask" + i + " " +  ".png";
		    //System.out.println(String.format("Writing %s", filename));
		    //Highgui.imwrite(filename, mask);
			
			List<Point> pointsInterest = new ArrayList<Point>();
	        int rows = mask.rows();
	        int cols = mask.cols();
	        
	        for(int row = 0; row < rows; ++row)
	        {
		        for(int col = 0; col < cols; ++col)
		        {
		        	if(mask.get(row, col)[0] == 255)
		        	{
		        		pointsInterest.add(new Point(col, row));
		        	}
		        	
		        }
	        }
	        
	        MatOfPoint2f mop = new MatOfPoint2f();
	        mop.fromList(pointsInterest);
	        RotatedRect mr = Imgproc.minAreaRect(mop);
	        
	        if(verifySizes(mr))
	        {
	            Point[] rect_points = new Point[4]; 
	            mr.points(rect_points);
	            for( int j = 0; j < 4; j++ )
	            {
	                Core.line(result, rect_points[j], rect_points[(j+1) % 4], new Scalar(0,0,255), 1, 8, 0);
	            }
	            
	    	    filename = fileName + "_line_" + i + " " +  ".png";
	    	    //System.out.println(String.format("Writing %s", filename));
	    	    Highgui.imwrite(filename, result);
	            
	        	double r = (double)mr.size.width / (double)mr.size.height;;
	        	double angle = mr.angle;
	        	if(r < 1)
	        	{
	        		angle += 90;
	        	}
	        	
	        	Mat rotmat = Imgproc.getRotationMatrix2D(mr.center, angle, 1);	        		
	        	Mat imgRot = new Mat();
	        	Imgproc.warpAffine(image, imgRot, rotmat, image.size(), Imgproc.INTER_CUBIC);
	        		
	        	Size size = mr.size;
	        	if(r < 1)
	        	{
	        	    double hw = size.width;
	        	    size.width = size.height;
	        	    size.height = hw;
	        	}
	        	    
	        	Mat newMat = new Mat();
	        	Imgproc.getRectSubPix(imgRot, size, mr.center, newMat);
	        	    
	        	filename = fileName + "_ROTA_" + i + " " +  ".png";
	        	//System.out.println(String.format("Writing %s", filename));
	        	Highgui.imwrite(filename, newMat);
	        
	        
	        Mat resultResized = new Mat();
	        resultResized.create(33,144, CvType.CV_8UC3);
	        Imgproc.resize(newMat, resultResized, resultResized.size(), 0, 0, Imgproc.INTER_CUBIC);
	        //Equalize croped image
	        Mat grayResult = new Mat();
	        Imgproc.cvtColor(resultResized, grayResult, Imgproc.COLOR_BGR2GRAY);
	        Imgproc.blur(grayResult, grayResult, new Size(3,3));
	        grayResult = histeq(grayResult);
	        //grayResult = Imgproc.calc.get(grayResult);
	        
	    	filename = outputDir + fileName +  ".png";
	    	System.out.println(String.format("Writing %s", filename));
	    	Highgui.imwrite(filename, grayResult);
	        
	        //output.push_back(Plate(grayResult,minRect.boundingRect())); 
	    	output = grayResult;
	  }
  }
		return output;
  }
  
  public void run() throws ParserConfigurationException, SAXException, IOException, InterruptedException {
    System.out.println("\nRunning Detect License Demo");

    
    // Create a face detector from the cascade file in the resources
    // directory.
    String trainingDir = "D:\\Work Related\\Source_Code\\ANPR\\train\\";
    String trainingResDir = "D:\\Work Related\\Source_Code\\ANPR\\trainResult\\";
    File folder = new File(trainingDir);
    
    //Thread.sleep(10000);
    
    int index = 1;
    for (final File fileEntry : folder.listFiles()) {
        if (!fileEntry.isDirectory()) {
        
            System.out.println(fileEntry.getName());
            Mat image = Highgui.imread(trainingDir + fileEntry.getName());   
            //Highgui.imwrite(trainingResDir+index+"_"+index+".jpg", image);
            //System.out.println(fileEntry.getName() + " " + image.channels());
            Mat result = cropLicensePlates(image, trainingResDir, ""+index++);
        }
    }

    //FileStorage fs = 
    folder = new File(trainingResDir + "plates\\");
    File[] listOfFiles = folder.listFiles();

    Mat trainImage = new Mat(new Size(144*33, 1), 0);
    Mat trainImages = new Mat(new Size(0, 0), 0);
    List<Integer> trainClasses = new ArrayList<Integer>();
    for (File file : listOfFiles) {
        if (file.isFile()) {
            Mat img = Highgui.imread(folder + "\\" + file.getName(), 0);
            //System.out.println("Mat String-" + folder + "\\" + file.getName());
            //System.out.println("Mat String-" + img.channels());
            //Highgui.imwrite(folder + "\\" + "_" + file.getName(), img);
            img = img.reshape(1, 1);
            img.copyTo(trainImage);
            trainImages.push_back(trainImage);
            trainClasses.add(1);
        }
    }
    
    folder = new File(trainingResDir + "noise\\");
    listOfFiles = folder.listFiles();

    for (File file : listOfFiles) {
        if (file.isFile()) {
            Mat img = Highgui.imread(folder + "\\" + file.getName(), 0);
            img = img.reshape(1, 1);
            img.copyTo(trainImage);
            trainImages.push_back(trainImage);
            trainClasses.add(0);
        }
    }
      
    CvSVMParams params = new CvSVMParams();
    params.set_kernel_type(CvSVM.LINEAR);
    params.set_svm_type(CvSVM.C_SVC);
    params.set_degree(0);
    params.set_gamma(1);
    params.set_coef0(0);
    params.set_C(1);
    params.set_nu(0);
    params.set_p(0);
    params.set_term_crit(new TermCriteria(TermCriteria.MAX_ITER, 1000, 0.01));
    
    //trainingData = trainingData.reshape(1,trainingData.rows);
    Mat trainingData = new Mat();
	trainImages.convertTo(trainingData , CvType.CV_32FC1);
    Mat trainCl = new Mat(trainClasses.size(), 1, CvType.CV_32FC1);
    int rowIndex = 0;
    for(Integer label: trainClasses){
    	double[] val = {label};
    	trainCl.put(rowIndex++, 0, val);
    }

    CvSVM svm = new CvSVM();
    System.out.println("train img data-" + trainImages.rows() +"," +  trainImages.cols() + " classes-" + trainCl.rows());
    System.out.println("train data-" + trainingData.rows() +"," +  trainingData.cols() + " classes-" + trainCl.rows());
    svm.train(trainingData, trainCl, new Mat(), new Mat(), params);
    
    String testDir = "D:\\Work Related\\Source_Code\\ANPR\\test\\";
    String validationResDir = "D:\\Work Related\\Source_Code\\ANPR\\validation\\";
    folder = new File(testDir);
    
    index = 1;
    for (final File fileEntry : folder.listFiles()) {
        if (!fileEntry.isDirectory()) {
        
            //System.out.println(fileEntry.getName());
            Mat image = Highgui.imread(testDir + fileEntry.getName());   
            //Highgui.imwrite(trainingResDir+index+"_"+index+".jpg", image);
            //System.out.println(fileEntry.getName() + " " + image.channels());
            Mat result = cropLicensePlates(image, validationResDir, ""+index++);
        }
    }
    

	TaFileStorage tf = new TaFileStorage();
	tf.open("D:\\Work Related\\Source_Code\\ANPR\\bin\\resources\\OCR.xml", TaFileStorage.READ);
	Mat tData = tf.readMat("TrainingDataF5");
	Mat cData = tf.readMat("classes");
	
	System.out.println("**********" + tData.size().width * tData.size().height);
	System.out.println("**********" + cData.size().width * cData.size().height + "***" + cData.rows() + "****" + cData.cols());
	
	Mat layers = new Mat(1,3, CvType.CV_32SC1);
    layers.put(0,  0, tData.cols());
    layers.put(0,  1,  3);
    layers.put(0,  2,  36);
    CvANN_MLP ann = new CvANN_MLP();
    ann.create(layers, CvANN_MLP.SIGMOID_SYM, 1, 1);

    Mat weights = new Mat( 1, tData.rows(), CvType.CV_32FC1, Scalar.all(1) );

    Mat trainCls = new Mat();
    trainCls.create( tData.rows(), 36, CvType.CV_32FC1 );
    for( int i = 0; i <  trainCls.rows(); i++ )
    {
        for( int k = 0; k < trainCls.cols(); k++ )
        {
            //If class of data i is same than a k class
            if( k == (int)cData.get(i,  0)[0] ){
            	trainCls.put(i, k, 1);
            	System.out.println("########" + (int)cData.get(i,  0)[0]);
            	
            }
            else
            	trainCls.put(i, k, 0);
        }
    }
    
    //Learn classifier
    ann.train( tData, trainCls, weights );
    //Mat TrainingData = new Mat( );
    //Mat Classes;


    //train(TrainingData, Classes, 10);
    
    folder = new File(validationResDir);
    listOfFiles = folder.listFiles();
    int newInd = 1;
    for (File file : listOfFiles) {
        if (file.isFile()) {
        	Mat img = Highgui.imread(validationResDir + file.getName(), 0 );
        	System.out.println("Predicting-" + validationResDir + file.getName());
        	Mat imgNew = img.reshape(1, 1);
        	//Mat tImage = new Mat(new Size(144*33, 1), 0);
        	//imgNew.copyTo(tImage);
        	Mat newImg = new Mat();
        	
        	imgNew.convertTo(newImg , CvType.CV_32FC1);
        	//System.out.println("tImage image size-" + tImage.rows() + "," + tImage.cols() + "," + img.cols());
        	///System.out.println("pred image size-" + newImg.rows() + "," + newImg.cols() + "," + img.cols());
        	//System.out.println(img.type() +"," + CvType.CV_32FC1 + "," + newImg.type());
        	int res = (int)svm.predict(newImg);
        	if(res == 1){
        		System.out.println("YES");
        		Mat imgThresh = img.clone();
        		Imgproc.threshold(img, imgThresh, 60, 255, Imgproc.THRESH_BINARY_INV);
        		Highgui.imwrite("D:\\Work Related\\Source_Code\\ANPR\\" +"_"+index++ +".jpg", imgThresh);
        		
        		Mat imgCont = new Mat();
        		imgThresh.copyTo(imgCont);
        		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
				Imgproc.findContours(imgCont, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
				
			    // Draw blue contours on a white image
			    Mat result = new Mat();
			    imgThresh.copyTo(result);
			    Imgproc.cvtColor(result, result, Imgproc.COLOR_GRAY2RGB);
			    Imgproc.drawContours(result, contours, -1,   new Scalar(255,0,0),   1); 

			    List<Mat> charSeg = new ArrayList<Mat>();
			    
				for(MatOfPoint mop: contours){
					
			        Rect mr= Imgproc.boundingRect(mop);
			        Point br = mr.br();
			        br.x = br.x - 1;
			        br.y = br.y - 1;
			        //Core.rectangle(result, br, mr.tl(), new Scalar(0,255,0));
			        //Crop image
			        Mat auxRoi = new Mat(imgThresh, mr);
			        
			        if(verifyCharSize(auxRoi)){
			            auxRoi = preprocessChar(auxRoi);
			            System.out.println("Character found");
			            charSeg.add(auxRoi);
			            //output.push_back(CharSegment(auxRoi, mr));
			            //Core.rectangle(result, br, mr.tl(), new Scalar(0,125,255));  
			            //Mat newM = result.submat(mr);
			             //newM);
			        }		
				}
				
				for(Mat ch: charSeg){
						Mat f = getFeatures(ch, 5);
						int resCh = -1;
					    Mat output = new Mat(1, 36, CvType.CV_32FC1);
					    ann.predict(f, output);
					    Point maxLoc;
					    double maxVal;
					    MinMaxLocResult mmR = Core.minMaxLoc(output);//, 0, &maxVal, 0, &maxLoc);
					    //We need know where in output is the max val, the x (cols) is the class.
					    char charac = strCharacters[(int) mmR.maxLoc.x];
					    System.out.println("identified char is-" + charac);
					    Highgui.imwrite("D:\\Work Related\\Source_Code\\ANPR\\chars\\" +"_"+ charac + "_" + newInd++ +".jpg", ch);
				}
				
				
				 Highgui.imwrite("D:\\Work Related\\Source_Code\\ANPR\\chars\\" +"_"+ newInd++ +".jpg", result);
        		
        	}
        }
    }
    
    //CvSVM svm = trainSVM();
    
  }
  
private static Mat getFeatures(Mat in, int sizeData) {
    //Histogram features
    Mat vhist = ProjectedHistogram(in, 1);//vertical
    Mat hhist = ProjectedHistogram(in, 0);//horizontal
    
    //Low data feature
    Mat lowData = new Mat();
    Imgproc.resize(in, lowData, new Size(sizeData, sizeData) );

    //Last 10 is the number of moments components
    int numCols = vhist.cols() + hhist.cols() + lowData.cols()*lowData.cols();
    System.out.println("num col-" + lowData.cols() + "-" + vhist.cols() + "-" + hhist.cols());
    
    Mat out = Mat.zeros(1, numCols, CvType.CV_32F);
    //Asign values to feature
    int j = 0;
    for(int i = 0; i < vhist.cols(); i++)
    {
    	//System.out.println("Feature-" + vhist.get(0, i));
        out.put(0, j, vhist.get(0, i));        
        j++;
    }
    
    for(int i = 0; i < hhist.cols(); i++)
    {
    	//System.out.println("Feature-" + vhist.get(0, i).length);
    	out.put(0, j, hhist.get(0, i));
        j++;
    }
    
    for(int x = 0; x < lowData.cols(); x++)
    {
        for(int y = 0; y < lowData.rows(); y++){
        	out.put(0, j, lowData.get(x, y));
            j++;
        }
    }

    return out;
}

private static Mat ProjectedHistogram(Mat img, int t) {
    int sz = (t != 0) ? img.rows(): img.cols();
    Mat mhist = Mat.zeros(1, sz, CvType.CV_32F);

    for(int j = 0; j < sz; j++){
        Mat data = (t != 0) ? img.row(j): img.col(j);
        mhist.put(0, j, Core.countNonZero(data));
    }

    //Normalize histogram
    double min, max;
    MinMaxLocResult minMaxRes = Core.minMaxLoc(mhist);
    min = minMaxRes.minVal;
    max = minMaxRes.maxVal;
    
    if(max > 0)
        mhist.convertTo(mhist, -1 , 1.0f/max, 0);

    return mhist;
}
private static char[] strCharacters = {'0','1','2','3','4','5','6','7','8','9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'};
private static int totalChars = 36;

private Mat preprocessChar(Mat in) {
    //Remap image
    int h = in.rows();
    int w = in.cols();
    Mat transformMat = Mat.eye(2, 3, CvType.CV_32F);
    
    int m = Math.max(w,h);
    
    transformMat.put(0,  2, m/2 - w/2);
    transformMat.put(1,  2, m/2 - h/2);

    Mat warpImage = new Mat(m, m, in.type());
    Imgproc.warpAffine(in, warpImage, transformMat, warpImage.size(), Imgproc.INTER_LINEAR, Imgproc.BORDER_CONSTANT, new Scalar(0) );

    Mat out = in.clone();
    int charSize = 20;
    Imgproc.resize(warpImage, out, new Size(charSize, charSize) ); 

    return out;
}

private boolean verifyCharSize(Mat m) {
	double aspect = 45.0d/77.0d;
	double charAspect = ((double)m.cols())/m.rows();
	double error = 0.50;
	double minH = 15;
	double maxH = 28;
	
	double minAR = aspect - aspect*error;
	double maxAR = aspect + aspect*error;
	
	double area = Core.countNonZero(m);
	double bbArea = m.cols()*m.rows();
	double percPixels = area/bbArea;
	
	if(percPixels < 0.8 && charAspect >= minAR && charAspect <= maxAR && m.rows() >= minH && m.rows() < maxH){
		return true;
	}
	else{
		return false;
	}
}
	
private CvSVM trainSVM() {
	// TODO Auto-generated method stub
	
	CvSVM svm = new CvSVM();
	return null;
}

private static Mat histeq(Mat in){
	Mat out = new Mat(in.size(), in.type());
    if(in.channels()==3){
        Mat hsv = in.clone();
        List<Mat> hsvSplit = new ArrayList<Mat>();
        Imgproc.cvtColor(in, hsv, Imgproc.COLOR_BGR2HSV);
        Core.split(hsv, hsvSplit);
        Imgproc.equalizeHist(hsvSplit.get(2), hsvSplit.get(2));
        Core.merge(hsvSplit, hsv);
        Imgproc.cvtColor(hsv, out, Imgproc.COLOR_HSV2BGR);
        
    }else if(in.channels()==1){
    	Imgproc.equalizeHist(in, out);
    }

    return out;	
}

private static boolean verifySizes(RotatedRect mr) 
{
	// TODO Auto-generated method stub
	float error = 0.4f;
	float aspect = 4.7272f;
	
	int min = (int) (15*aspect*15);
	int max = (int) (125*aspect*125);
	
	float rmin = aspect - aspect*error;
	float rmax = aspect + aspect*error;
	
	int area = (int) (mr.size.height * mr.size.width);
	float r = (float)mr.size.width / (float)mr.size.height;
	
	if(r < 1)
	{
		r = 1/r;
	}
	
	if(((area < min) || (area > max)) || ((r < rmin) || (r > rmax)))
	{
		return false;
	}
	
	return true;		
}

}

public class ANPR {
  public static void main(String[] args) throws ParserConfigurationException, SAXException, IOException, InterruptedException {
    System.out.println("Hello, OpenCV");

    // Load the native library. 
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

    new DetectFaceDemoDTC().run();
  }
}

