package detectingFlames;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Vector;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.CSVSaver;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.SMO;
import weka.classifiers.Evaluation;

import org.tartarus.snowball.ext.englishStemmer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Vector;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.io.*;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.TextDirectoryLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.StringToWordVector;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.LMT;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.j48.NBTreeSplit;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.TextDirectoryLoader;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class starter {

	private Instances instances;
	private Instances testInstances;	
	
	public Instances getInstances() {
		return instances;
	}
	public void setInstances(Instances instances) {
		this.instances = instances;
	}
	
	public Instances getTestInstances() {
		return testInstances;
	}
	public void setTestInstances(Instances testInstances) {
		this.testInstances = testInstances;
	}

	public static void main(String args[]) throws Exception
	{
		System.out.println("reaady to rockk and roll!");

	//	starter obj = new starter();
	//	obj.createInstance("train");
	//	obj.createInstance("test");
	//	obj.readCSV("train");
	//	obj.readCSV("test");
	//	obj.createTextDataFile("train");
	//	obj.createTextDataFile("test");
	//	obj.applyBatchFilter();
		//		obj.setTokenizer(1, 2, "train");
	//	obj.setTokenizer(1, 2, "test");
		starter.runTest(null);
	 
	}
	
	private static String pathNameTest = System.getProperty("user.dir")+"\\src\\detectingFlames\\";
	private static String pathName = System.getProperty("user.dir");
	private static final int NUM_TRAIN = 1; /*, 1000, 1250, 1500, 1750, 2000}; */
//	private static final String TRAIN_PREFIX = pathNameTest + "spam_train_", TRAIN_SUFFIX = ".arff", TEST = pathNameTest +"spam_test.arff"; 
	private static final String TRAIN_PREFIX = pathName + "\\QAFiles\\Traintokenized.arff", TEST = pathName +"\\QAFiles\\Testtokenized.arff"; 
	
	public static void runTest(String args[]) throws Exception {
		try{
		System.out.println(TRAIN_PREFIX);
		System.out.println(TEST);
		
		//File currentDirectory = new File(new File().getAbsolutePath());
		final ArffLoader eval_ld = new ArffLoader();
		eval_ld.setFile(new File(TEST));
		
		final Instances testInstances = eval_ld.getDataSet();
		testInstances.setClassIndex(testInstances.numAttributes() - 1);
		
		for (int n = 0; n < NUM_TRAIN; ++n) {
			final ArffLoader ld = new ArffLoader();
			ld.setFile(new File(TRAIN_PREFIX));
			final Instances trainInstances = ld.getDataSet();
			final NaiveBayesMultinomial nb = new NaiveBayesMultinomial();
			final SMO svm = new SMO();
			final IBk ibk = new IBk();
			final J48 j48 = new J48();
			final RandomForest rndmForest = new RandomForest();
			final AdaBoostM1 adaBoost = new AdaBoostM1();
			final LMT lmTrees = new LMT();
			String optionString = " -P 100 -S 1 -I 10 -W weka.classifiers.trees.DecisionStump";
			adaBoost.setOptions(weka.core.Utils.splitOptions(optionString));
			String [] options = weka.core.Utils.splitOptions("-K 50 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -D\\\"\"");
			ibk.setOptions(options);
			
		//	Instance inst;
			trainInstances.setClassIndex(trainInstances.numAttributes() - 1);
			rndmForest.buildClassifier(trainInstances);
			nb.buildClassifier(trainInstances);
			svm.buildClassifier(trainInstances);
			ibk.buildClassifier(trainInstances);
			j48.buildClassifier(trainInstances);
			adaBoost.buildClassifier(trainInstances); 
			lmTrees.buildClassifier(trainInstances);
			
//			svm.setOptions(weka.core.Utils.splitOptions("-K 1 -D 2"));
			/*
			while ((inst = ld.getNextInstance(trainInstances)) != null) nb.updateClassifier(inst);
			*/
			Evaluation eval = new Evaluation(trainInstances);
			/* System.out.println(nb); */
	//		eval.evaluateModel(nb, testInstances);
	//		System.out.println();
	//		System.out.println("NB model accuracy: " + eval.pctCorrect() +"\n" + "correct:"+eval.correct()+ "incorrect:"+ eval.incorrect());
			eval.evaluateModel(rndmForest, testInstances);
			System.out.println("Random Forest model accuracy: " + eval.pctCorrect() +"\n" + "correct:"+eval.correct()+ "incorrect:"+ eval.incorrect());
			
			eval = new Evaluation(trainInstances);
			eval.evaluateModel(ibk, testInstances);
			System.out.println("IBK model accuracy: " + eval.pctCorrect() +"\n" + "correct:"+eval.correct()+ "incorrect:"+ eval.incorrect());
			
			eval = new Evaluation(trainInstances);
			eval.evaluateModel(svm, testInstances);
			System.out.println("SVM model accuracy: " + eval.pctCorrect() +"\n" + "correct:"+eval.correct()+ "incorrect:"+ eval.incorrect());
			
			eval = new Evaluation(trainInstances);
			eval.evaluateModel(nb, testInstances);
			System.out.println("Naive Bayes model accuracy: " + eval.pctCorrect() +"\n" + "correct:"+eval.correct()+ "incorrect:"+ eval.incorrect());
			
			eval = new Evaluation(trainInstances);
			eval.evaluateModel(j48, testInstances);
			System.out.println("J48 model accuracy: " + eval.pctCorrect() +"\n" + "correct:"+eval.correct()+ "incorrect:"+ eval.incorrect());
			
			eval = new Evaluation(trainInstances);
			eval.evaluateModel(adaBoost, testInstances);
			System.out.println("Ada Boosting M1  model accuracy: " + eval.pctCorrect() +"\n" + "correct:"+eval.correct()+ "incorrect:"+ eval.incorrect());
			
			eval = new Evaluation(trainInstances);
			eval.evaluateModel(lmTrees, testInstances);
			System.out.println("Logistic Regression Trees model accuracy: " + eval.pctCorrect() +"\n" + "correct:"+eval.correct()+ "incorrect:"+ eval.incorrect());
			
		}
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
	}

	public static int indexFrom(String s,char chars, int lastIndex)
    {
        for (int i=0;i<s.length();i++)
           if (s.charAt(i) == chars && i>lastIndex)
              return i;
        return -1;
    } 
	 public void readCSV(String type) throws IOException {
		 	File currentDirectory = new File(new File(".").getAbsolutePath());
		 	System.out.println(currentDirectory.getCanonicalPath());
			
			BufferedReader br = null;
			String line = "";
			String csvFile = "";
			char csvSplitBy = ',';
			FileWriter writer = null;
			if (type.equals("train"))
			{
				csvFile = currentDirectory.getCanonicalPath()+ "/DataFiles/train.csv";
			writer = new FileWriter(currentDirectory.getCanonicalPath()+ "/QAFiles/trainPreProcess.txt");
			}
			else
			{
				csvFile = currentDirectory.getCanonicalPath()+ "/DataFiles/test_with_solutions.csv";
				writer = new FileWriter(currentDirectory.getCanonicalPath()+ "/QAFiles/testPreProcess.txt");
							
			}
			String comments[] = new String[3];
			
			try {
		 
				br = new BufferedReader(new FileReader(csvFile));
				int c =0;
				while ((line = br.readLine()) != null) {
					
						if(c++ <1)
						continue;
						System.out.println(c);
						
						int r = 0;	
						int commaIndex =0;
						int firstIndex =0;
						String s = "";
						while(r<2)
						{
							commaIndex = starter.indexFrom(line, csvSplitBy, commaIndex);
							String toSave = line.substring(firstIndex,commaIndex);
							firstIndex = commaIndex+1;
							comments[r] = toSave;
							r++;
						}
					comments[2] =line.substring(firstIndex,line.length());
					if(type.equals("train"))
					{
						this.addInstance(parseThroughComment(comments[2], writer),Integer.valueOf(comments[0]),"train");
					}
					else
						this.addInstance(parseThroughComment(comments[2], writer),Integer.valueOf(comments[0]),"test");
					}
			writer.flush();
		    writer.close();
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			} finally {
				if (br != null) {
					try {
						br.close();
					} catch (IOException e) {
						e.printStackTrace();
					}
				}
			}
		 
			System.out.println("Done with"+type+"reading");
		  }

	public String parseThroughComment(String comment, FileWriter fileWriter) throws IOException
	{	
		String processComment = " ";
		try
		{
		    //generate whatever data you want
			processComment = preProcessComment(comment);
			String[] words = processComment.split(" ");   
			for (String word : words)  
			{  
			   //System.out.println(word);  
			}  
			fileWriter.write(processComment +  System.getProperty("line.separator"));
		}
		catch(IOException e)
		{
		     e.printStackTrace();
		} 
		return processComment; 
	}
	private String preProcessComment(String comment) {
		comment = comment.replace('_', ' ');
		comment = comment.toLowerCase();
		comment = comment.replace("\\\\", "\\");
		comment = comment.replaceAll("\\\\n", "");
		comment = comment.replaceAll("\\\\r", "");
		comment = comment.replaceAll("\\\\'", "'");
		comment = comment.replaceAll("\\\\\\\\'", "'");
		comment = comment.replaceAll("\\\\\\\\","");
		comment = comment.replaceAll("-", "");
		comment = comment.replaceAll("--", "");
		comment = comment.replaceAll("#", "");
		comment = comment.replaceAll("&", "");
		comment = comment.replaceAll("%", "");
		comment = comment.replaceAll(">", "");
		comment = comment.replaceAll("<", "");
		comment = comment.replaceAll("=", "");
		comment = this.removeUTFCharacters(comment);
		comment = comment.replaceAll("\\\\t","");
		comment = comment.replaceAll("\\\\", "");
		comment = comment.replaceAll("class", ""); 
		comment = comment.replaceAll("[0-9]+","");
		comment = comment.replaceAll("privatetest","");
		comment = comment.replaceAll("publictest", "");
		comment = comment.replaceAll("'", "");
		comment = this.removeUnicodeCharacters(comment);
		comment = this.removeSpecialCharacters(comment);
		return comment;
	}
	
	public String removeUTFCharacters(String data){
		Pattern p = Pattern.compile("\\\\x(\\p{XDigit}{2})");
		Matcher m = p.matcher(data);
		StringBuffer buf = new StringBuffer(data.length());
		while (m.find()) {
		String ch = String.valueOf((char) Integer.parseInt(m.group(1), 16));
		m.appendReplacement(buf, Matcher.quoteReplacement(ch));
		}
		m.appendTail(buf);
		return buf.toString();
		}
	public String removeSpecialCharacters(String data){
		return data.replaceAll("[^a-zA-Z0-9]"," ");
		}
	public String removeUnicodeCharacters(String data){
		Pattern p = Pattern.compile("\\\\u(\\p{XDigit}{4})");
		Matcher m = p.matcher(data);
		StringBuffer buf = new StringBuffer(data.length());
		while (m.find()) {
		String ch = String.valueOf((char) Integer.parseInt(m.group(1), 16));
		m.appendReplacement(buf, Matcher.quoteReplacement(ch));
		}
		m.appendTail(buf);
		return buf.toString();
		}
	
	//Adding new stuff
	
	class Entry {
		HashMap<String, Integer> features = new HashMap<String, Integer>();
		int label;
		HashSet<Integer> subjectIndices = new HashSet<Integer>();
		String subjectNerTag;
		Vector<Vector<String>> words;
		String id;
		
		void addSubjectIndices(int start, int end) {
			for (int i = start; i <= end; ++i) {
				subjectIndices.add(new Integer(i));
			}
		}
		
		boolean isSubjectWord(int index) {
			return subjectIndices.contains(new Integer(index));			
		}
		
		int getWindowOfWordIndex(int index) {
			int min = 10000;
			Iterator<Integer> iter = subjectIndices.iterator();
			while (iter.hasNext()) {
				Integer i = iter.next();
				int val = Math.abs(i.intValue() - index);
				if (val < min) {
					min = val;
				}
			}
			if (min <= 5) {
				return 1;
			} else if (min <= 7) {
				return 2;
			} 
			return 3;
		}
	}
	
/*	class CVPair{
		Vector<Entry> train = new Vector<Entry>();
		Vector<Entry> test = new Vector<Entry>();
	}
	Vector<CVPair> buildCV(Vector<Entry> data, int fold) throws Exception {
		Vector<CVPair> pairs = new Vector<CVPair>();
		int amountInSet =  data.size() / fold;
		Vector<Integer> indices = new Vector<Integer>();
		for (int i = 0; i < data.size(); ++i)
			indices.add(new Integer(i));
		for (int i = 0; i < fold; ++i) {
			CVPair cvpair = new CVPair();
			pairs.add(cvpair);
			while (cvpair.test.size() < amountInSet) {
				int rand = (int)(Math.random() * indices.size());
				Entry entry = data.get(indices.get(rand).intValue());
				cvpair.test.add(entry);
				indices.remove(rand);
			}
		}
		
		for (int i = 0; i < pairs.size(); ++i) {
			CVPair p = pairs.get(i);
			for (int j = 0; j < pairs.size(); ++j) {
				if (j == i)
					continue;
				p.train.addAll(pairs.get(j).test);
			}
		}
		return pairs;
	}*/
	/*Stat testClassifier(Classifier classifier, Instances trainData, Instances testData) throws Exception {
		Stat stat = new Stat();
		classifier.buildClassifier(trainData);
		double accuracy = 0;
		for (int i = 0; i < testData.numInstances(); ++i) {
			Instance inst = testData.instance(i);
			double pred = classifier.classifyInstance(inst);
			if (inst.classValue() == pred) {
				accuracy++;
			} 
		}
		accuracy = accuracy / (double)testData.numInstances();
		stat.accuracyTest = accuracy;
		
		accuracy = 0;
		for (int i = 0; i < trainData.numInstances(); ++i) {
			Instance inst = trainData.instance(i);
			double pred = classifier.classifyInstance(inst);
			if (inst.classValue() == pred)
				accuracy++;
		}
		accuracy = accuracy / (double)trainData.numInstances();
		stat.accuracyTrain = accuracy;
		
		return stat;
	}*/
	
	/*class Stat{
		String classifierName;
		double accuracyTest;
		double accuracyTrain;
		
		
		void applyToAverage(Stat t, int amount) {
			accuracyTest = (accuracyTest * (amount - 1)) + t.accuracyTest;
			accuracyTest /= (double)amount;
			
			accuracyTrain = (accuracyTrain * (amount - 1)) + t.accuracyTrain;
			accuracyTrain /= (double)amount;
		}
		
		public String toString() {
			return classifierName + ": [ACC TEST: " + accuracyTest + ", ACC TRAIN: " + accuracyTrain + "]";
		}
	}*/
/*	Vector<Stat> runTrainAndTest(Vector<Entry> train, Vector<Entry> test, String stringToWordOptions) throws Exception {

	    Instances dataRaw = this.getInstances();
	    StringToWordVector filter = new StringToWordVector();
	    filter.setOptions(weka.core.Utils.splitOptions(stringToWordOptions));
	    filter.setInputFormat(dataRaw);
	    Instances trainFiltered = Filter.useFilter(dataRaw, filter);
	    System.out.println(trainFiltered);
//	    Reorder reorder = new Reorder();
//		reorder.setOptions(weka.core.Utils.splitOptions("-R 2-last,first "));
//		reorder.setInputFormat(trainFiltered);
//		trainFiltered = Filter.useFilter(trainFiltered, reorder);
		
	    dataRaw = this.getTestInstances();
	    filter.setInputFormat(dataRaw);
	    Instances testFiltered = Filter.useFilter(dataRaw, filter);
//	    Reorder reorder2 = new Reorder();
//	 		reorder2.setOptions(weka.core.Utils.splitOptions("-R 2-last,first"));
//	 		reorder2.setInputFormat(trainFiltered);
//		testFiltered = Filter.useFilter(testFiltered, reorder2);
	
		
		File currentDirectory = new File(new File(".").getAbsolutePath());
		String fileNameTrain = currentDirectory.getCanonicalPath()+ "/QAFiles/Traintokenized.txt";
		String fileNameTest = currentDirectory.getCanonicalPath()+ "/QAFiles/Testtokenized.txt";
		FileWriter writerTokenized = new FileWriter(fileNameTrain);
		writerTokenized.write(trainFiltered.toString());
		writerTokenized.close();
		writerTokenized = new FileWriter(fileNameTest);
		writerTokenized.write(testFiltered.toString());
		writerTokenized.close();
		
		Vector<Stat> stats = new Vector<Stat>();
		
//		Stat stat = null;
//		String []options = weka.core.Utils.splitOptions("");
//		Classifier classifier = new NaiveBayesMultinomial();
//		classifier.getCapabilities().enableAllAttributes();
		//NAIVE BAYES - multinomial
		Classifier classifier = new NaiveBayesMultinomial();
		classifier.getCapabilities().enableAllAttributes();
		String []options = weka.core.Utils.splitOptions("");
		//classifier.setOptions(options);
		Stat stat = testClassifier(classifier, trainFiltered, trainFiltered);
		stat.classifierName = "NaiveBayes";
		stats.add(stat);*/
//		
//		/// SVM with polynimial degree 2 kernel
//		classifier = new LibSVM();
//		options = weka.core.Utils.splitOptions("-K 1 -D 2");
//		classifier.setOptions(options);
//		stat = testClassifier(classifier, trainFiltered, testFiltered);
//		stat.classifierName = "LibSVM-POLY2";
//		stats.add(stat);
//		
//		/// SVM with linear kernel
//		classifier = new LibSVM();
//		options = weka.core.Utils.splitOptions("-K 0");
//		classifier.setOptions(options);
//		stat = testClassifier(classifier, trainFiltered, testFiltered);
//		stat.classifierName = "LibSVM-LIN";
//		stats.add(stat);
//		
//		/// kNN - no normalization (k = 1)
//		classifier = new IBk();
//		options = weka.core.Utils.splitOptions("-K 1 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -D\\\"\"");
//		classifier.setOptions(options);
//		stat = testClassifier(classifier, trainFiltered, testFiltered);
//		stat.classifierName = "KNN(1)";
//		stats.add(stat);
//		
//		/// kNN - no normalization (k = 3)
//		classifier = new IBk();
//		options = weka.core.Utils.splitOptions("-K 3 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -D\\\"\"");
//		classifier.setOptions(options);
//		stat = testClassifier(classifier, trainFiltered, testFiltered);
//		stat.classifierName = "KNN(3)";
//		stats.add(stat);
//		
//		/// kNN - no normalization (k = 5)
//		classifier = new IBk();
//		options = weka.core.Utils.splitOptions("-K 5 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -D\\\"\"");
//		classifier.setOptions(options);
//		stat = testClassifier(classifier, trainFiltered, testFiltered);
//		stat.classifierName = "KNN(5)";
//		stats.add(stat);
//		
//		/// kNN - no normalization (k = 10)
//		classifier = new IBk();
//		options = weka.core.Utils.splitOptions("-K 10 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -D\\\"\"");
//		classifier.setOptions(options);
//		stat = testClassifier(classifier, trainFiltered, testFiltered);
//		stat.classifierName = "KNN(10)";
//		stats.add(stat);
//		
//		/// kNN - no normalization (k = 20)
//		classifier = new IBk();
//		options = weka.core.Utils.splitOptions("-K 20 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -D\\\"\"");
//		classifier.setOptions(options);
//		stat = testClassifier(classifier, trainFiltered, testFiltered);
//		stat.classifierName = "KNN(20)";
//		stats.add(stat);
//		
//		/// kNN - no normalization (k = 30)
//		classifier = new IBk();
//		options = weka.core.Utils.splitOptions("-K 30 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -D\\\"\"");
//		classifier.setOptions(options);
//		stat = testClassifier(classifier, trainFiltered, testFiltered);
//		stat.classifierName = "KNN(30)";
//		stats.add(stat);
//		
//		/// kNN - no normalization (k = 50)
//		classifier = new IBk();
//		options = weka.core.Utils.splitOptions("-K 50 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -D\\\"\"");
//		classifier.setOptions(options);
//		stat = testClassifier(classifier, trainFiltered, testFiltered);
//		stat.classifierName = "KNN(50)";
//		stats.add(stat);
//		
//		/// kNN - with normalization (k = 1)
//		classifier = new IBk();
//		options = weka.core.Utils.splitOptions("-K 1 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance\\\"\"");
//		classifier.setOptions(options);
//		stat = testClassifier(classifier, trainFiltered, testFiltered);
//		stat.classifierName = "KNN(1)Norm";
//		stats.add(stat);
//		
//		/// kNN - with normalization (k = 3)
//		classifier = new IBk();
//		options = weka.core.Utils.splitOptions("-K 3 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance\\\"\"");
//		classifier.setOptions(options);
//		stat = testClassifier(classifier, trainFiltered, testFiltered);
//		stat.classifierName = "KNN(3)Norm";
//		stats.add(stat);
//		
//		/// kNN - with normalization (k = 5)
//		classifier = new IBk();
//		options = weka.core.Utils.splitOptions("-K 5 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance\\\"\"");
//		classifier.setOptions(options);
//		stat = testClassifier(classifier, trainFiltered, testFiltered);
//		stat.classifierName = "KNN(5)Norm";
//		stats.add(stat);
//		
//		/// kNN - with normalization (k = 10)
//		classifier = new IBk();
//		options = weka.core.Utils.splitOptions("-K 10 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance\\\"\"");
//		classifier.setOptions(options);
//		stat = testClassifier(classifier, trainFiltered, testFiltered);
//		stat.classifierName = "KNN(10)Norm";
//		stats.add(stat);
//		
//		/// kNN - with normalization (k = 20)
//		classifier = new IBk();
//		options = weka.core.Utils.splitOptions("-K 20 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance\\\"\"");
//		classifier.setOptions(options);
//		stat = testClassifier(classifier, trainFiltered, testFiltered);
//		stat.classifierName = "KNN(20)Norm";
//		stats.add(stat);
//		
//		/// kNN - with normalization (k = 30)
//		classifier = new IBk();
//		options = weka.core.Utils.splitOptions("-K 30 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance\\\"\"");
//		classifier.setOptions(options);
//		stat = testClassifier(classifier, trainFiltered, testFiltered);
//		stat.classifierName = "KNN(30)Norm";
//		stats.add(stat);
//		
		// kNN - with normalization (k = 50)
//		classifier = new IBk();
//		options = weka.core.Utils.splitOptions("-K 50 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance\\\"\"");
//		stat = testClassifier(classifier, trainFiltered, testFiltered);
//		stat.classifierName = "KNN(50)Norm";
//		stats.add(stat);
//		
//		return stats;
//	}
	public void createInstance(String type)
	{
			ArrayList<Attribute> atts = new ArrayList<Attribute>(2);
	        ArrayList<String> classVal = new ArrayList<String>();
	        classVal.add("?NotSure");
	        classVal.add("isGood");
	        classVal.add("isBad");
	        atts.add(new Attribute("content",(ArrayList<String>)null));
	        atts.add(new Attribute("class",classVal));

	        Instances dataRaw = new Instances("CommentsInstances",atts,0);
	        if (type.equals("train"))
	        {
	        this.setInstances(dataRaw);
	        }
	        else
	        {
	        	this.setTestInstances(dataRaw);
	        }

	    }
	
	public void addInstance(String text, int classification, String type)
	{
		Instances dataRaw;
		if (type.equals("train"))
		{
			dataRaw = this.getInstances();
		}
		else
		{
			dataRaw = this.getTestInstances();
		}
		//System.out.println(dataRaw.numAttributes());
		 double[] instanceValue1 = new double[dataRaw.numAttributes()];

	        instanceValue1[0] = dataRaw.attribute(0).addStringValue(text);
	        instanceValue1[1] = classification+1;
	       // System.out.println(classification);
	        dataRaw.add(new DenseInstance(1.0, instanceValue1));
	}
	
	public String stemWord(String s)
	{
	englishStemmer stemmer = new englishStemmer();
	stemmer.setCurrent(s);
	if (stemmer.stem()){
		return stemmer.getCurrent();
	}
	return s;
	}
	
	public void createTextDataFile(String type) throws IOException
	{
		Instances dataFiltered = null;
		String fileName = "";
		File currentDirectory = new File(new File(".").getAbsolutePath());
		if(type.equals("train")){
			dataFiltered = this.getInstances();
			System.out.println(dataFiltered);
			fileName = currentDirectory.getCanonicalPath()+ "/QAFiles/Traintokenized.arff";}
			else{		
			dataFiltered = this.getTestInstances();
			fileName = currentDirectory.getCanonicalPath()+ "/QAFiles/Testtokenized.arff";}	
		FileWriter fileWriter = new FileWriter(fileName);
		fileWriter.write(dataFiltered.toString());
		fileWriter.close();
	}
	
	public void applyBatchFilter() throws IOException
	{
		File currentDirectory = new File(new File(".").getAbsolutePath());
		String fileName = currentDirectory.getCanonicalPath()+ "/QAFiles/Traintokenized.arff";
		String csvFileName = currentDirectory.getCanonicalPath()+ "/QAFiles/Traintokenized.csv";
		System.out.println(Runtime.getRuntime().exec("java weka.core.converters.CSVSaver -i Testtokenized.arff -o test1234567.csv").getErrorStream().toString()
		);
	}
	
//	public void setTokenizer(int minNGrams, int maxNGrams,String type) throws Exception
//	{
//		
//		
//		NGramTokenizer tokenizer = new NGramTokenizer(); 
//		tokenizer.setNGramMinSize(minNGrams); 
//		tokenizer.setNGramMaxSize(maxNGrams); 
//		tokenizer.setDelimiters("\\W");
//		StringToWordVector filter = new StringToWordVector(); 
//		filter.setInputFormat(this.getInstances()); 
//		filter.setTokenizer(tokenizer); 
//		filter.setWordsToKeep(1000000); 
//		filter.setDoNotOperateOnPerClassBasis(true); 
//		filter.setLowerCaseTokens(true);
//		String arg[] = {"-S"};
//		filter.setOptions(arg);
//		Instances dataFiltered = null;
//		String fileName = "";
//		File currentDirectory = new File(new File(".").getAbsolutePath());
//		if(type.equals("train")){
//		dataFiltered = Filter.useFilter(this.getInstances(), filter);
//		fileName = currentDirectory.getCanonicalPath()+ "/QAFiles/Traintokenized.arff";}
//		else{		
//		dataFiltered = Filter.useFilter(this.getTestInstances(), filter);
//		fileName = currentDirectory.getCanonicalPath()+ "/QAFiles/Testtokenized.arff";}
//		Reorder reorder = new Reorder();
//		reorder.setOptions(weka.core.Utils.splitOptions("-R 2-last,first "));
//		reorder.setInputFormat(dataFiltered);
//		dataFiltered = Filter.useFilter(dataFiltered, reorder);
//		FilteredClassifier cls = new FilteredClassifier();
//		cls.setClassifier(new NaiveBayes());
//		dataFiltered.setClassIndex(dataFiltered.numAttributes()-1);
//		cls.setFilter(filter);
//		cls.buildClassifier(dataFiltered);
//		ObjectOutputStream oos = new ObjectOutputStream(
//        new FileOutputStream(currentDirectory.getCanonicalPath()+ "/QAFiles/Traintokenized.arff"));
//		oos.writeObject(cls);
//		oos.flush();
//		oos.close();
//	
//		//System.out.println("\n\nFiltered data:\n\n" + dataFiltered);
//		//System.out.println("done with"+type+"tokenizer");
//		//System.out.println(dataFiltered.toString());
//		
//	}
//	
//	public void runIt() throws Exception
//	{
//		
//		Vector<String> options = new Vector<String>();
//		options.add("");	// for FP weighting function
//		options.add("-T");	// for FF weighting function
//		options.add("-I -T");	// for TFIDF weighting function
//		for (int tt = 0; tt < options.size(); ++tt) {
//			System.out.println("WEIGHTING FUNCTION: " + options.get(tt));
//			System.out.println("+++++++++++++++++++++++++++++++++");
//			Vector<Stat> stats = runTrainAndTest(null, null,  options.get(tt));
//			System.out.println(stats);
//		}
//		
//	}
}
	

	

	
	
	
	
	
	
	

