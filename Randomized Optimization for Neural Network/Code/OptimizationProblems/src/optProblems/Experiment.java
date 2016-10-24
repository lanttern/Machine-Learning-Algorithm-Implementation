/**
 * 
 */
package optProblems;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import opt.EvaluationFunction;
import opt.HillClimbingProblem;
import opt.OptimizationAlgorithm;
import opt.OptimizationProblem;

/**
 * @author zhihuixie
 *
 */
public class Experiment {
    public List<String> names;
	/**
	 * method to generate optimization fitness values data
	 * @param algs
	 * @param ef
	 * @param name
	 */
	public void experiments(List<OptimizationAlgorithm> algs, EvaluationFunction ef, 
			               String name){
        // add names and algorithms  
		names = new ArrayList<>();
	    names.add("RHC");
	    names.add("SA");
	    names.add("GA");
	    names.add("MIMC");
        // check fitness to find iterations to converge
        int maxIters = 2001;
        List<List<Double>> fitnessValues = new ArrayList<>();
        List<Integer> numOfIters = new ArrayList<>();
        double timeStart = System.nanoTime();
        for(int i = 10; i < maxIters; i += 10) numOfIters.add(i);
        for (int i = 0; i < algs.size(); i++){
        	System.out.println(name + ": start converge analysis of " + names.get(i));
            CheckConverge cc = new CheckConverge(algs.get(i), ef, maxIters);
            List<Double> fitness = cc.fitnessValue();
            fitnessValues.add(fitness);
            System.out.println(name + ": completed converge analysis of " + names.get(i));
        }
        outputFitnessData(name + "FitnessData.csv", numOfIters, fitnessValues);
        System.out.println("Fitness running time for " + name + ": " + 
                          (System.nanoTime() - timeStart)/1E9 + " seconds.\n");
        
    }
	
	/**
	 * method to generate optimization fitness values data 
	 * using different parameters
	 * @param algs
	 * @param ef
	 * @param paramName
	 * @param params1
	 * @param params2
	 * @param algName
	 * @param probName
	 */
    public void optParams(List<OptimizationAlgorithm> algs, EvaluationFunction ef,
    		             String paramName,List<Integer>params1, 
    		             List<Double>params2, String algName, String probName){
    	double timeStart = System.nanoTime();
    	OptParams op = new OptParams();
        List<Double> fitnessParams = op.trainFunction(algs, ef);
        // write data to file
        String fileName = probName + algName + paramName+ "ParametersOptData.csv";
        outputOptParamsData(fileName, paramName,params1, params2, fitnessParams);
        System.out.println("Parameter optimization running time for " + probName + " "+ 
                           algName + " " + paramName + ": " + 
        		           (System.nanoTime() - timeStart)/1E9 + " seconds.\n");
    }
    
    /**
     * method to generate optimization fitness values data in 50 trials
     * @param algs
     * @param efs
     * @param bestIters
     * @param probName
     */
    public void voteBest(List<OptimizationAlgorithm> algs, List<EvaluationFunction> efs,
    		            int bestIters, String probName){
        // add names and algorithms 
    	names = new ArrayList<>();
	    names.add("RHC");
	    names.add("SA");
	    names.add("GA");
	    names.add("MIMC");
	    double timeStart = System.nanoTime();
        BestAlg bestAlg = new BestAlg(algs, names, efs);
        List<Map> bestVote= bestAlg.voteAlg(bestIters);
        // write data to file
        String fileName = probName + "BestVoteData.csv";
        outputBestAlg(fileName, bestVote);
        System.out.println("BestVote running time for " + probName +
		                   ": " + (System.nanoTime() - timeStart)/1E9 + " seconds.\n");
    }
    
    /**
     * method to write optimization fitness data to files
     * @param fileName
     * @param iters
     * @param values
     */
    public static void outputFitnessData(String fileName, 
    		                  List<Integer> iters, List<List<Double>> values){
    	//output
        File file = new File(fileName);
        FileWriter writer;
    	   try {
    		   writer = new FileWriter(file);
    	       PrintWriter pwtr = new PrintWriter(new BufferedWriter(writer));
    	       pwtr.println("Number of iters, FitnessRHC, FitnessSA, "
    	       		       + "FitnessGA, FitnessMIMIC");
    	       for(int i = 0; i < iters.size(); i++) {
    	           pwtr.println(iters.get(i) + "," + values.get(0).get(i) + "," 
    	                       + values.get(1).get(i) + "," + values.get(2).get(i)
    	                       + "," + values.get(3).get(i));
    	       }
    	       pwtr.close();
    	       System.out.println("Fitness data written to file SUCCEED!");
    	    } catch (IOException e) {
    		// TODO Auto-generated catch block
    		   e.printStackTrace();
    		   System.out.println("Fitness data written to file FAIL!");
    	  }
    }
    
    /**
     * method to write parameter optimization fitness data to files
     * @param fileName
     * @param paramName
     * @param params1
     * @param params2
     * @param values
     */
    public static void outputOptParamsData(String fileName, String paramName,
            List<Integer> params1, List<Double> params2, List<Double> values){
         //output
         File file = new File(fileName);
         FileWriter writer;
         try {
              writer = new FileWriter(file);
              PrintWriter pwtr = new PrintWriter(new BufferedWriter(writer));
              pwtr.println(paramName + ", fitnessValue");
              for(int i = 0; i < values.size(); i++) {
            	  if (params1 != null){
                	  pwtr.println(params1.get(i) + "," + values.get(i));
            	  }
            	  else {
            		  pwtr.println(params2.get(i) + "," + values.get(i));
            	  }

              }
              pwtr.close();
              System.out.println("OptParams data written to file SUCCEED!");
         } catch (IOException e) {
        	 e.printStackTrace();
        	 System.out.println("OptParams data written to file FAIL!");
         }
    }
    
    /**
     * method to write fitness data in 50 trials to files
     * @param fileName
     * @param values
     */
    public static void outputBestAlg(String fileName, List<Map> values){
    	//output
        File file = new File(fileName);
        FileWriter writer;
    	   try {
    		   writer = new FileWriter(file);
    	       PrintWriter pwtr = new PrintWriter(new BufferedWriter(writer));
    	       pwtr.println("algName, fitnessScore, runTime");
    	       Map<String, List<Double>> scores = values.get(0);
    		   Map<String, List<Double>> runTime = values.get(1);
    	       for(String name:scores.keySet()) {
    	    	   for (int i = 0; i < scores.get(name).size(); i++){
    	    		   pwtr.println(name + "," + scores.get(name).get(i) + "," 
    	    		               + runTime.get(name).get(i));
    	    	   }
    	       }
    	       pwtr.close();
    	       System.out.println("Best vote data written to file SUCCEED!");
    	    } catch (IOException e) {
    		// TODO Auto-generated catch block
    		   e.printStackTrace();
    		   System.out.println("Best vote data written to file FAIL!");
    	  }
    }

}
