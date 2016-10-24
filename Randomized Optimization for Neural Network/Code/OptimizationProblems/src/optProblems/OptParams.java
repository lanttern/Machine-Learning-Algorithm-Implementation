/**
 * 
 */
package optProblems;

import java.util.ArrayList;
import java.util.List;

import opt.EvaluationFunction;
import opt.HillClimbingProblem;
import opt.OptimizationAlgorithm;
import opt.OptimizationProblem;
import opt.SimulatedAnnealing;
import opt.example.TravelingSalesmanEvaluationFunction;
import shared.FixedIterationTrainer;

/**
 * @author zhihuixie
 *
 */
public class OptParams {
    /**
     * method to get fitness values for parameter optimization test
     * @param algs
     * @param ef
     * @return
     */
	public List<Double> trainFunction(List<OptimizationAlgorithm> algs, EvaluationFunction ef) {
		List<Double> fitnessScore = new ArrayList<>();
		for (int i = 0; i < algs.size(); i++){
             FixedIterationTrainer fit = new FixedIterationTrainer(algs.get(i), 2000);
	         fit.train();
	         fitnessScore.add(ef.value(algs.get(i).getOptimal()));	
		}
		
		return fitnessScore;
	}

}
