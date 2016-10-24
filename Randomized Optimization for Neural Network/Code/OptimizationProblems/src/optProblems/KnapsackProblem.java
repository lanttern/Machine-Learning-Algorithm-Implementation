package optProblems;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.NQueensFitnessFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * A test of the knap sack problem
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class KnapsackProblem {
    /** Random number generator */
    private static final Random random = new Random();
    /** The number of items */
    private static final int NUM_ITEMS = 40;
    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum volume for a single element */
    private static final double MAX_VOLUME = 50;
    /** The volume of the knapsack */
    private static final double KNAPSACK_VOLUME = 
         MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4;
    /**
     * method to generate optimization fitness values 
     * data for knapsack problems
     * @param args ignored
     */
    public static void experiment(String name) {
        String probName = "KnapsackProblem";
    	List<OptimizationAlgorithm> algs = new ArrayList<>();
        int[] copies = new int[NUM_ITEMS];
        Arrays.fill(copies, COPIES_EACH);
        double[] weights = new double[NUM_ITEMS];
        double[] volumes = new double[NUM_ITEMS];
        for (int i = 0; i < NUM_ITEMS; i++) {
            weights[i] = random.nextDouble() * MAX_WEIGHT;
            volumes[i] = random.nextDouble() * MAX_VOLUME;
        }
         int[] ranges = new int[NUM_ITEMS];
        Arrays.fill(ranges, COPIES_EACH + 1);
        EvaluationFunction ef = new KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 25, gap);
        MIMIC mimic = new MIMIC(200, 100, pop);
        
        algs.add(rhc);
        algs.add(sa);
        algs.add(ga);
        algs.add(mimic);
        Experiment newExp = new Experiment();
        
        
        newExp.experiments(algs, ef, name);
        
        // test temperature for SA
        String paramName = "Temperature";
        String algName = "SA";
        List<Double> params2 = new ArrayList<>();
        List<OptimizationAlgorithm> algsTest = new ArrayList<>();
        for (double i = 100; i < 1001; i+=10){
        	params2.add(i);
        	sa = new SimulatedAnnealing(i, .95, hcp);
        	algsTest.add(sa);
        }
        newExp.optParams(algsTest, ef, paramName, null, params2, algName, probName);
        
        // test cooling rate for SA
        paramName = "CoolingRate";
        params2 = new ArrayList<>();
        algsTest = new ArrayList<>();
        for (double i = 0.01; i < 1; i+=0.01){
        	params2.add(i);
        	sa = new SimulatedAnnealing(100, i, hcp);
        	algsTest.add(sa);
        }
        newExp.optParams(algsTest, ef, paramName, null, params2, algName, probName);
        
        // test populationSize for GA
        paramName = "populationSize";
        algName = "GA";
        List<Integer> params1 = new ArrayList<>();
        algsTest = new ArrayList<>();
        for (int i = 200; i < 1201; i+=10){
        	params1.add(i);
        	ga = new StandardGeneticAlgorithm(i, 150, 25, gap);
        	algsTest.add(ga);
        }
        newExp.optParams(algsTest, ef, paramName, params1, null, algName, probName);
        
        // test toMate for GA
        paramName = "toMate";
        params1 = new ArrayList<>();
        algsTest = new ArrayList<>();
        for (int i = 2; i < 200; i+=2){
        	params1.add(i);
        	ga = new StandardGeneticAlgorithm(200, i, 25, gap);
        	algsTest.add(ga);
        }
        newExp.optParams(algsTest, ef, paramName, params1, null, algName, probName);
        
        // test toMutate for GA
        paramName = "toMutate";
        params1 = new ArrayList<>();
        algsTest = new ArrayList<>();
        for (int i = 10; i < 1001; i+=10){
        	params1.add(i);
        	ga = new StandardGeneticAlgorithm(200, 150, i, gap);
        	algsTest.add(ga);
        }
        newExp.optParams(algsTest, ef, paramName, params1, null, algName, probName);
        
        // test samples for MIMIC
        paramName = "samples";
        algName = "MIMIC";
        params1 = new ArrayList<>();
        algsTest = new ArrayList<>();
        for (int i = 200; i < 1201; i += 10){
        	params1.add(i);
        	mimic = new MIMIC(i, 100, pop);
        	algsTest.add(mimic);
        }
        newExp.optParams(algsTest, ef, paramName, params1, null, algName, probName);
        
        // test tokeep for MIMIC
        algName = "MIMIC";
        paramName = "tokeeep";
        params1 = new ArrayList<>();
        algsTest = new ArrayList<>();
        for (int i = 1; i < 101; i+=1){
        	params1.add(i);
        	mimic = new MIMIC(200, i, pop);
        	algsTest.add(mimic);
        }
        newExp.optParams(algsTest, ef, paramName, params1, null, algName, probName); 
        
        // test different algorithms with various NQueensProblems
        // set up algorithms
        algs = new ArrayList<>();
        sa = new SimulatedAnnealing(550, .87, hcp);
        ga = new StandardGeneticAlgorithm(320, 196, 20, gap);
        mimic = new MIMIC(1200, 100, pop);
        algs.add(rhc);
        algs.add(sa);
        algs.add(ga);
        algs.add(mimic);
        // set up different efs
        List<EvaluationFunction> efs = new ArrayList<>();
        for (double i = 0.1; i < 5.1; i+=0.1){
        	ef = new KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME*i, copies);
        	efs.add(ef);
        }
        newExp.voteBest(algs, efs, 2000, probName);
    }

}
