/**
 * @author zhihuixie
 *
 */
package AnnAnalysis;

import java.util.ArrayList;
import java.util.List;

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;
import func.nn.backprop.StochasticBackPropagationTrainer;
import shared.ConvergenceTrainer;
import shared.DataSet;
import shared.SumOfSquaresError;

public class BackProp {
	ConvergenceTrainer trainer;
	BackPropagationNetwork clf;
	/**
	 * constructor for BackProp class
	 * @param clf
	 */
	public BackProp(BackPropagationNetwork clf){
		this.clf = clf;
	}
	// train a back prop neural network classifier
	/** method to train a classifier
	 * @param data
	 */
	public void trainModel(DataSet data){
	      this.trainer = new ConvergenceTrainer(new StochasticBackPropagationTrainer(data, 
	    		             this.clf, new SumOfSquaresError(), new RPROPUpdateRule()));
          this.trainer.train();
		
	}
	
	/**
	 * method to predict using input data
	 * @param X
	 * @return
	 */
	public List<Double> predict(List<ArrayList<Double>> X){
		List<Double> preds = new ArrayList<Double>();
        for (int j = 0; j < X.size(); j++) {
        	convertToVector convert = new convertToVector();
        	this.clf.setInputValues(convert.convertDoubleListToVector(X.get(j)));
            this.clf.run();
            preds.add(this.clf.getOutputValues().get(0));
        }
        return preds;
		
	}

}
