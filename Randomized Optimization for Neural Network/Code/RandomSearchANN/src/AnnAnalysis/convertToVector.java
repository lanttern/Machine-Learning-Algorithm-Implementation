package AnnAnalysis;

import java.util.Iterator;
import java.util.List;

import util.linalg.DenseVector;
import util.linalg.Vector;

public class convertToVector {
	
	/**
	 * method to convert an ArrayList<Double> to a double[]  
	 * @param list
	 * @return
	 */
    public static double[] convertDoubleListToArray(List<Double> list) {
        double[] ret = new double[list.size()];
        Iterator<Double> iterator = list.iterator();
        for (int i = 0; i < ret.length; i++) {
            ret[i] = iterator.next().intValue();
        }
        return ret;
    }
    
    /**
     * method to convert an ArrayList<Double> to a util.linalg.Vector
     * @param list
     * @return
     */
    public Vector convertDoubleListToVector(List<Double> list) {
		return new DenseVector(convertDoubleListToArray(list));
    }

}
