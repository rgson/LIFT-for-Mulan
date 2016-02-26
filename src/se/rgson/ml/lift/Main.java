package se.rgson.ml.lift;

import mulan.classifier.MultiLabelLearner;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.*;

public class Main {

    public static void main(String[] args) throws Exception {

        String[] datasets = {
//                "enron",
                "scene",
                "yeast",
        };

        for (String dataset : datasets) {
            System.out.println("====================");
            System.out.println(dataset);
            System.out.println();

            MultiLabelInstances multiLabelInstances = new MultiLabelInstances(
                    "datasets/" + dataset + ".arff",
                    "datasets/" + dataset + ".xml");

            runEvaluations(multiLabelInstances, new LIFT());
        }

    }

    /**
     * Runs some evaluations on the given multi-label dataset.
     *
     * @param instances A multi-label dataset.
     */
    private static void runEvaluations(MultiLabelInstances instances, MultiLabelLearner learner) {
        final int NUM_FOLDS = 10;
        Evaluator eval = new Evaluator();
        MultipleEvaluation evaluations = eval.crossValidate(learner, instances, NUM_FOLDS);
        printEvaluationResults(evaluations);
    }

    /**
     * Prints evaluation results for the following metrics:
     * - Average precision
     * - Coverage
     * - Ranking loss
     * - Hamming loss
     * - One erroe
     *
     * @param results The evaluation results.
     */
    private static void printEvaluationResults(MultipleEvaluation results) {
        final Measure[] MEASURES = new Measure[]{
                new AveragePrecision(),
                new Coverage(),
                new RankingLoss(),
                new HammingLoss(),
                new OneError(),
        };

        for (Measure measure : MEASURES) {
            String name = measure.getName();
            System.out.println(String.format("%20s: %.4f±%.4f (ideal: %s)", name, results.getMean(name),
                    results.getStd(name), measure.getIdealValue()));
        }
    }

}
