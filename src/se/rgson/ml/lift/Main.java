package se.rgson.ml.lift;

import mulan.classifier.MultiLabelLearner;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.*;

public class Main {

    public static void main(String[] args) throws Exception {

        final String DATASET = "scene";

        long time = System.currentTimeMillis();

        MultiLabelInstances multiLabelInstances = new MultiLabelInstances(
                "datasets/" + DATASET + ".arff",
                "datasets/" + DATASET + ".xml");

        runEvaluations(multiLabelInstances, new LIFT());

        float seconds = (System.currentTimeMillis() - time) / 1000;
        System.out.printf("Finished evaluation for dataset '%s' in %f.2 seconds.\n", DATASET, seconds);

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
