import mulan.classifier.MultiLabelLearner;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.*;
import se.rgson.ml.LIFT;

import java.util.concurrent.TimeUnit;

public class Main {

    static final String DATASET = "datasets/flags";

    static final int NUM_FOLDS = 10;

    static final Measure[] MEASURES = new Measure[]{
            new AveragePrecision(),
            new Coverage(),
            new RankingLoss(),
            new HammingLoss(),
            new OneError(),
    };

    public static void main(String[] args) throws Exception {

        long startTime = System.currentTimeMillis();

        LIFT lift = new LIFT();
        Evaluator eval = new Evaluator();
        MultiLabelInstances mlInstances = new MultiLabelInstances(DATASET + ".arff", DATASET + ".xml");

        MultipleEvaluation evaluations = eval.crossValidate(lift, mlInstances, NUM_FOLDS);

        for (Measure measure : MEASURES) {
            String name = measure.getName();
            System.out.println(String.format("%20s: %.4f±%.4f (ideal: %s)",
                    name, evaluations.getMean(name), evaluations.getStd(name), measure.getIdealValue()));
        }

        long elapsedTime = System.currentTimeMillis() - startTime;
        long minutes = TimeUnit.MILLISECONDS.toMinutes(elapsedTime);
        long seconds = TimeUnit.MILLISECONDS.toSeconds(elapsedTime)
                - TimeUnit.MINUTES.toSeconds(minutes);

        System.out.printf("Finished evaluation for dataset '%s' in %d min %d sec.\n", DATASET, minutes, seconds);

    }

}
