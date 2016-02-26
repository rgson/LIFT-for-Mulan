package se.rgson.ml.lift;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.TransformationBasedMultiLabelLearner;
import mulan.data.MultiLabelInstances;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.clusterers.SimpleKMeans;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;

/**
 * An implementation of the LIFT algorithm.
 *
 * Based on the description by Zhang and Wu [1].
 *
 * [1] M. L. Zhang and L. Wu, "Lift: Multi-Label Learning with Label-Specific
 * Features," in IEEE Transactions on Pattern Analysis and Machine Intelligence,
 * vol. 37, no. 1, pp. 107-120, Jan. 1 2015. doi: 10.1109/TPAMI.2014.2339815
 *
 * @author Robin Gustafsson
 */
public class LIFT extends TransformationBasedMultiLabelLearner {

    /**
     * The clustering ratio. Used to determine the number of clusters when
     * constructing label-specific features. Should be in the range [0, 1].
     */
    private final float r;

    /**
     * Constructs an instance of LIFT using a clustering ratio of 0.1 and
     * weka.classifiers.functions.SMO as the base classifier.
     */
    public LIFT() {
        this(0.1f, new SMO());
    }

    /**
     * Constructs an instance of LIFT using weka.classifiers.functions.SMO as
     * the base classifier.
     * @param clusteringRatio The clustering ratio, in the range of [0, 1].
     */
    public LIFT(float clusteringRatio) {
        this(clusteringRatio, new SMO());
    }

    /**
     * Constructs an instance of LIFT using a clustering ratio of 0.1.
     * @param baseClassifier The binary classifier to use for each label.
     */
    public LIFT(Classifier baseClassifier) {
        this(0.1f, baseClassifier);
    }

    /**
     * Constructs an instance of LIFT.
     * @param clusteringRatio The clustering ratio, in the range of [0, 1].
     * @param baseClassifier The binary classifier to use for each label.
     */
    public LIFT(float clusteringRatio, Classifier baseClassifier) {
        super(baseClassifier);
        this.r = clusteringRatio;
    }

    @Override
    protected void buildInternal(MultiLabelInstances multiLabelInstances) throws Exception {
        // TODO implement
        throw new NotImplementedException();
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        // TODO implement
        throw new NotImplementedException();
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation info = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        info.setValue(TechnicalInformation.Field.AUTHOR , "M. L. Zhang and L. Wu");
        info.setValue(TechnicalInformation.Field.ISSN   , "0162-8828");
        info.setValue(TechnicalInformation.Field.JOURNAL, "IEEE Transactions on Pattern Analysis and Machine Intelligence");
        info.setValue(TechnicalInformation.Field.MONTH  , "1");
        info.setValue(TechnicalInformation.Field.NUMBER , "1");
        info.setValue(TechnicalInformation.Field.PAGES  , "107-120");
        info.setValue(TechnicalInformation.Field.TITLE  , "Lift: Multi-Label Learning with Label-Specific Features");
        info.setValue(TechnicalInformation.Field.VOLUME , "37");
        info.setValue(TechnicalInformation.Field.YEAR   , "2015");
        return info;
    }

    @Override
    public String toString() {
        return String.format("LIFT(r=%f, classifier=%s)", this.r, this.baseClassifier.getClass().getCanonicalName());
    }


    // =========================================================================
    // private class KMeansClusterer
    // =========================================================================

    /**
     * K-means clustering algorithm for label-specific feature selection.
     * The actual clustering is handled by the SimpleKMeans superclass.
     */
    private class KMeansClusterer extends SimpleKMeans {

        /**
         * Gets the distance from an instance to each of the cluster centroids.
         * @param instance The instance.
         * @return The distance from the instance to each of the centroids.
         */
        public double[] getDistances(Instance instance) {
            int numClusters = super.getNumClusters();
            Instances clusterCentroids = super.getClusterCentroids();
            DistanceFunction distanceFunction = super.getDistanceFunction();

            double[] distances = new double[numClusters];
            for (int i = 0; i < numClusters; i++) {
                distances[i] = distanceFunction.distance(instance, clusterCentroids.instance(i));
            }

            return distances;
        }

    }

}
