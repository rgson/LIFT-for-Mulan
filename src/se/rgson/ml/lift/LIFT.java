package se.rgson.ml.lift;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.TransformationBasedMultiLabelLearner;
import mulan.data.MultiLabelInstances;
import mulan.transformations.BinaryRelevanceTransformation;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.clusterers.SimpleKMeans;
import weka.core.*;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.function.Function;

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
     * The Binary Relevance transformation used to turn the multi-label
     * instances into binary classification instances.
     */
    private BinaryRelevanceTransformation brTransformation;

    /**
     * The instance mapping for each label.
     * Maps from an original instance to an instance using the label-specific
     * features, based on the performed clustering.
     */
    private InstanceMappingFunction[] mapping;

    /**
     * The trained classifiers for each label.
     */
    private Classifier[] classifiers;

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
        this.brTransformation = new BinaryRelevanceTransformation(multiLabelInstances);
        this.mapping = new InstanceMappingFunction[this.numLabels];
        this.classifiers = new Classifier[this.numLabels];
        Instances[] instancesByLabel = doBRTransformation(multiLabelInstances);

        for (int label = 0; label < this.numLabels; label++) {
            Instances instances = instancesByLabel[label];

            // Form P_k and N_k based on D according to Eq.(1);
            Instances posInstances = getPositiveInstances(instances);
            Instances negInstances = getNegativeInstances(instances);

            // Perform k-means clustering on P_k and N_k, each with m_k clusters
            // as defined in Eq.(2);
            int numClusters = (int) Math.ceil(this.r * Math.min(posInstances.numInstances(), negInstances.numInstances()));
            KMeans posClustering = new KMeans(posInstances, numClusters);
            KMeans negClustering = new KMeans(negInstances, numClusters);

            // Create the mapping phi_k for l_k according to Eq.(3);
            this.mapping[label] = new InstanceMappingFunction(posClustering, negClustering);

            // Form B_k according to Eq.(4);
            Instances labelSpecificDataset = createLabelSpecificDataset(instances, this.mapping[label]);

            // Induce g_k by invoking L on B_k, i.e. g_k <- L(B_k);
            this.classifiers[label] = AbstractClassifier.makeCopy(this.baseClassifier);
            this.classifiers[label].buildClassifier(labelSpecificDataset);
        }
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        boolean[] bipartition = new boolean[numLabels];
        double[] confidences = new double[numLabels];

        for (int label = 0; label < numLabels; label++) {
            Instance transformedInstance = this.brTransformation.transformInstance(instance, label);
            transformedInstance = this.mapping[label].apply(transformedInstance);

            double distribution[] = this.classifiers[label].distributionForInstance(transformedInstance);
            int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

            bipartition[label] = (maxIndex == 1) ? true : false;
            confidences[label] = distribution[1];
        }

        return new MultiLabelOutput(bipartition, confidences);
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

    /**
     * Splits the multi-label classification instances into binary
     * classification instance for each label using the Binary Relevance
     * transformation.
     * @param multiLabelInstances The multi-label classification instances.
     * @return The binary classification instances for each label.
     */
    private Instances[] doBRTransformation(MultiLabelInstances multiLabelInstances) throws Exception {
        Instances[] instances = new Instances[this.numLabels];
        for (int label = 0; label < this.numLabels; label++) {
            instances[label] = this.brTransformation.transformInstances(label);
        }
        return instances;
    }

    /**
     * Gets the positive instances from a binary classification instance set.
     * @param instances The full set of instances.
     * @return The positive instances.
     */
    private Instances getPositiveInstances(Instances instances) {
        return filterByAttributeValue(instances, instances.classAttribute(), 1);
    }

    /**
     * Gets the negative instances from a binary classification instance set.
     * @param instances The full set of binary classification instances.
     * @return The negative instances.
     */
    private Instances getNegativeInstances(Instances instances) {
        return filterByAttributeValue(instances, instances.classAttribute(), 0);
    }

    /**
     * Gets the instances with a specific value for the given attribute.
     * @param instances The full set of binary classification instances.
     * @param attribute The attribute to filter by.
     * @param value The value to filter by.
     * @return The instances with a specific value for the given attribute.
     */
    private Instances filterByAttributeValue(Instances instances, Attribute attribute, double value) {
        Instances result = new Instances(instances, instances.numInstances());
        instances.stream()
                .filter(instance -> instance.value(attribute) == value)
                .forEach(result::add);
        return result;
    }

    /**
     * Creates a set of binary classification instances with label-specific
     * features. The created set of instances contains all instances from the
     * provided set of binary classification instances. The provided mapping
     * function is used for transforming instances to the new feature space.
     * @param original A set of binary classification instances.
     * @param instanceMapping The mapping function to use for transformation.
     * @return The created set of binary classification instances.
     */
    private Instances createLabelSpecificDataset(Instances original, InstanceMappingFunction instanceMapping) {
        // Create the attribute info.
        int dimensionality = instanceMapping.getDimensionality();
        ArrayList<Attribute> attributes = new ArrayList<>(dimensionality);
        for (int i = 0; i < dimensionality - 1; i++) {
            attributes.add(new Attribute("attribute_" + i));
        }
        attributes.add(copyNominalAttribute(original.classAttribute()));

        // Create the instance set.
        Instances labelSpecific = new Instances("", attributes, original.numInstances());
        labelSpecific.setClassIndex(labelSpecific.numAttributes() - 1);
        original.stream()
                .map(instanceMapping)
                .forEach(labelSpecific::add);

        return labelSpecific;
    }

    /**
     * Creates a copy of a nominal attribute with the same name and values.
     * @param attribute A nominal attribute.
     * @return A new nominal attribute with the same name and values.
     */
    private Attribute copyNominalAttribute(Attribute attribute) {
        List<String> nominalValues = new ArrayList<>(attribute.numValues());
        Enumeration originalValues = attribute.enumerateValues();
        while (originalValues.hasMoreElements()) {
            nominalValues.add((String) originalValues.nextElement());
        }
        return new Attribute(attribute.name(), nominalValues);
    }


    // =========================================================================
    // Inner classes
    // =========================================================================

    /**
     * K-means clustering algorithm for label-specific feature selection.
     * The actual clustering is handled by the SimpleKMeans superclass.
     */
    private class KMeans extends SimpleKMeans {

        /**
         * Construcs a new KMeans and builds the clusters.
         * @param instances The instances to cluster.
         * @param numClusters The number of clusters.
         * @throws Exception
         */
        public KMeans(Instances instances, int numClusters) throws Exception {
            Instances classless = removeClassAttribute(instances);
            this.setDistanceFunction(new EuclideanDistance());
            this.setNumClusters(numClusters);
            this.buildClusterer(classless);
        }

        /**
         * Gets the distance from an instance to each of the cluster centroids.
         * @param instance The instance.
         * @return The distance from the instance to each of the centroids.
         */
        public double[] getDistances(Instance instance) {
            int numClusters = super.getNumClusters();
            Instances clusterCentroids = super.getClusterCentroids();
            DistanceFunction distanceFunction = super.getDistanceFunction();
            Instance classless = removeClassAttribute(instance);

            double[] distances = new double[numClusters];
            for (int i = 0; i < numClusters; i++) {
                distances[i] = distanceFunction.distance(classless, clusterCentroids.instance(i));
            }

            return distances;
        }

        /**
         * Removes the class attribute from a set of instances.
         * @param instances The set of instances.
         * @return The set of instances with the class attribute removed.
         */
        private Instances removeClassAttribute(Instances instances) {
            int classIndex = instances.classIndex();
            if (classIndex > -1) {
                instances = new Instances(instances);
                instances.setClassIndex(-1);
                instances.deleteAttributeAt(classIndex);
            }
            return instances;
        }

        /**
         * Removes the class attribute from an instance.
         * @param instance The instance.
         * @return The instance with the class attribute removed.
         */
        private Instance removeClassAttribute(Instance instance) {
            int classIndex = instance.classIndex();
            if (classIndex > -1) {
                instance = new DenseInstance(instance);
                instance.deleteAttributeAt(classIndex);
            }
            return instance;
        }

    }

    /**
     * A mapping function, transforming an instance from the original dataset to
     * an instance of label-specific features.
     */
    private final class InstanceMappingFunction implements Function<Instance, Instance> {

        /**
         * The clusters of positive instances.
         */
        private final KMeans positives;

        /**
         * The clusters of negative instances.
         */
        private final KMeans negatives;

        /**
         * The dimensionality of the resulting instances, including the class
         * attribute.
         */
        private final int dimensionality;

        /**
         * Constructs a new InstanceMappingFunction.
         * @param positives The clustering of positive instances.
         * @param negatives The clustering of negative instances.
         */
        public InstanceMappingFunction(KMeans positives, KMeans negatives) {
            this.positives = positives;
            this.negatives = negatives;
            this.dimensionality = this.positives.getNumClusters()
                    + this.negatives.getNumClusters()
                    + 1; // +1 for the class attribute.
        }

        /**
         * Gets the dimensionality of the resulting instances, including the
         * class attribute.
         * @return The dimensionality of the resulting instances.
         */
        public int getDimensionality() {
            return this.dimensionality;
        }

        @Override
        public Instance apply(Instance original) {
            double[] values = new double[this.dimensionality];
            double[] positiveDistances = this.positives.getDistances(original);
            double[] negativeDistances = this.negatives.getDistances(original);

            System.arraycopy(positiveDistances, 0, values, 0, positiveDistances.length);
            System.arraycopy(negativeDistances, 0, values, positiveDistances.length, negativeDistances.length);
            values[values.length - 1] = original.value(original.classAttribute());

            Instance result = new DenseInstance(values.length);
            for (int i = 0; i < values.length; i++) {
                result.setValue(i, values[i]);
            }

            return result;
        }

    }

}
