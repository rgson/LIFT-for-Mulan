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
     * The distance function. Used for constructing the label-specific
     * features from cluster centroids.
     */
    private final DistanceFunction distanceFunction;

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
    private InstanceMappingFunction[] mappings;

    /**
     * The trained classifier for each label.
     */
    private Classifier[] classifiers;

    /**
     * The label-specific dataset for each label.
     */
    private Instances[] datasets;

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
        this.distanceFunction = new EuclideanDistance();
    }

    @Override
    protected void buildInternal(MultiLabelInstances multiLabelInstances) throws Exception {
        this.brTransformation = new BinaryRelevanceTransformation(multiLabelInstances);
        this.mappings = new InstanceMappingFunction[this.numLabels];
        this.classifiers = new Classifier[this.numLabels];
        this.datasets = new Instances[this.numLabels];
        Instances[] instancesByLabel = doBRTransformation(multiLabelInstances);

        for (int label = 0; label < this.numLabels; label++) {
            Instances instances = instancesByLabel[label];

            // Form P_k and N_k based on D according to Eq.(1);
            Instances posInstances = getPositiveInstances(instances);
            Instances negInstances = getNegativeInstances(instances);

            // Perform k-means clustering on P_k and N_k, each with m_k clusters
            // as defined in Eq.(2);
            int numClusters = Math.min(
                    (int) Math.ceil(this.r * posInstances.numInstances()),
                    (int) Math.ceil(this.r * negInstances.numInstances()));

            ClusterCentroids clusterCentroids;
            if (numClusters == 0) {
                // NOTE: The algorithm definition in [1] does not specify how the case of (m_k == 0) should be handled.
                // For this, the MATLAB implementation by Zhang [2] has been used as the reference implementation.
                // [2] http://cse.seu.edu.cn/PersonalPage/zhangml/files/LIFT.rar
                clusterCentroids = getClusterCentroids(instances, Math.min(50, instances.numInstances()));
            }
            else {
                ClusterCentroids posClusterCentroids = getClusterCentroids(posInstances, numClusters);
                ClusterCentroids negClusterCentroids = getClusterCentroids(negInstances, numClusters);
                clusterCentroids = combineClusterCentroids(posClusterCentroids, negClusterCentroids);
            }

            // Create the mapping phi_k for l_k according to Eq.(3);
            this.mappings[label] = new InstanceMappingFunction(clusterCentroids);

            // Form B_k according to Eq.(4);
            this.datasets[label] = createLabelSpecificDataset(instances, this.mappings[label]);

            // Induce g_k by invoking L on B_k, i.e. g_k <- L(B_k);
            this.classifiers[label] = AbstractClassifier.makeCopy(this.baseClassifier);
            this.classifiers[label].buildClassifier(this.datasets[label]);
        }
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        boolean[] bipartition = new boolean[numLabels];
        double[] confidences = new double[numLabels];

        for (int label = 0; label < numLabels; label++) {
            Instance transformedInstance = this.brTransformation.transformInstance(instance, label);
            transformedInstance = this.mappings[label].apply(transformedInstance);
            transformedInstance.setDataset(this.datasets[label]);

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
     * Constructs and gets the cluster centroids for a set of instances.
     *
     * @param instances The instances to cluster.
     * @param numClusters The number of clusters.
     * @return The cluster centroids.
     */
    private ClusterCentroids getClusterCentroids(Instances instances, int numClusters) throws Exception {
        if (instances.numInstances() == 1) {
            // No need to cluster if there's only a single instance.
            return new ClusterCentroids(removeClassAttribute(instances));
        }
        SimpleKMeans kmeans = new SimpleKMeans();
        kmeans.setDistanceFunction(this.distanceFunction);
        kmeans.setNumClusters(numClusters);
        kmeans.buildClusterer(removeClassAttribute(instances));
        return new ClusterCentroids(kmeans.getClusterCentroids());
    }

    /**
     * Combines two ClusterCentroid objects.
     *
     * @param centroids1 The first ClusterCentroids object.
     * @param centroids2 The second ClusterCentroids object.
     * @return The combined ClusterCentroids object with the centroids from both.
     */
    private ClusterCentroids combineClusterCentroids(ClusterCentroids centroids1, ClusterCentroids centroids2) {
        ClusterCentroids clusterCentroids = new ClusterCentroids(new Instances(centroids1.centroids));
        for (Instance centroid : centroids2.centroids) {
            // The centroids must be added individually in order for the dataset reference to be changed correctly.
            clusterCentroids.centroids.add(centroid);
        }
        return clusterCentroids;
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


    // =========================================================================
    // Inner classes
    // =========================================================================

    /**
     * Represents the centroids of a clustering.
     */
    private class ClusterCentroids {

        /**
         * The centroids.
         */
        private final Instances centroids;

        /**
         * Constructs a new ClusterCentroid object from a set of centroid instances.
         *
         * @param centroids The centroid instnaces.
         */
        public ClusterCentroids(Instances centroids) {
            this.centroids = centroids;
        }

        /**
         * Gets the number of centroids.
         *
         * @return The number of centroids.
         */
        public int getNumClusters() {
            return this.centroids.numInstances();
        }

        /**
         * Gets the distance from an instance to each of the cluster centroids.
         * @param instance The instance.
         * @return The distance from the instance to each of the centroids.
         */
        public double[] getDistances(Instance instance) {
            int numClusters = this.centroids.numInstances();
            DistanceFunction distanceFunction = LIFT.this.distanceFunction;
            Instance classless = LIFT.this.removeClassAttribute(instance);

            double[] distances = new double[numClusters];
            for (int i = 0; i < numClusters; i++) {
                distances[i] = distanceFunction.distance(classless, this.centroids.instance(i));
            }

            return distances;
        }

    }

    /**
     * A mapping function, transforming an instance from the original dataset to
     * an instance of label-specific features.
     */
    private final class InstanceMappingFunction implements Function<Instance, Instance> {

        /**
         * The clusters centroids.
         */
        private final ClusterCentroids centroids;

        /**
         * The dimensionality of the resulting instances, including the class
         * attribute.
         */
        private final int dimensionality;

        /**
         * Constructs a new InstanceMappingFunction.
         * @param centroids The cluster centroids.
         */
        public InstanceMappingFunction(ClusterCentroids centroids) {
            this.centroids = centroids;
            this.dimensionality = this.centroids.getNumClusters() + 1; // +1 for the class attribute.
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
            double[] distances = this.centroids.getDistances(original);

            System.arraycopy(distances, 0, values, 0, distances.length);
            values[values.length - 1] = original.value(original.classAttribute());

            Instance result = new DenseInstance(values.length);
            for (int i = 0; i < values.length; i++) {
                result.setValue(i, values[i]);
            }

            return result;
        }

    }

}
