# LIFT for Mulan

An implementation of the multi-label classification algorithm LIFT for use with Mulan.

Developed during the "PA2537 Research Methodology" course at BTH.

## Dependencies

* Java 8
* The Mulan framework (version 1.5)
* The Weka framework (version 3.7.10)
* *(Optional)* LibSVM (version 1.0.7)

## Usage

The `se.rgson.ml.lift.LIFT` class contains the LIFT implementation. It offers configurability of the internal binary classifier and the clustering ratio parameter.

By convention, LIFT is used with LibSVM. To use LibSVM with Weka, the LibSVM JAR must also be included. Refer to [the Weka wiki](https://weka.wikispaces.com/LibSVM) for details.
