package nak.classify

import breeze.math.MutableLPVectorField
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import breeze.optimize.FirstOrderMinimizer.OptParams
import breeze.linalg._

/**
 * 
 * @author dlwh
 */
@RunWith(classOf[JUnitRunner])
class LogisticClassifierTest 
extends ClassifierTrainerTestHarness 
with ContinuousTestHarness {

  def trainer[L,T]:Classifier.Trainer[L,Counter[T,Double]] = {
    implicit val lps = MutableLPVectorField.make[Counter[T,Double],Double]
    new LogisticClassifier.Trainer[L,Counter[T,Double]](
      OptParams(tolerance=1E-2,regularization = 1.0))
  }
}
