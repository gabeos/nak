package nak.classify

import breeze.linalg.Counter
import nak.classify.Classifier.Trainer

/**
 * nak
 * 10/27/14
 * @author Gabriel Schubiner <gabeos@cs.washington.edu>
 *
 *
 */
class LLClassifierTest extends ClassifierTrainerTestHarness with ContinuousTestHarness {
  def trainer[L, T]: Trainer[L, Counter[T, Double]] = new LLClassifier.LLTrainer[L,Counter[T,Double],T](5)
}
