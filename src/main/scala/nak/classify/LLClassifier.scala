package nak.classify

import breeze.linalg.{dim, max, Counter}
import breeze.linalg.support.CanTraverseKeyValuePairs
import breeze.linalg.support.CanTraverseKeyValuePairs.KeyValuePairsVisitor
import breeze.util.Index
import nak.classify.Classifier.Trainer
import nak.data.Example
import nak.liblinear._

import scala.collection.mutable

/**
 * nak
 * 10/27/14
 * @author Gabriel Schubiner <gabeos@cs.washington.edu>
 *
 *
 */
class LLClassifier[L,T](model: Model,labelIndex: Index[L])(implicit featurize: T => Array[Feature]) extends Classifier[L,T] {
  /** For the observation, return the score for each label that has a nonzero
    * score. */
  def scores(o: T): Counter[L, Double] = {
    val probsOrVals = Array.ofDim[Double](model.getNrClass)
    if (model.isProbabilityModel) {
      Linear.predictProbability(model,o,probsOrVals)
    } else {
      Linear.predictValues(model,o,probsOrVals)
    }
    Counter(probsOrVals.zipWithIndex.map(di => (labelIndex.get(di._2),di._1)))
  }
}

object LLClassifier {

  class LLTrainer[L,T,K](numFeatures: Int, config: LiblinearConfig = LiblinearConfig())(implicit iter: CanTraverseKeyValuePairs[T,K,Double]) extends Trainer[L,T] {
    type MyClassifier = LLClassifier[L,T]

    if (config.showDebug) Linear.enableDebugOutput() else Linear.disableDebugOutput()
    val param = new Parameter(config.solverType, config.cost, config.eps)

    def train(data: Iterable[Example[L, T]]): MyClassifier = {
      val labelIndex = Index[L]()
      val (responses, observations) =
        ((mutable.ArrayBuilder.make[Double], mutable.ArrayBuilder.make[Array[Feature]]) /: data) { case ((li, fe), ex) =>
          (li += labelIndex.index(ex.label).toDouble, fe += iterateKVFeaturize(ex.features))
                                                                                                 }
      val problem = LiblinearProblem(responses.result(),observations.result(),numFeatures)
      val model = Linear.train(problem,param)
      new LLClassifier[L,T](model,labelIndex)
    }

    private val keyIndex = Index[K]()
    implicit def iterateKVFeaturize(vector: T)(implicit canIterKV: CanTraverseKeyValuePairs[T,K,Double]) = {
      val feats = mutable.ArrayBuilder.make[Feature]
      val kvVisitor = new KeyValuePairsVisitor[K,Double] {
        var i = 0
        def visit(k: K, a: Double): Unit = {
          feats += new FeatureNode(keyIndex.index(k)+1,a)
          i += 1
        }
        def zeros(numZero: Int, zeroKeys: Iterator[K], zeroValue: Double): Unit = Unit
      }

      canIterKV.traverse(vector,kvVisitor)
      feats.result().sortBy(_.getIndex)
    }
  }
}