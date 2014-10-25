package nak.classify

import breeze.collection.mutable.Beam
import breeze.linalg.Counter
import nak.data.Example
import scala.collection.parallel.ParIterable

/**
 * nak
 * 10/21/14
 * @author Gabriel Schubiner <gabeos@cs.washington.edu>
 *
 *
 */
trait kNNClassifier[L,T] extends Classifier[L,T] {

  // Iterable of (example, distance) tuples
  type DistanceResult = (L,T,Double)
  protected val distTuple3Ord = Ordering.by(-(_: DistanceResult)._3)
  protected val distTuple2Ord = Ordering.by(-(_: (L,Double))._2)

  def distance(a: T, b: T): Double

  def topK(o: T): Iterable[DistanceResult]

  def scoresFromTopK(topk: Iterable[DistanceResult]): Counter[L,Double]
  
  def distance(o: T,l: L): DistanceResult

  def leaveOneOutAccuracy(): Double
}

trait ParKNNClassifier[L,T] extends kNNClassifier[L,T] {

  val examples: Iterable[Example[L,T]]
  val k: Int
  val parEx: ParIterable[Example[L, T]] = examples.par

  private def distanceResult(e: Example[L,T],o: T): DistanceResult = (e.label,e.features, distance(e.features,o))

  def topK(o: T): Iterable[DistanceResult] = {
    val beam = Beam[(L,T, Double)](k)(distTuple3Ord)
    beam ++= parEx.map(distanceResult(_,o)).seq
  }

  // Counts each hit in top-k as 1/k, resulting in max voting rule for top-k classification
  def scoresFromTopK(topk: Iterable[DistanceResult]): Counter[L,Double] = Counter(topk.map(lfd => lfd._1 -> 1.0/k))

  def distance(o: T, l: L): DistanceResult = {
    val beam = Beam[(L,T,Double)](1)(distTuple3Ord)
    beam ++= parEx.filter(_.label == l).map(distanceResult(_,o)).seq
    beam.head
  }

  def leaveOneOutAccuracy(): Double = {
    val indexedExamples = examples.zipWithIndex
    indexedExamples.map({ case (ex, i) =>
      val beam = Beam[(L, Double)](k)(Ordering.by(-(_: (_, Double))._2))
      beam ++= indexedExamples.withFilter(_._2 != i).map({ case (e, j) => (e.label, distance(e.features,ex.features))})
      beam.groupBy(_._1).maxBy(_._2.size)._1 == ex.label
                        }).count(identity).toDouble / examples.size
  }

  def scores(o: T): Counter[L, Double] = scoresFromTopK(topK(o))
}




