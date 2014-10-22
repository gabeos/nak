package nak.classify

import nak.data.Example
import breeze.generic.UFunc.UImpl2
import nak.serialization.DataSerialization
import nak.serialization.DataSerialization._
import scala.collection.mutable
import breeze.storage.Zero
import breeze.linalg._
import breeze.collection.mutable.Beam

/**
 * kNearestNeighbor
 * 6/8/14
 * @author Gabriel Schubiner <gabeos@cs.washington.edu>
 *
 *
 */
class kNearestNeighbor[L, T, D](val examples: Iterable[Example[L, T]],
                                val k: Int = 1)(implicit dm: UImpl2[D, T, T, Double]) extends ParKNNClassifier[L, T] {


  def distance(a: T, b: T): Double = dm(a,b)
//
//  def testLOO(): Double = {
//    val indexedExamples = examples.zipWithIndex
//    indexedExamples.map({case (ex,i) =>
//      val beam = Beam[(L, Double)](k)(Ordering.by(-(_: (_, Double))._2))
//      beam ++= indexedExamples.
//               withFilter(_._2 != i).
//               map({ case (e,j) => (e.label, dm(e.features, ex.features))})
//      beam.groupBy(_._1).maxBy(_._2.size)._1 == ex.label
//    }).count(identity).toDouble / examples.size
//  }
//
//  /*
//   * Additional method to extract distances of k nearest neighbors
//   */
//  def distances(o: T): Iterable[DistanceResult] = {
//    val beam = Beam[DistanceResult](k)(distTuple3Ord)
//    beam ++= examples.map(e => (e.label,e.features, dm(e.features, o)))
//  }
//
//  /** For the observation, return the max voting label with prob = 1.0
//    */
//  override def scores(o: T): Counter[L, Double] = {
//    // Beam reverses ordering from min heap to max heap, but we want min heap
//    // since we are tracking distances, not scores.
//    val beam = Beam[(L, Double)](k)(Ordering.by(-(_: (_, Double))._2))
//
//    // Add all examples to beam, tracking label and distance from testing point
//    beam ++= examples.map(e => (e.label, dm(e.features, o)))
//
//    // Max voting classification rule
//    val predicted = beam.groupBy(_._1).maxBy(_._2.size)._1
//
//    // Degenerate discrete distribution with prob = 1.0 at predicted label
//    Counter((predicted, 1.0))
//  }
}

object kNearestNeighbor {

  class Trainer[L, T, D](k: Int = 1)(implicit dm: UImpl2[D, T, T, Double]) extends Classifier.Trainer[L, T] {
    type MyClassifier = kNearestNeighbor[L, T, D]

    override def train(data: Iterable[Example[L, T]]): MyClassifier = new kNearestNeighbor[L, T, D](data, k)
  }

  implicit def kNNReadWritable[L, T, D](implicit formatL: DataSerialization.ReadWritable[L], dm: UImpl2[D,T,T,Double],
                                        formatT: DataSerialization.ReadWritable[T]) =
    new ReadWritable[kNearestNeighbor[L, T, D]] {
      def read(source: Input): kNearestNeighbor[L, T, D] = {
        val k = source.readInt()
        val ex = DataSerialization.read[Iterable[Example[L,T]]](source)(DataSerialization.iterableReadWritable[Example[L,T]](Example.exReadWritable[L,T]))
        new kNearestNeighbor[L,T,D](ex,k)
      }

      def write(sink: Output, what: kNearestNeighbor[L, T, D]): Unit = {
        DataSerialization.write(sink,what.k)
        DataSerialization.write(sink,what.examples)
      }
    }


}
