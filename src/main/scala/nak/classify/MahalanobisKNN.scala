package nak.classify

import breeze.linalg.operators.OpMulMatrix
import breeze.math.MutableInnerProductModule
import nak.data.Example
import nak.serialization.DataSerialization
import nak.serialization.DataSerialization.{Input,Output,ReadWritable}

/**
 * ${PROJECT_NAME}
 * 10/21/14
 * @author Gabriel Schubiner <gabeos@cs.washington.edu>
 *
 *
 */
class MahalanobisKNN[L, T, M](val examples: Iterable[Example[L, T]], val k: Int, val projection: M)
                             (implicit vspace: MutableInnerProductModule[T, Double], opMulMT: OpMulMatrix.Impl2[M, T, T]) extends ParKNNClassifier[L, T] {
  import vspace._

  def distance(a: T,b: T): Double = {
    val tA = opMulMT(projection,a)
    val tB = opMulMT(projection,b)
    val diff = subVV(tA,tB)
    diff dot diff
  }
}

object MahalanobisKNN {

  implicit def mkNNReadWritable[L, T, M](implicit formatL: DataSerialization.ReadWritable[L], //formatE: DataSerialization.ReadWritable[Example[L,T]],
                                         formatT: DataSerialization.ReadWritable[T],
                                         formatM: DataSerialization.ReadWritable[M], man: Manifest[M], vspace: MutableInnerProductModule[T, Double],
                                         opMulMT: OpMulMatrix.Impl2[M, T, T]) =
    new ReadWritable[MahalanobisKNN[L, T, M]] {
      def read(source: Input): MahalanobisKNN[L, T, M] = {
        val k = source.readInt()
        val proj = DataSerialization.read[M](source)
        val ex = DataSerialization.read[Iterable[Example[L,T]]](source)(DataSerialization.iterableReadWritable[Example[L,T]](Example.exReadWritable[L,T]))
        new MahalanobisKNN(ex,k,proj)
      }

      def write(sink: Output, what: MahalanobisKNN[L, T, M]): Unit = {
        DataSerialization.write(sink,what.k)
        DataSerialization.write(sink,what.projection)
        DataSerialization.write(sink,what.examples)
      }
    }
}