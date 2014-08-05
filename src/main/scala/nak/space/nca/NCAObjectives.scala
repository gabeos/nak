package nak.space.nca

import breeze.linalg._
import breeze.linalg.operators.OpMulMatrix
import breeze.linalg.support.CanTranspose
import breeze.math._
import breeze.numerics._
import breeze.optimize.{BatchDiffFunction, StochasticDiffFunction}
import breeze.util.Isomorphism
import com.typesafe.scalalogging.slf4j.LazyLogging
import nak.data.Example

import scala.reflect.ClassTag

/**
 * nak
 * 6/27/14
 * @author Gabriel Schubiner <gabeos@cs.washington.edu>
 *
 *         Different styles of objective functions for NCA
 */
object NCAObjectives extends LazyLogging {

  class Iso_DM_DV(r: Int, c: Int) extends Isomorphism[DenseMatrix[Double], DenseVector[Double]] {
    override def forward(t: DenseMatrix[Double]): DenseVector[Double] = t.flatten()

    override def backward(u: DenseVector[Double]): DenseMatrix[Double] = u.asDenseMatrix.reshape(r, c)
  }

  class Iso_CSC_SV(r: Int, c: Int) extends Isomorphism[CSCMatrix[Double], SparseVector[Double]] {

    override def forward(t: CSCMatrix[Double]): SparseVector[Double] = t.flatten()

    override def backward(u: SparseVector[Double]): CSCMatrix[Double] = reshape(u, r, c)
  }

  //  class Iso_M_V[M,V](r: Int, c: Int)(implicit vView: V <:< Vector[Double], mView: M <:< Matrix[Double]) extends Isomorphism[M,V] {
  //    override def forward(t: M): Vector[Double] = t.flatten()
  //
  //    override def backward(u: V): M = reshape(u,r,c)
  //  }

  object Objectives {


    class NCABatchObjective[L, T, M](data: Iterable[Example[L, T]])(implicit optspace: MutableOptimizationSpace[M,T,Double]
//                                                                    opMulMV: OpMulMatrix.Impl2[M, T, T]
//                                                                    opMulVTV: OpMulMatrix.Impl2[T, TT, M]
//                                                                    opMulMM: OpMulMatrix.Impl2[M, M, M]
      ) extends BatchDiffFunction[M] {
      import optspace._
      val size = data.size
      val featureSize = dim(data.head.features)
      val iData = data.map(_.features).toIndexedSeq
      val iLabel = data.map(_.label).toIndexedSeq

      private def eNSqProjNorm(v1: T, v2: T, proj: M): Double = {
        exp(-pow(norm(proj * (v1 - v2)), 2))
      }

      private def timeToSecs(current: Long, last: Long): Double = {
        (current - last) / 1000.0
      }

      override def calculate(A: M, batch: IndexedSeq[Int]): (Double, M) = {
        logger.debug(s"Calculating objective gradient and value for batch indices:\n$batch")
        // shortcut to access indexed data through batch indices
        val batchData = iData.compose(batch.apply)
        val batchLabel = iLabel.compose(batch.apply)
        val rBatchInd = batch.zipWithIndex.toMap
        val parBatch = batch.toArray.par
        var time = System.currentTimeMillis()

        logger.debug(s"Calculating cache...")
        val ensqpnCache = parBatch.map(i => parBatch.map(k => if (i == k) 0.0 else eNSqProjNorm(iData(i),iData(k),A)))
        var ctime = System.currentTimeMillis()
        logger.debug(s"Calculated ensqpnCache in ${timeToSecs(ctime,time)}")
        time = ctime

//        logger.debug(s"Calculating smNorms...")
        val smNorms = ensqpnCache.map(_.sum)
        ctime = System.currentTimeMillis()
        logger.debug(s"Calculated smNorms in ${timeToSecs(ctime,time)}")
        time = ctime
//        logger.debug(s"smNorms: ${smNorms.toSeq.seq}")

//        logger.debug(s"Calculating sMaxes...")
        val smaxes = ensqpnCache.zip(smNorms).map({case (i,n) => i.map(ik => ik / n)})
        ctime = System.currentTimeMillis()
        logger.debug(s"Calculated smaxes in ${timeToSecs(ctime,time)}")
        time = ctime
//        logger.debug(s"smaxes: ${smaxes(0)}")
          //Array.tabulate[Double](batch.size, batch.size)((i, k) => ensqpnCache(i)(k) / smNorms(i))
        //        val smaxes = Array.tabulate[Double](batch.size, batch.size)((i, k) => {
        //          if (i == k) 0.0
        //          else eNSqProjNorm(batchData(i), batchData(k), A) / smNorms(i)
        //        })
        def term(i: Int, j: Int): M = {
          val dift = iData(i) - iData(j)
          (dift * dift.t) :* smaxes(rBatchInd(i))(rBatchInd(j))
        }

        def nterm(i: Int, j: Int, n: Double): M = {
          val dift = iData(i) - iData(j)
          (dift * dift.t) :* smaxes(rBatchInd(i))(rBatchInd(j))
        }

        def bterm(i: Int, j: Int): M = {
          val dift = batchData(i) - batchData(j)
          val dtt: M = dift * dift.t
          val smij: Double = smaxes(i)(j)
          val res = dtt :* smij
          res
        }

//        logger.debug(s"Evaluating terms...")
//        val terms = parBatch.zip(smaxes).map({ case (i,ism) =>
//          parBatch.zip(ism).map({ case (k,iksm) => nterm(i,k,iksm)})})
//        logger.debug(s"Max_Min_Term_Norms: ${val norms = terms.map(_.map(norm(_))); (norms.map(_.min).min,norms.map(_.max).max)}")
//        ctime = System.currentTimeMillis()
//        logger.debug(s"Calculated terms in ${timeToSecs(ctime,time)}")
//        time = ctime


//        logger.debug(s"Summing terms...")

//        val firstTerms = terms.map(trm => trm.reduce(_ + _))
//        ctime = System.currentTimeMillis()
//        logger.debug(s"Calculated first terms in ${timeToSecs(ctime, time)}")
//        time = ctime
//        val equalLabelFilter = parBatch.map(i => parBatch.map(j => iLabel(i) == iLabel(j)))
//        ctime = System.currentTimeMillis()
//        logger.debug(s"Calculated label filter in ${timeToSecs(ctime, time)}")
//        time = ctime
//        val secondTerms = terms.zip(equalLabelFilter).map({case (tL,bL) => tL.zip(bL).withFilter(_._2).map(_._1).reduce(_ + _)})
//        ctime = System.currentTimeMillis()
//        logger.debug(s"Calculated second terms in ${timeToSecs(ctime, time)}")
//        time = ctime
//        val p_i = smaxes.zip(equalLabelFilter).map({case (vL,bL) => vL.zip(bL).withFilter(_._2).map(_._1).reduce(_ + _)})
//        ctime = System.currentTimeMillis()
//        logger.debug(s"Calculated p_i in ${timeToSecs(ctime, time)}")
//        time = ctime
//        val valuep = p_i.reduce(_ + _)
//        ctime = System.currentTimeMillis()
//        logger.debug(s"Calculated value in ${timeToSecs(ctime, time)}")
//        time = ctime
//        val gradRHSP = p_i.zip(firstTerms).map({case (d,m) => m :* d}).zip(secondTerms).map({case (f,s) => f - s}).reduce(_ + _)
//        ctime = System.currentTimeMillis()
//        logger.debug(s"Calculated gradRHS in ${timeToSecs(ctime, time)}")
//        time = ctime

        val (probs, elGrads) = (0 until batch.size).par.map(i => {
          val thisTime = System.currentTimeMillis()
          val f = zeroLikeM(A)
          val s = zeroLikeM(A)
          var p_ind = 0.0
          var j = 0
          while (j < batch.size) {
            val bt = bterm(i,j)
            f += bt
            if (batchLabel(i) == batchLabel(j)) {
              s += bt
              p_ind += smaxes(i)(j)
            }
            j += 1
          }
          f :*= p_ind
          f -= s
          logger.debug(s"Completed iteration [$i / ${batch.size}] in ${timeToSecs(System.currentTimeMillis(),thisTime)}")
          (p_ind,f)
        }).unzip
        ctime = System.currentTimeMillis()
        logger.debug(s"Computed terms in ${timeToSecs(ctime,time)}")
        time = ctime

        val value = probs.sum
        ctime = System.currentTimeMillis()
        logger.debug(s"Computed value in ${timeToSecs(ctime,time)}")
        time = ctime
        val gradRHS = elGrads.reduce(_ + _)
        ctime = System.currentTimeMillis()
        logger.debug(s"Computed grad in ${timeToSecs(ctime,time)}")
        time = ctime

        //        var value = 0.0
//        val gradRHS = zeroLikeM(A)
//        var i = 0
//        while (i < batch.size) {
//          if (i % (batch.size / 10).toInt == 0)
//            logger.debug(s"Calculating for point: [$i / ${batch.size}]")
//          var p_ind = 0.0
//          val f = zeroLikeM(A)
//          val s = zeroLikeM(A)
//
//          var j = 0
//          while (j < batch.size) {
////            logger.debug(s"Summing point: [$j / ${batch.size}")
//            val kTerm = bterm(i, j)
//            f += kTerm
//            if (batchLabel(i) == batchLabel(j)) {
//              s += kTerm
//              p_ind += smaxes(i)(j)
//            }
//            j += 1
//          }
//          value += p_ind
//          f :*= p_ind
//          f -= s
//          gradRHS += f
//          i += 1
//        }
        val nA = (A * gradRHS) :* (-2.0)
        logger.debug(s"Gradient (numActive): ${nA.activeValuesIterator.length}")
        logger.debug(s"Done! Val = $value, grad: ${norm(nA)}")
        (-value, nA)
      }

      override def fullRange: IndexedSeq[Int] = 0 until size

    }
  }

  object DenseObjectives {

    private def eNSqProjNorm(v1: DenseVector[Double], v2: DenseVector[Double], proj: DenseMatrix[Double]) =
      exp(-pow(norm((proj * v1) - (proj * v2)), 2))

    class NCAStochasticOnlineObjective[L](data: Iterable[Example[L, DenseVector[Double]]]) extends StochasticDiffFunction[DenseMatrix[Double]] {
      val size = data.size
      val featureSize = data.head.features.length
      val iData = data.map(_.features).toIndexedSeq
      val iLabel = data.map(_.label).toIndexedSeq
      var iter = 0

      override def calculate(A: DenseMatrix[Double]): (Double, DenseMatrix[Double]) = {

        val i = iter % size
        iter += 1

        val smNorm = (0 until size).withFilter(_ != i).map(k => eNSqProjNorm(iData(i), iData(k), A)).sum

        val smax = DenseVector.tabulate[Double](size)(k => {
          if (k == i) 0.0
          else eNSqProjNorm(iData(i), iData(k), A) / smNorm
        })

        // cache p_i
        val p_i =
          (0 until size).
            withFilter(j => iLabel(j) == iLabel(i)).
            map(j => smax(j)).sum

        //Expected number of points correctly classified, negated for minimization
        val value: Double = -p_i

        // gradient, negated for minimization
        val grad: DenseMatrix[Double] = {

          def term(j: Int) = {
            val diff = iData(i) - iData(j)
            diff * diff.t * smax(j)
          }

          val (first, second) = (0 until size).foldLeft(
            (DenseMatrix.zeros[Double](featureSize, featureSize),
              DenseMatrix.zeros[Double](featureSize, featureSize)))({
            case ((f, s), k) =>
              val kTerm = term(k)
              (f :+ kTerm, if (iLabel(k) == iLabel(i)) s :+ kTerm else s)
          })
          (A * -2.0) * ((first :* p_i) - second) //(0 until size).map(i => (first(i) - second(i)) * p_i(i)).reduce(_ + _)
        }

        (value, grad)
      }
    }

    class NCABatchObjective[L](data: Iterable[Example[L, DenseVector[Double]]]) extends BatchDiffFunction[DenseMatrix[Double]] {
      val size = data.size
      val featureSize = data.head.features.length
      val iData = data.map(_.features).toIndexedSeq
      val iLabel = data.map(_.label).toIndexedSeq

      override def calculate(A: DenseMatrix[Double], batch: IndexedSeq[Int]): (Double, DenseMatrix[Double]) = {

        // shortcut to access indexed data through batch indices
        val batchData = iData.compose(batch.apply)

        val smNorms =
          batch.map(i =>
            (0 until size).withFilter(_ != i).map(k => eNSqProjNorm(iData(i), iData(k), A)).sum)

        val smaxes = Array.tabulate[Double](batch.size, batch.size)((i, k) => {
          if (i == k) 0.0
          else eNSqProjNorm(batchData(i), batchData(k), A) / smNorms(i)
        })

        def term(i: Int, j: Int) = {
          val diff = iData(i) - iData(j)
          val ddt = diff * diff.t
          diff * diff.t * smaxes(i)(j)
        }

        def scaleNegate(gv: (Double, DenseMatrix[Double])) = (-gv._1, (A * gv._2) * -2.0)

        scaleNegate(batch.foldLeft((0.0, DenseMatrix.zeros[Double](featureSize, featureSize)))({
          case ((vAgg, gAgg), i) => {
            val (first, second, p_i) = (0 until size).foldLeft(
              (DenseMatrix.zeros[Double](featureSize, featureSize),
                DenseMatrix.zeros[Double](featureSize, featureSize), 0.0))({
              case ((f, s, p), j) =>
                val kTerm = term(i, j)
                if (iLabel(j) == iLabel(i))
                  (f :+ kTerm, s :+ kTerm, p + smaxes(i)(j))
                else (f :+ kTerm, s, p)
            })
            (vAgg + p_i, gAgg + ((first :* p_i) - second))
          }
        }))
      }

      override def fullRange: IndexedSeq[Int] = 0 until size
    }

  }

  object SparseObjectives {
    private def eNSqProjNorm(v1: SparseVector[Double], v2: SparseVector[Double], proj: CSCMatrix[Double]) =
      exp(-pow(norm((proj * v1) - (proj * v2)), 2))

    class NCASparseBatchObjective[L](data: Iterable[Example[L, SparseVector[Double]]]) extends BatchDiffFunction[CSCMatrix[Double]] {
      val size = data.size
      val featureSize = data.head.features.length
      val iData = data.map(_.features).toIndexedSeq
      val iLabel = data.map(_.label).toIndexedSeq

      override def calculate(A: CSCMatrix[Double], batch: IndexedSeq[Int]): (Double, CSCMatrix[Double]) = {

        // shortcut to access indexed data through batch indices
        val batchData = iData.compose(batch.apply)

        val smNorms =
          batch.map(i =>
            (0 until size).withFilter(_ != i).map(k => eNSqProjNorm(iData(i), iData(k), A)).sum)

        val smaxes = Array.tabulate[Double](batch.size, batch.size)((i, k) => {
          if (i == k) 0.0
          else eNSqProjNorm(batchData(i), batchData(k), A) / smNorms(i)
        })

        def term(i: Int, j: Int) = {
          val diff = iData(i) - iData(j)
          (diff.asCSCMatrix().t * diff.t) * smaxes(i)(j)
        }

        def scaleNegate(gv: (Double, CSCMatrix[Double])) = (-gv._1, (A * gv._2) * -2.0)

        scaleNegate(batch.foldLeft((0.0, CSCMatrix.zeros[Double](featureSize, featureSize)))({
          case ((vAgg, gAgg), i) => {
            val (first, second, p_i) = (0 until size).foldLeft(
              (CSCMatrix.zeros[Double](featureSize, featureSize),
                CSCMatrix.zeros[Double](featureSize, featureSize), 0.0))({
              case ((f, s, p), j) =>
                val kTerm = term(i, j)
                if (iLabel(j) == iLabel(i))
                  (f :+ kTerm, s :+ kTerm, p + smaxes(i)(j))
                else (f :+ kTerm, s, p)
            })
            (vAgg + p_i, gAgg + ((first :* p_i) - second))
          }
        }))
      }

      override def fullRange: IndexedSeq[Int] = 0 until size
    }

  }

}
