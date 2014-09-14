
package nak.classify

import java.io.File

import breeze.collection.mutable.Beam
import breeze.linalg.operators.OpMulMatrix
import breeze.linalg.support.{CanTranspose, CanTraverseValues}
import breeze.linalg._
import breeze.math._
import breeze.optimize.FirstOrderMinimizer.OptParams
import breeze.optimize._
import breeze.util.Implicits._
import breeze.stats.distributions.{ThreadLocalRandomGenerator, RandBasis}
import com.typesafe.scalalogging.slf4j.LazyLogging
import nak.classify.Initializers._
import nak.data.Example
import nak.space.DMImplicits
import DMImplicits.decomposedMahalanobis
import nak.space.nca.NCAObjectives.Objectives.NCABatchObjective
import nak.space.nca.NCAObjectives._
import nak.space.nca.NCAObjectives.{Iso_CSC_SV, Iso_DM_DV}
import org.apache.commons.math3.random.MersenneTwister

import scala.reflect.ClassTag

/**
 * dialogue
 * 6/19/14
 * @author Gabriel Schubiner <gabeos@cs.washington.edu>
 *
 *
 */
class NCA[L, T, M](examples: Iterable[Example[L, T]], k: Int, A: M)(implicit vspace: MutableInnerProductModule[T, Double],
                                                                    opMulMT: OpMulMatrix.Impl2[M, T, T]) extends Classifier[L, T] {

  import vspace._

  // Iterable of (example, distance) tuples
  type DistanceResult = Iterable[(Example[L, T], Double)]

  val projection = A

  def testLOO(): Double = {
    val indexedExamples = examples.zipWithIndex
    indexedExamples.map({
      case (ex, i) =>
        val beam = Beam[(L, Double)](k)(Ordering.by(-(_: (_, Double))._2))
        beam ++= indexedExamples.
          withFilter(_._2 != i).
          map({ case (e, j) => (e.label, {
          val tF = opMulMT(A,e.features)
          val tO = opMulMT(A,ex.features)
          val diff = subVV(tF,tO)
          diff dot diff
          //      (A * e.features) - (A * o)
        })})
        beam.groupBy(_._1).maxBy(_._2.size)._1 == ex.label
    }).count(identity _).toDouble / examples.size
  }

  /*
   * Additional method to extract distances of k nearest neighbors
   */
  def distances(o: T): DistanceResult = {
    val beam = Beam[(Example[L, T], Double)](k)(Ordering.by(-(_: (_, Double))._2))
    beam ++= examples.par.map(e => (e, {
      val tF = opMulMT(A,e.features)
      val tO = opMulMT(A,o)
      val diff = subVV(tF,tO)
      diff dot diff
//      (A * e.features) - (A * o)
    })).seq
//    beam ++= examples.par.map(e => (e, decomposedMahalanobis(e.features, o, A))).seq
  }

  /** For the observation, return the max voting label with prob = 1.0
    */
  override def scores(o: T): Counter[L, Double] = {
    // Beam reverses ordering from min heap to max heap, but we want min heap
    // since we are tracking distances, not scores.
    val beam = Beam[(L, Double)](k)(Ordering.by(-(_: (_, Double))._2))

    // Add all examples to beam, tracking label and distance from testing point
    beam ++= examples.map(e => (e.label, {
      val tF = opMulMT(A,e.features)
      val tO = opMulMT(A,o)
      val diff = subVV(tF,tO)
      diff dot diff
      //      (A * e.features) - (A * o)
    }))

    // Max voting classification rule
    val predicted = beam.groupBy(_._1).maxBy(_._2.size)._1

    // Degenerate discrete distribution with prob = 1.0 at predicted label
    Counter((predicted, 1.0))
  }

}

object NCA {

  class Trainer[L, T, M](opt: NCAOptParams = NCAOptParams())
                        (implicit optspace: MutableOptimizationSpace[M, T, Double],
                         mspace: MutableFiniteCoordinateField[M, (Int, Int), Double],
                         canDiag: diag.Impl[T, M])
    extends Classifier.Trainer[L, T] with LazyLogging {
    self: Initializer[L, T, M] =>
    type MyClassifier = NCA[L, T, M]

    logger.info(s"Initializing NCA Trainer with OptParams: $opt")

    def train(data: Iterable[Example[L, T]]): MyClassifier = {
      logger.info(s"Training NCA-kNN classifier with ${data.size} examples.")

      logger.info(s"Initializing NCA Transformation Matrix.")
      val initial: M = if (opt.dimension > 0) init(data, opt.dimension) else init(data)
      logger.debug(s"Initial value: \n$initial")

      logger.info(s"Initializing Batch Objective")
      val df = new Objectives.NCABatchObjective[L, T, M](data,opt.gradientLogDir)
      logger.info(s"Optimizing NCA Matrix.")
      val A = opt.minimize(df,initial)

      import optspace.mulMVV
      new NCA[L, T, M](data, opt.K, A)
    }
  }

  case class NCAOptParams(K: Int = 1,
                          dimension: Int = -1,
                          regularization: Double = 0.0,
                          alpha: Double = 0.5,
                          maxIterations: Int = 1000,
                          useL1: Boolean = false,
                          tolerance: Double = 1E-5,
                          useStochasticBatches: Boolean = false,
                          useScanningBatches: Boolean = false,
                          batchSize: Int = 512,
                          useStochasticTruncatedSums: Boolean = false,
                          useScanningTruncatedSums: Boolean = false,
                          truncatedSumSize: Int = 512,
                          randomSeed: Int = 0,
                          gradientLogDir: Option[File] = None) {
    private implicit val random = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(randomSeed)))
    gradientLogDir.foreach(f => require(f.isDirectory && f.canWrite, "File specified must be writable directory"))

    def minimize[T](f: NCABatchObjective[_,_,T], init: T)(implicit space: MutableFiniteCoordinateField[T, _, Double]): T = {
      this.iterations(f, init).last.x
    }

    def iterations[T](f: NCABatchObjective[_, _, T], init: T)(implicit
                                                              space: MutableFiniteCoordinateField[T, _, Double]): Iterator[FirstOrderMinimizer[T, BatchDiffFunction[T]]#State] = {
      val it = (useStochasticBatches, useScanningBatches, useStochasticTruncatedSums, useScanningTruncatedSums) match {
        case (true, false, false, false)  => this.iterations(f.withRandomBatches(batchSize), init)(space)
        case (false, true, false, false)  => this.iterations(f.withScanningBatches(batchSize), init)(space)
        case (false, false, true, false)  => this.iterations(f.withRandomTruncatedSums(truncatedSumSize), init)(space)
        case (false, false, false, true)  => this.iterations(f.withScanningTruncatedSums(truncatedSumSize), init)(space)
        case (true, false, true, false)   => this.iterations(f.withRandomBatchAndSum(batchSize, truncatedSumSize), init)(space)
        case (false, true, true, false)   => this.iterations(f.withScanningBatchAndRandomSum(batchSize, truncatedSumSize),init)(space)
        case (false, false, false, false) => this.iterations(f: DiffFunction[T], init)(space)
        case _                            => throw new UnsupportedOperationException(
          "stochastic/scanning batch parameters aren't supported")
      }

      it.asInstanceOf[Iterator[FirstOrderMinimizer[T, BatchDiffFunction[T]]#State]]
    }

    def iterations[T](f: StochasticDiffFunction[T], init:T)(implicit space: MutableFiniteCoordinateField[T, _, Double]):Iterator[FirstOrderMinimizer[T, StochasticDiffFunction[T]]#State] = {
      val r = if(useL1) {
        new AdaptiveGradientDescent.L1Regularization[T](regularization, eta=alpha, maxIter = maxIterations)(space, random)
      } else { // L2
        new AdaptiveGradientDescent.L2Regularization[T](regularization, alpha,  maxIterations)(space, random)
      }
      r.iterations(f,init)
    }

    def iterations[T](f: DiffFunction[T], init:T)(implicit space: MutableCoordinateField[T, Double]): Iterator[LBFGS[T]#State] = {
      if(useL1) new OWLQN[T](maxIterations, 5, regularization, tolerance)(space).iterations(f,init)
      else (new LBFGS[T](maxIterations, 5, tolerance=tolerance)(space)).iterations(DiffFunction.withL2Regularization(f,regularization),init)
    }
  }

  class DenseTrainer[L](opt: OptParams = OptParams(), K: Int = 1)
                       (implicit vspace: MutableInnerProductModule[DenseVector[Double], Double],
                        canTraverse: CanTraverseValues[DenseVector[Double], Double],
                        man: ClassTag[DenseVector[Double]]) extends Classifier.Trainer[L, DenseVector[Double]] with LazyLogging {
    self: DenseInitializer[L, DenseMatrix[Double]] =>

    override type MyClassifier = NCA[L, DenseVector[Double], DenseMatrix[Double]]

    override def train(data: Iterable[Example[L, DenseVector[Double]]]): MyClassifier = {
      logger.debug(s"Training NCA-kNN classifier with ${data.size} examples.")

      logger.debug(s"Initializing NCA Transformation Matrix.")
      val initial: DenseMatrix[Double] = init(data)

      logger.debug(s"Initializing Batch Objective")
      val df = new DenseObjectives.NCABatchObjective[L](data)

      implicit val mvIso = new Iso_DM_DV(initial.rows, initial.cols)

      logger.debug(s"Optimizing NCA Matrix.")
      val A: DenseMatrix[Double] = mvIso.backward(opt.minimize(df.throughLens[DenseVector[Double]], mvIso.forward(initial)))

      new NCA[L, DenseVector[Double], DenseMatrix[Double]](data, K, A)
    }
  }

  class SparseTrainer[L](opt: OptParams = OptParams(), K: Int = 1)
                        (implicit vspace: MutableInnerProductModule[SparseVector[Double], Double],
                         canTraverse: CanTraverseValues[SparseVector[Double], Double],
                         man: ClassTag[SparseVector[Double]]) extends Classifier.Trainer[L, SparseVector[Double]] {
    self: CSCInitializer[L, CSCMatrix[Double]] =>

    override type MyClassifier = NCA[L, SparseVector[Double], CSCMatrix[Double]]

    override def train(data: Iterable[Example[L, SparseVector[Double]]]): MyClassifier = {

      val initial: CSCMatrix[Double] = init(data)

      val df = new SparseObjectives.NCASparseBatchObjective[L](data)

      implicit val mvIso = new Iso_CSC_SV(initial.rows, initial.cols)

      val A: CSCMatrix[Double] = mvIso.backward(opt.minimize(df.throughLens[SparseVector[Double]], mvIso.forward(initial)))

      new NCA[L, SparseVector[Double], CSCMatrix[Double]](data, K, A)
    }
  }

}
