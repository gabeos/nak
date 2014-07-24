package nak.classify
//
import breeze.linalg._
import nak.classify.Classifier.Trainer
import nak.classify.Initializers.CSCInitializers.ScaledDiagSparseInitializer
import nak.classify.Initializers.DenseInitializers.ScaledDiagDenseInitializer
import nak.classify.Initializers.GenericScaledDiagInitializer
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

/**
 * nak
 * 7/8/14
 * @author Gabriel Schubiner <gabeos@cs.washington.edu>
 *
 *
 */
@RunWith(classOf[JUnitRunner])
class NCAGenDenseClassifierTest
  extends DenseNearestNeighborTestHarness {
  var i = 0
  def trainer[L]: Trainer[L, DenseVector[Double]] = {
    import DenseMatrix.FrobeniusInnerProductDenseMatrixSpace._
    import breeze.math.MutableOptimizationSpace.DenseOptimizationSpace._
    new NCA.Trainer[L, DenseVector[Double], DenseMatrix[Double]]() with GenericScaledDiagInitializer[L,DenseVector[Double],DenseMatrix[Double]]
  }
}
