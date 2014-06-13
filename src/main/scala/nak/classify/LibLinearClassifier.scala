package nak.classify

import breeze.linalg.{SparseVector, Vector, Counter}
import nak.data.Example
import nak.liblinear.{Feature => LiblinearFeature, _}
import breeze.util.{Encoder, Index}

/**
 * LibLinear wrapper
 * 6/12/14
 * @author Gabriel Schubiner <gabeos@cs.washington.edu>
 *
 *
 */
@SerialVersionUID(1L)
class LibLinearClassifier[L, T](liblinear: LibLinear,
                                inputEncoder: T => SparseVector[Double],
                                labelIndex: Index[L]) extends Classifier[L, T] {
  /** For the observation, return the score for each label that has a nonzero
    * score.
    */
  override def scores(o: T): Counter[L, Double] = {
    Encoder.fromIndex(labelIndex).decode(liblinear(inputEncoder(o)), true)
  }
}

object LibLinearClassifier {

  class SparseTrainer[L](config: LiblinearConfig = LiblinearConfig())
    extends Classifier.Trainer[L, SparseVector[Double]] {

    type MyClassifier = LibLinearClassifier[L, SparseVector[Double]]

    override def train(data: Iterable[Example[L, SparseVector[Double]]]) = {
      if (!config.showDebug) Linear.disableDebugOutput()

      val labels = Index[L]()
      data foreach {labels index _.label}


      var maxFeatureIndex = 0
      val (responses, observations) =
        data.toArray.map(ex =>
          (labels(ex.label).toDouble,
            ex.features.activeIterator.map({
              case (a, v) =>
                if (a > maxFeatureIndex) maxFeatureIndex = a
                new FeatureNode(a, v).asInstanceOf[LiblinearFeature]
            })
            .toArray)).unzip

      val param = new Parameter(config.solverType, config.cost, config.eps)
      val problem = LiblinearProblem(responses, observations, maxFeatureIndex + 1)
      new LibLinearClassifier(new LibLinear(Linear.train(problem, param), labels.size), identity[SparseVector[Double]],
        labels)
    }
  }

  class CounterTrainer[L, T](config: LiblinearConfig = LiblinearConfig())
    extends Classifier.Trainer[L, Counter[T, Double]] {

    type MyClassifier = LibLinearClassifier[L, Counter[T, Double]]

    override def train(data: Iterable[Example[L, Counter[T, Double]]]) = {
      if (!config.showDebug) Linear.disableDebugOutput()

      val labels = Index[L]()
      data foreach {labels index _.label}
      val featureIndex = Index[T]()
      for (d <- data; f <- d.features.keysIterator) featureIndex.index(f)
      val fEncoder = Encoder.fromIndex(featureIndex)

      val processedData = data.toArray.map {
        d =>
          labels(d.label).toDouble -> fEncoder.encodeSparse(d.features).activeIterator.toArray.sortBy(_._1).map({
            case (a, v) =>
              new FeatureNode(a+1, v).asInstanceOf[LiblinearFeature]
          }).toArray
      }

      val (responses, observations) = processedData.unzip

      val param = new Parameter(config.solverType, config.cost, config.eps)
      val problem = LiblinearProblem(responses, observations, featureIndex.size)
      new LibLinearClassifier(new LibLinear(Linear.train(problem, param), labels.size),
      {fEncoder.encodeSparse(_: Counter[T, Double], true)},
      labels)
    }
  }

}

/**
 * Configure the options for Liblinear training.
 */
case class LiblinearConfig(
                            solverType: SolverType = SolverType.L2R_LR,
                            cost: Double = 1.0,
                            eps: Double = 0.01,
                            showDebug: Boolean = false)

/**
 * Set up a problem to be solved.
 */
object LiblinearProblem {

  def apply(responses: Array[Double], observations: Array[Array[LiblinearFeature]], numFeats: Int) = {
    val problem = new Problem
    problem.y = responses
    problem.x = observations
    problem.l = responses.length
    problem.n = numFeats
    problem
  }

}

/**
 * An object to help with solver descriptions.
 */
object Solver {

  /**
   * The set of all valid solver types.
   */
  lazy val solverTypes = Set(
    "L2R_LR", "L2R_L2LOSS_SVC_DUAL", "L2R_L2LOSS_SVC", "L2R_L1LOSS_SVC_DUAL",
    "MCSVM_CS", "L1R_L2LOSS_SVC", "L1R_LR",
    "L2R_LR_DUAL", "L2R_L2LOSS_SVR", "L2R_L2LOSS_SVR_DUAL", "L2R_L1LOSS_SVR_DUAL")

  /**
   * Select the right solver given the textual description.
   */
  def apply(solverDescription: String) = solverDescription match {
    case "L2R_LR"              => SolverType.L2R_LR
    case "L2R_L2LOSS_SVC_DUAL" => SolverType.L2R_L2LOSS_SVC_DUAL
    case "L2R_L2LOSS_SVC"      => SolverType.L2R_L2LOSS_SVC
    case "L2R_L1LOSS_SVC_DUAL" => SolverType.L2R_L1LOSS_SVC_DUAL
    case "MCSVM_CS"            => SolverType.MCSVM_CS
    case "L1R_L2LOSS_SVC"      => SolverType.L1R_L2LOSS_SVC
    case "L1R_LR"              => SolverType.L1R_LR
    case "L2R_LR_DUAL"         => SolverType.L2R_LR_DUAL
    case "L2R_L2LOSS_SVR"      => SolverType.L2R_L2LOSS_SVR
    case "L2R_L2LOSS_SVR_DUAL" => SolverType.L2R_L2LOSS_SVR_DUAL
    case "L2R_L1LOSS_SVR_DUAL" => SolverType.L2R_L1LOSS_SVR_DUAL
    case invalid               => throw new MatchError("No solver with the name " + invalid)
  }

}

/**
 * Helper functions for working with Liblinear.
 */
object LiblinearUtil {


  /**
   * Convert tuples into Liblinear Features, basically.
   */
  def createLiblinearMatrix(observations: Seq[Seq[(Int, Double)]]): Array[Array[LiblinearFeature]] =
    observations.map {
      features =>
        features.map {case (a, v) => new FeatureNode(a, v).asInstanceOf[LiblinearFeature]}.toArray
    }.toArray

  /**
   * Convert tuples into Liblinear Features, basically.
   *
   * TODO: Condense so there is just one createLiblinearMatrix.
   */
  def createLiblinearMatrix(observations: Array[Array[(Int, Float)]]): Array[Array[LiblinearFeature]] =
    observations.map {
      features => {
        features
        .sortBy(_._1)
        .map {case (a, v) => new FeatureNode(a, v).asInstanceOf[LiblinearFeature]}
      }
    }

}