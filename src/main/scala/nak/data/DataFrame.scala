package nak.data

import breeze.generic.UFunc
import breeze.linalg._
import breeze.linalg.operators.{OpMulMatrix, OpMulInner, OpNeg, OpType}
import breeze.linalg.support.CanTraverseValues.ValuesVisitor
import breeze.linalg.support._
import breeze.math.{MutableCoordinateField, Semiring, Field, MutableFiniteCoordinateField}
import breeze.storage.Zero
import breeze.util._

import scala.reflect.ClassTag

/**
 * nak
 * 8/19/14
 * @author Gabriel Schubiner <gabeos@cs.washington.edu>
 *
 *
 */

// TODO:: Figure out TF mess, i.e. how to ensure feature vectors are vectors, how to make DataFrame from FVs
// TODO:: :: Possibly subclass DataFrame for this specific usage (replace LFMatrix)
// TODO:: :: Add implicits to create specific type of underlying matrix needed, e.g. SV -> CSC, DV -> DM
// TODO:: Fix LogisticClassifier.Trainer as test for DataFrame/DF subclass
// TODO:: Figure out where OptimizationSpace is needed, or if it is needed at all.
// TODO:: :: In all classifiers, to tie V type to M type? Or never, use implicits mentioned above
// TODO:: Update LinearClassifier to use DataFrame subclass naturally, and make sure appropriate Multiplication Ops work and get in scope
// TODO:: Add read/write/etc methods (later)
class DataFrame[RL, CL, M, @specialized(Int, Float, Double, Long) S](protected[data] val underlying: M,
                                                                     private var _rowIndex: Index[RL],
                                                                     private var _colIndex: Index[CL])
                                                                    (implicit man: ClassTag[S],
                                                                     zero: Zero[S],
                                                                     matEv: M <:< Matrix[S],
                                                                     space: MutableFiniteCoordinateField[M, (Int, Int), S])
  extends Matrix[S] with MatrixLike[S, DataFrame[RL,CL,M,S]] {

  def repr: DataFrame[RL, CL, M, S] = this

  def apply(i: Int, j: Int): S = underlying(i,j)
  def update(i: Int, j: Int, e: S): Unit = underlying.update(i,j,e)

  def rows: Int = underlying.rows
  def cols: Int = underlying.cols

  def rowIndex = _rowIndex
  def colIndex = _colIndex

  def empty = new DataFrame[RL,CL,M,S](space.zeroLike(underlying),rowIndex,colIndex)

  def reindex(rowInd: Index[RL], colInd: Index[CL]) = {
    _rowIndex = rowInd
    _colIndex = colInd
  }

  def copy: DataFrame[RL,CL,M,S] = new DataFrame[RL,CL,M,S](underlying.copy.asInstanceOf[M],rowIndex,colIndex)
  def flatten(view: View = View.Prefer): Vector[S] = underlying.flatten(view)

  def activeSize: Int = underlying.activeSize
  def activeIterator: Iterator[((Int, Int), S)] = underlying.activeIterator
  def activeKeysIterator: Iterator[(Int, Int)] = underlying.activeKeysIterator
  def activeValuesIterator: Iterator[S] = underlying.activeValuesIterator


  // TODO: Include column labels in colWidth calculation
  // TODO: Account for v. large matrices / SparseMatrices?
  // TODO: Add _ and |
  // TODO: Truncate long doubles?
  // TODO: prettify?
  override def toString(maxLines: Int = Terminal.terminalHeight-3,
                        maxWidth: Int = Terminal.terminalWidth): String = {
    val showRows = if (rows > maxLines) maxLines - 1 else rows
    val rowLabelBuffer = 2
    def colWidth(col : Int) =
      if (showRows > 0) (0 until showRows).map(row => this(row,col).toString.length+2).max else 0

    val sortedRowLabels = rowIndex.pairs.toIndexedSeq.sortBy(_._2).map(_._1.toString)
    val rowLabelWidth = sortedRowLabels.map(_.length).max

    val colWidths = new scala.collection.mutable.ArrayBuffer[Int]
    var col = 0
    while (col < cols && colWidths.sum < maxWidth) {
      colWidths += colWidth(col)
      col += 1
    }

    // make space for "... (K total)"
    if (colWidths.size < cols) {
      while (colWidths.sum + cols.toString.length + 12 >= maxWidth) {
        if (colWidths.isEmpty) {
          return "%d x %d matrix".format(rows, cols)
        }
        colWidths.remove(colWidths.length - 1)
      }
    }

    val newline = Terminal.newline

    val rv = new scala.StringBuilder
    rv.append(" " * (rowLabelWidth + rowLabelBuffer))
    colIndex.pairs.toIndexedSeq.sortBy(_._2).take(colWidths.length).foreach(t => {
      val colLab = t._1.toString
      rv.append(colLab)
      rv.append(" " * (colWidths(t._2) - colLab.length))
    })
    rv.append(newline)
    for (row <- 0 until showRows; col <- 0 until colWidths.length) {
      if (col == 0) {
        rv.append(sortedRowLabels(row))
        rv.append(" " * ((rowLabelWidth - sortedRowLabels(row).length) + rowLabelBuffer))
      }
      val cell = this(row,col).toString
      rv.append(cell)
      rv.append(" " * (colWidths(col) - cell.length))
      if (col == colWidths.length - 1) {
        if (col < cols - 1) {
          rv.append("...")
          if (row == 0) {
            rv.append(" (")
            rv.append(cols)
            rv.append(" total)")
          }
        }
        if (row + 1 < showRows) {
          rv.append(newline)
        }
      }
    }

    if (rows > showRows) {
      rv.append(newline)
      rv.append("... (")
      rv.append(rows)
      rv.append(" total)")
    }

    rv.toString
  }
}

object DataFrame {

  def apply[M, S](underlying: M)(implicit man: ClassTag[S],
                                 zero: Zero[S],
                                 matEv: M <:< Matrix[S],
                                 space: MutableFiniteCoordinateField[M, (Int, Int), S]) =
    new DataFrame[Int, Int, M, S](underlying, new DenseIntIndex(underlying.rows), new DenseIntIndex(underlying.cols))

  def LFFrame[RL,M,S](underlying: M, labelIndex: Index[RL])(implicit man: ClassTag[S],
                                                            zero: Zero[S],
                                                            matEv: M <:< Matrix[S],
                                                            space: MutableFiniteCoordinateField[M, (Int, Int), S]) =
    new DataFrame[RL,Int,M,S](underlying,labelIndex,new DenseIntIndex(underlying.cols))

  def LFFrame[L,M,V,S](featureSize: Int,labelIndex: Index[L])
                      (implicit man: ClassTag[S],zero: Zero[S],mulMat: OpMulMatrix.Impl2[V,V,M],
                       evMat: M <:< Matrix[S],mspace: MutableFiniteCoordinateField[M,(Int,Int),S],zeros: CanCreateZeros[V,Int]) =
    LFFrame[L,M,S](mulMat(zeros(featureSize),zeros(featureSize)),labelIndex)

  implicit def liftOpVSV[Op <: OpType,RL,CL,M,S]
  (implicit op: UFunc.UImpl2[Op,M,S,M], man: ClassTag[S], zero: Zero[S], field: Field[S],
   matEv: M <:< Matrix[S], uspace: MutableFiniteCoordinateField[M,(Int,Int),S]) =
    new UFunc.UImpl2[Op,DataFrame[RL,CL,M,S],S,DataFrame[RL,CL,M,S]] {
      def apply(v: DataFrame[RL, CL, M, S], v2: S): DataFrame[RL, CL, M, S] =
        new DataFrame[RL,CL,M,S](op(v.underlying,v2),v.rowIndex,v.colIndex)
    }

  implicit def liftOpVVV[Op <: OpType, RL, CL, M, S]
  (implicit op: UFunc.UImpl2[Op,M,M,M], man: ClassTag[S], zero: Zero[S], field: Field[S],
   matEv: M <:< Matrix[S], uspace: MutableFiniteCoordinateField[M,(Int,Int),S]) =
    new UFunc.UImpl2[Op,DataFrame[RL,CL,M,S],DataFrame[RL,CL,M,S],DataFrame[RL,CL,M,S]] {
      def apply(v: DataFrame[RL, CL, M, S], v2: DataFrame[RL, CL, M, S]): DataFrame[RL, CL, M, S] =
        new DataFrame[RL,CL,M,S](op(v.underlying,v2.underlying),v.rowIndex,v2.colIndex)
    }

  implicit def liftInPlaceOpVSV[Op <: OpType,RL,CL,M,S]
  (implicit op: UFunc.InPlaceImpl2[Op,M,S], field: Field[S]) =
    new UFunc.InPlaceImpl2[Op,DataFrame[RL,CL,M,S],S] {
      def apply(v: DataFrame[RL, CL, M, S], v2: S) = op(v.underlying,v2)
    }

  implicit def liftInPlaceOpVVV[Op <: OpType, RL, CL, M, S]
  (implicit op: UFunc.InPlaceImpl2[Op,M,M], field: Field[S]) =
    new UFunc.InPlaceImpl2[Op,DataFrame[RL,CL,M,S],DataFrame[RL,CL,M,S]] {
      def apply(v: DataFrame[RL, CL, M, S], v2: DataFrame[RL, CL, M, S]) = op(v.underlying,v2.underlying)
    }

  private implicit def liftNorm2[RL,CL,M,S](implicit _uNorm2: norm.Impl2[M,Double,Double]) =
    new norm.Impl2[DataFrame[RL,CL,M,S],Double,Double] {
      def apply(v: DataFrame[RL,CL,M,S], v2: Double): Double = norm(v.underlying)
    }

  private implicit def liftInnerProduct[RL, CL, M, S](implicit _uDot: OpMulInner.Impl2[M, M, S]) =
    new OpMulInner.Impl2[DataFrame[RL, CL, M, S], DataFrame[RL, CL, M, S], S] {
      def apply(v: DataFrame[RL, CL, M, S], v2: DataFrame[RL, CL, M, S]): S = _uDot(v.underlying,v2.underlying)
    }

  implicit def canCopy[RL, CL, M, S] =
    new CanCopy[DataFrame[RL,CL,M,S]] {
      def apply(t: DataFrame[RL, CL, M, S]): DataFrame[RL, CL, M, S] = t.copy
    }

  implicit def scaleAddVSV[RL,CL,M,S](implicit _uScaleAdd: scaleAdd.InPlaceImpl3[M,S,M]) =
    new scaleAdd.InPlaceImpl3[DataFrame[RL,CL,M,S],S,DataFrame[RL,CL,M,S]] {
      def apply(v: DataFrame[RL, CL, M, S], v2: S, v3: DataFrame[RL, CL, M, S]): Unit = _uScaleAdd(v.underlying,v2,v3.underlying)
    }

  implicit def zerosLike[RL,CL,M,S](implicit man: ClassTag[S], zero: Zero[S], matEv: M <:< Matrix[S], uspace: MutableFiniteCoordinateField[M,(Int,Int),S]) =
    new CanCreateZerosLike[DataFrame[RL,CL,M,S],DataFrame[RL,CL,M,S]] {
      def apply(from: DataFrame[RL, CL, M, S]): DataFrame[RL, CL, M, S] =
        new DataFrame[RL,CL,M,S](from.underlying,from.rowIndex,from.colIndex)
    }

  // Warning -- Creates Zeros DataFram with empty mutable indices. Must re-index to recover actual Zeros.
  implicit def createZeros[RL,CL,M,S](implicit man: ClassTag[S], zero: Zero[S], matEv: M <:< Matrix[S], uspace: MutableFiniteCoordinateField[M,(Int,Int),S]) =
    new CanCreateZeros[DataFrame[RL,CL,M,S],(Int,Int)] {
      def apply(d: (Int, Int)): DataFrame[RL, CL, M, S] = new DataFrame[RL,CL,M,S](uspace.zero(d),Index[RL](),Index[CL]())
    }

  implicit def canDim[RL,CL,M,S] = new dim.Impl[DataFrame[RL,CL,M,S],(Int,Int)] {
    def apply(v: DataFrame[RL, CL, M, S]): (Int, Int) = (v.rows,v.cols)
  }

  implicit def canNegate[RL, CL, M, S](implicit _uNegate: OpNeg.Impl[M, M], man: ClassTag[S], zero: Zero[S],
                                       matEv: M <:< Matrix[S], uspace: MutableFiniteCoordinateField[M, (Int, Int), S]) =
    new OpNeg.Impl[DataFrame[RL, CL, M, S], DataFrame[RL, CL, M, S]] {
      def apply(v: DataFrame[RL, CL, M, S]): DataFrame[RL, CL, M, S] =
        new DataFrame[RL, CL, M, S](_uNegate(v.underlying), v.rowIndex, v.colIndex)
    }

  implicit def canTabulate[RL, CL, M, S](implicit _uTab: CanTabulate[(Int, Int), M, S], man: ClassTag[S], zero: Zero[S],
                                         matEv: M <:< Matrix[S],
                                         uspace: MutableFiniteCoordinateField[M, (Int, Int), S]) =
    new CanTabulate[(Int, Int), DataFrame[RL, CL, M, S], S] {
      def apply(d: (Int, Int), f: ((Int, Int)) => S): DataFrame[RL, CL, M, S] =
        new DataFrame[RL, CL, M, S](_uTab(d, f), Index[RL](), Index[CL]())
    }

  implicit def canTraverse[RL,CL,M,S](implicit _uTraverse: CanTraverseValues[M,S]) =
    new CanTraverseValues[DataFrame[RL,CL,M,S],S] {
      def traverse(from: DataFrame[RL, CL, M, S], fn: ValuesVisitor[S]): Unit = _uTraverse.traverse(from.underlying,fn)

      def isTraversableAgain(from: DataFrame[RL, CL, M, S]): Boolean = _uTraverse.isTraversableAgain(from.underlying)
    }

  implicit def canMap[RL,CL,M,NM,S,NS](implicit _uMap: CanMapValues[M,S,NS,NM], man: ClassTag[NS], zero: Zero[NS],
                                       matEv: NM <:< Matrix[NS], uspace: MutableFiniteCoordinateField[NM, (Int, Int), NS]) =
    new CanMapValues[DataFrame[RL,CL,M,S],S,NS,DataFrame[RL,CL,NM,NS]] {
      def map(from: DataFrame[RL, CL, M, S], fn: (S) => NS): DataFrame[RL, CL, NM, NS] =
        new DataFrame[RL,CL,NM,NS](_uMap.map(from.underlying,fn),from.rowIndex,from.colIndex)

      def mapActive(from: DataFrame[RL, CL, M, S], fn: (S) => NS): DataFrame[RL, CL, NM, NS] =
        new DataFrame[RL,CL,NM,NS](_uMap.mapActive(from.underlying,fn),from.rowIndex,from.colIndex)
    }

  implicit def canZipMap[RL,CL,M,NM,S,NS](implicit _uZipMap: CanZipMapValues[M,S,NS,NM], man: ClassTag[NS], zero: Zero[NS],
                                          matEv: NM <:< Matrix[NS], uspace: MutableFiniteCoordinateField[NM, (Int, Int), NS]) =
    new CanZipMapValues[DataFrame[RL,CL,M,S],S,NS,DataFrame[RL,CL,NM,NS]] {
      def map(from: DataFrame[RL, CL, M, S], from2: DataFrame[RL, CL, M, S],
              fn: (S, S) => NS): DataFrame[RL, CL, NM, NS] = {
        require(from.rows == from2.rows,"DataFrame row dimensions must match")
        require(from.cols == from2.cols,"DataFrame col dimensions must match")
        require(from.rowIndex.pairs.forall(li => from2.rowIndex(li._1) == li._2), "DataFrame row indices don't match")
        require(from.colIndex.pairs.forall(li => from2.colIndex(li._1) == li._2), "DataFrame col indices don't match")
        new DataFrame[RL,CL,NM,NS](_uZipMap.map(from.underlying,from2.underlying,fn),from.rowIndex,from.colIndex)
      }
    }

  implicit def mulVector[RL,CL,M,V,S](implicit _uMul: OpMulMatrix.Impl2[M,V,V], evVec: V <:< Vector[S],
                                      zero: Zero[S], semiring: Semiring[S], man: ClassTag[S]) =
    new OpMulMatrix.Impl2[DataFrame[RL,CL,M,S],V,Counter[RL,S]] {
      def apply(v: DataFrame[RL, CL, M, S], v2: V): Counter[RL, S] = {
        Encoder.fromIndex(v.rowIndex).decode(_uMul(v.underlying,v2))
      }
    }
//
//  implicit def space[RL,CL,M,S](implicit uspace: MutableCoordinateField[M,S],
//                                man: ClassTag[S], zero: Zero[S], matEv: M <:< Matrix[S])
//  : MutableCoordinateField[DataFrame[RL,CL,M,S],S] = {
//    import uspace._
//    MutableCoordinateField.make[DataFrame[RL,CL,M,S],S]
//  }
}