/*
 * dialogue -- The University of Washington Dialogue Framework
 *
 * Copyright 2013 - Gabriel Schubiner
 *
 * LibLinear.scala is part of dialogue.
 *
 * dialogue is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * dialogue is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with dialogue.  If not, see <http://www.gnu.org/licenses/>.
 */

package nak.liblinear

import breeze.linalg.Vector

/**
 * dialogue
 * 6/12/14
 * @author Gabriel Schubiner <gabeos@cs.washington.edu>
 *
 *
 */
class LibLinear(model: Model, numLabels: Int) extends (Vector[Double] => Vector[Double]) {

  override def apply(x: Vector[Double]): Vector[Double] = {
    val ctxt = x.iterator.map(c => new FeatureNode(c._1+1, c._2).asInstanceOf[Feature]).toArray
    val labelScores = Array.fill(numLabels)(0.0)
    Linear.predictProbability(model, ctxt, labelScores)
    Vector(labelScores)
  }
}
