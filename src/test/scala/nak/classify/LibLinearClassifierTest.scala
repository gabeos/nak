/*
 * dialogue -- The University of Washington Dialogue Framework
 *
 * Copyright 2013 - Gabriel Schubiner
 *
 * LibLinearClassifierTest.scala is part of dialogue.
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

package nak.classify

import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import breeze.linalg.{SparseVector, Counter}

/**
 * dialogue
 * 6/12/14
 * @author Gabriel Schubiner <gabeos@cs.washington.edu>
 *
 *
 */

@RunWith(classOf[JUnitRunner])
class LibLinearClassifierTest
  extends ClassifierTrainerTestHarness
  with ContinuousTestHarness {
  def trainer[L, T]: Classifier.Trainer[L, Counter[T,Double]] =
    new LibLinearClassifier.CounterTrainer[L,T]
}

