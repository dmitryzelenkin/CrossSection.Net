// <copyright>
//https://github.com/IbrahimFahdah/CrossSection.Net

//Copyright(c) 2019 Ibrahim Fahdah

//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:

//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.
//</copyright>

using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;


namespace MatrixTestsNm
{
	[TestClass]
	public class MatrixTests
	{
		private const double Tolerance = 1e-10;

		[TestMethod]
		public void TestPositiveDefiniteMatrix()
		{
			// Test case 1: Simple 2x2 positive definite matrix
			double[,] matData = {
			{ 4, 1 },
			{ 1, 3 }
		};
			double[] b = { 1, 2 };
			double[] expectedX = { 0.0909090909090909, 0.6363636363636364 };

			Matrix mat = new Matrix(matData);
			double[] result = mat.Solve(b);

			AssertVectorEqual(expectedX, result, Tolerance);
		}

		[TestMethod]
		public void TestLargerPositiveDefiniteMatrix()
		{
			// Test case 2: 3x3 positive definite matrix
			double[,] matData = {
			{ 4, -1, 0 },
			{ -1, 4, -1 },
			{ 0, -1, 4 }
		};
			double[] b = { 1, 5, 0 };
			double[] expectedX = { 0.625, 1.5, 0.375 };

			Matrix mat = new Matrix(matData);
			double[] result = mat.Solve(b);

			AssertVectorEqual(expectedX, result, Tolerance);
		}

		[TestMethod]
		public void TestNonPositiveDefiniteMatrix()
		{
			// Test case 3: Non-positive definite matrix (will use LU decomposition)
			double[,] matData = {
			{ 1, 2, 3 },
			{ 2, 5, 3 },
			{ 1, 0, 8 }
		};
			double[] b = { 14, 18, 20 };
			double[] expectedX = { -92, 32, 14 };

			Matrix mat = new Matrix(matData);
			double[] result = mat.Solve(b);

			AssertVectorEqual(expectedX, result, Tolerance);
		}

		// [TestMethod]
		// public void TestSymmetricPositiveDefiniteMatrix()
		// {
		// 	// Test case 4: 4x4 symmetric positive definite matrix
		// 	double[,] matData = {
		// 	{ 5, 1, 2, 0 },
		// 	{ 1, 4, -1, 0 },
		// 	{ 2, -1, 6, 2 },
		// 	{ 0, 0, 2, 5 }
		// };
		// 	double[] b = { 10, 3, 15, 12 };
		// 	double[] expectedX = { 1.5471698113207547, 0.6226415094339622, 2.0754716981132075, 1.8867924528301887 };

		// 	Matrix mat = new Matrix(matData);
		// 	double[] result = mat.Solve(b);

		// 	AssertVectorEqual(expectedX, result, Tolerance);
		// }

		[TestMethod]
		public void TestHilbertMatrix()
		{
			// Test case 5: 3x3 Hilbert matrix (known to be poorly conditioned)
			double[,] matData = {
			{ 1, 1.0/2, 1.0/3 },
			{ 1.0/2, 1.0/3, 1.0/4 },
			{ 1.0/3, 1.0/4, 1.0/5 }
		};
			double[] b = { 1, 0, 0 };
			double[] expectedX = { 9, -36, 30 }; // Approximate solution

			Matrix mat = new Matrix(matData);
			double[] result = mat.Solve(b);

			AssertVectorEqual(expectedX, result, 1e-6); // Note: Using larger tolerance due to conditioning
		}

		[TestMethod]
		[ExpectedException(typeof(InvalidOperationException))]
		public void TestSingularMatrix()
		{
			// Test case 6: Singular matrix (should throw exception)
			double[,] matData = {
			{ 1, 2, 3 },
			{ 2, 4, 6 },
			{ 1, 2, 3 }
		};
			double[] b = { 1, 2, 3 };

			Matrix mat = new Matrix(matData);
			mat.Solve(b); // Should throw InvalidOperationException
		}

		[TestMethod]
		[ExpectedException(typeof(ArgumentNullException))]
		public void TestNullRightHandSide()
		{
			// Test case 7: Null right-hand side vector
			double[,] matData = {
			{ 1, 0 },
			{ 0, 1 }
		};
			Matrix mat = new Matrix(matData);
			mat.Solve(null); // Should throw ArgumentNullException
		}

		[TestMethod]
		public void TestVerifyingSolution()
		{
			// Test case 8: Verify solution by multiplying back
			double[,] matData = {
			{ 4, 1 },
			{ 1, 3 }
		};
			double[] b = { 1, 2 };

			Matrix mat = new Matrix(matData);
			double[] x = mat.Solve(b);

			// Verify Ax = b
			double[] computed = MultiplyMatrixVector(matData, x);
			AssertVectorEqual(b, computed, Tolerance);
		}

		private void AssertVectorEqual(double[] expected, double[] actual, double tolerance)
		{
			Assert.AreEqual(expected.Length, actual.Length, "Vectors have different lengths");
			for (int i = 0; i < expected.Length; i++)
			{
				Assert.AreEqual(expected[i], actual[i], tolerance,
					$"Vector difference at index {i}: expected {expected[i]}, got {actual[i]}");
			}
		}

		private double[] MultiplyMatrixVector(double[,] matrix, double[] vector)
		{
			int rows = matrix.GetLength(0);
			int cols = matrix.GetLength(1);
			double[] result = new double[rows];

			for (int i = 0; i < rows; i++)
			{
				result[i] = 0;
				for (int j = 0; j < cols; j++)
				{
					result[i] += matrix[i, j] * vector[j];
				}
			}

			return result;
		}
	}
}
