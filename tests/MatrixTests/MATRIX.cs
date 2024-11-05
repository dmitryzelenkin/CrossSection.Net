using System;

public class Matrix
{
	private readonly int rows;
	private readonly int cols;
	private readonly double[,] mat;

	// Cholesky decomposition components
	private double[,] choleskyL;
	private double[,] choleskyLTranspose;
	private bool isPositiveDefinite;

	// LU decomposition components
	private double[,] luL;
	private double[,] luU;
	private double[,] P;
	private int[] pi;
	private double detOfP;

	public int Rows => rows;
	public int Cols => cols;

	public Matrix(double[,] matarray)
	{
		this.rows = matarray.GetLength(0);
		this.cols = matarray.GetLength(1);
		this.mat = new double[rows, cols];
		Array.Copy(matarray, mat, rows * cols);

		if (!IsSquare())
			throw new InvalidOperationException("Matrix must be square!");

		// Check if matrix is positive definite and compute Cholesky if it is
		this.isPositiveDefinite = CheckPositiveDefinite();
		if (this.isPositiveDefinite)
		{
			ComputeCholesky();
			this.choleskyLTranspose = TransposeArray(choleskyL);
		}
		else
		{
			// Compute LU decomposition if not positive definite
			ComputeLU();
		}
	}

	public bool IsSquare() => rows == cols;

	public double this[int row, int col]
	{
		get => mat[row, col];
		set => mat[row, col] = value;
	}

	private bool CheckPositiveDefinite()
	{
		if (!IsSquare()) return false;

		try
		{
			var testL = (double[,])mat.Clone();

			int n = rows;

			int i, j, k;

			double sum;
			for (i = 0; i < n; i++)
			{
				for (j = i; j < n; j++)
				{
					for (sum = testL[i, j], k = i - 1; k >= 0; k--)
					{
						sum -= testL[i, k] * testL[j, k];
					}

					if (i == j)
					{
						if (sum <= 0.0) //A, with rounding errors, is not positive-definite.
						{
							return false;
						}

					}
					else
					{
						testL[j, i] = sum / testL[i, i];
					}
				}
			}
			return true;
		}
		catch
		{
			return false;
		}
	}

	private void ComputeCholesky()
	{
		choleskyL = (double[,])mat.Clone();

		int n = rows;

		int i, j, k;

		double sum;
		for (i = 0; i < n; i++)
		{
			for (j = i; j < n; j++)
			{
				for (sum = choleskyL[i, j], k = i - 1; k >= 0; k--)
				{
					sum -= choleskyL[i, k] * choleskyL[j, k];
				}

				if (i == j)
				{
					if (sum <= 0.0) //A, with rounding errors, is not positive-definite.
					{
						throw new System.SystemException("Cholesky failed");
					}

					choleskyL[i, i] = System.Math.Sqrt(sum);
				}
				else
				{
					choleskyL[j, i] = sum / choleskyL[i, i];
				}
			}
		}

		for (i = 0; i < n; i++)
			for (j = 0; j < i; j++)
				choleskyL[j, i] = 0.0;

	}


	private void ComputeLU()
	{
		luL = new double[rows, rows];
		luU = new double[rows, rows];
		P = new double[rows, rows];
		pi = new int[rows];
		detOfP = 1;

		// Initialize L as identity matrix
		for (int i = 0; i < rows; i++)
			luL[i, i] = 1.0;

		// Copy matrix to U
		Array.Copy(mat, luU, rows * cols);

		// Initialize permutation array
		for (int i = 0; i < rows; i++)
			pi[i] = i;

		for (int k = 0; k < rows - 1; k++)
		{
			// Find pivot
			double p = 0;
			int k0 = k;
			for (int i = k; i < rows; i++)
			{
				double absVal = Math.Abs(luU[i, k]);
				if (absVal > p)
				{
					p = absVal;
					k0 = i;
				}
			}

			if (p == 0)
				throw new InvalidOperationException("Matrix is singular");

			// Swap rows if necessary
			if (k != k0)
			{
				(pi[k], pi[k0]) = (pi[k0], pi[k]);
				detOfP *= -1;

				for (int j = 0; j < k; j++)
					(luL[k, j], luL[k0, j]) = (luL[k0, j], luL[k, j]);

				for (int j = 0; j < rows; j++)
					(luU[k, j], luU[k0, j]) = (luU[k0, j], luU[k, j]);
			}

			// Compute multipliers and eliminate
			for (int i = k + 1; i < rows; i++)
			{
				double multiplier = luU[i, k] / luU[k, k];
				luL[i, k] = multiplier;

				for (int j = k; j < rows; j++)
					luU[i, j] -= multiplier * luU[k, j];
			}
		}

		// Build permutation matrix
		for (int i = 0; i < rows; i++)
			P[pi[i], i] = 1;
	}

	public double[] Solve(double[] v)
	{
		if (v == null)
			throw new ArgumentNullException(nameof(v));
		if (rows != v.Length)
			throw new InvalidOperationException("Wrong number of results in solution vector!");

		if (isPositiveDefinite)
		{
			Console.WriteLine("Cholesky L matrix:");
			for (int i = 0; i < rows; i++)
			{
				for (int j = 0; j < cols; j++)
				{
					Console.Write($"{choleskyL[i, j]:F6} ");
				}
				Console.WriteLine();
			}

			// Solve Ly = b
			double[] y = ForwardSubstitution(choleskyL, v);
			Console.WriteLine("\ny vector:");
			for (int i = 0; i < y.Length; i++)
			{
				Console.WriteLine($"y[{i}] = {y[i]:F6}");
			}

			// Solve L^T x = y
			var result = BackwardSubstitution(choleskyLTranspose, y);
			Console.WriteLine("\nFinal x vector:");
			for (int i = 0; i < result.Length; i++)
			{
				Console.WriteLine($"x[{i}] = {result[i]:F6}");
			}

			return result;
		}
		else
		{
			// Create permuted right-hand side
			double[] b = new double[rows];
			for (int i = 0; i < rows; i++)
				b[i] = v[pi[i]];

			// Solve Ly = b
			double[] y = ForwardSubstitution(luL, b);
			// Solve Ux = y
			return BackwardSubstitution(luU, y);
		}
	}


	private static double[] ForwardSubstitution(double[,] A, double[] b)
	{
		int n = b.Length;
		double[] x = new double[n];

		for (int i = 0; i < n; i++)
		{
			double sum = b[i];
			for (int j = 0; j < i; j++)
				sum -= A[i, j] * x[j];
			x[i] = sum / A[i, i];
		}
		return x;
	}

	private static double[] BackwardSubstitution(double[,] A, double[] b)
	{
		int n = b.Length;
		double[] x = new double[n];

		for (int i = n - 1; i >= 0; i--)
		{
			double sum = b[i];
			for (int j = i + 1; j < n; j++)
				sum -= A[i, j] * x[j];
			x[i] = sum / A[i, i];
		}
		return x;
	}

	private static double[,] TransposeArray(double[,] matrix)
	{
		int rows = matrix.GetLength(0);
		int cols = matrix.GetLength(1);
		var result = new double[cols, rows];

		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++)
				result[j, i] = matrix[i, j];

		return result;
	}
}