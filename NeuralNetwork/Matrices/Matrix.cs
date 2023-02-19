using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tyzegt.NN.Matrices
{
    public class Matrix
    {
        private double[,] data;

        private int m;
        private int n;

        /// <summary>
        /// Rows count
        /// </summary>// Строки
        public int M { get => m; }


        /// <summary>
        /// Columns count
        /// </summary>
        public int N { get => n; }

        public Matrix(int m, int n)
        {
            this.m = m;
            this.n = n;
            data = new double[m, n];
        }

        public Matrix(double[] inputValues)
        {
            m = inputValues.Length;
            n = 1;
            data = new double[m, n];
            for (int i = 0; i < m; i++)
            {
                data[i, 0] = inputValues[i];
            }
        }

        public void ProcessFunctionOverData(Action<int, int> func)
        {
            for (var i = 0; i < M; i++)
            {
                for (var j = 0; j < N; j++)
                {
                    func(i, j);
                }
            }
        }

        private void SetToZero(int i, int j)
        {
            data[i, j] = 0;
        }

        public double this[int x, int y]
        {
            get
            {
                return data[x, y];
            }
            set
            {
                data[x, y] = value;
            }
        }

        public static Matrix operator *(Matrix matrix, double value)
        {
            var result = new Matrix(matrix.M, matrix.N);
            result.ProcessFunctionOverData((i, j) =>
                result[i, j] = matrix[i, j] * value);
            return result;
        }

        public static Matrix operator *(Matrix m1, Matrix m2)
        {
            if (m1.N != m2.N || m1.M != m2.M) throw new ArgumentException("matrices have different dimentions");
            var result = new Matrix(m1.M, m1.N);
            result.ProcessFunctionOverData((i, j) =>
                result[i, j] = m1[i, j] * m2[i, j]);
            return result;
        }

        public static Matrix operator +(Matrix m1, Matrix m2)
        {
            if (m1.N != m2.N || m1.M != m2.M) throw new ArgumentException("matrices have different dimentions");
            var result = new Matrix(m1.M, m1.N);
            result.ProcessFunctionOverData((i, j) =>
                result[i, j] = m1[i, j] + m2[i, j]);
            return result;
        }

        public static Matrix operator *(double value, Matrix matrix)
        {
            return matrix * value;
        }

        public static Matrix operator +(Matrix matrix, double value)
        {
            var result = new Matrix(matrix.M, matrix.N);
            result.ProcessFunctionOverData((i, j) =>
                result[i, j] = matrix[i, j] + value);
            return result;
        }

        public static Matrix operator -(double value, Matrix matrix)
        {
            var result = new Matrix(matrix.M, matrix.N);
            result.ProcessFunctionOverData((i, j) =>
                result[i, j] = value - matrix[i, j]);
            return result;
        }

        public static Matrix Dot(Matrix matrix, Matrix matrix2)
        {
            if (matrix.N != matrix2.M)
            {
                throw new ArgumentException("matrices can not be multiplied");
            }
            var result = new Matrix(matrix.M, matrix2.N);
            result.ProcessFunctionOverData((i, j) => {
                for (var k = 0; k < matrix.N; k++)
                {
                    result[i, j] += matrix[i, k] * matrix2[k, j];
                }
            });
            return result;
        }

        public static Matrix Transpose(Matrix m)
        {
            var result = new Matrix(m.N, m.M);

            result.ProcessFunctionOverData((i, j) =>
                result[i, j] = m[j, i]);

            //for (int i = 0; i < m.M; i++)
            //{
            //    for (int j = 0; j < m.N; j++)
            //    {
            //        result[j, i] = m[i, j];
            //    }
            //}

            return result;
        }

        public override string ToString()
        {
            var sb = new StringBuilder();
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    sb.Append($"{this[i, j]}   ");
                }
                sb.Append(Environment.NewLine);
            }
            return sb.ToString();
        }

        public double[] ToArray()
        {
            if (N > 1) throw new ArgumentException("cannot convert matrix with multiple columns");

            var result = new double[M];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = data[i, 0];
            }

            return result;
        }

        public void SubstractAllValuesFrom(double value)
        {
            this.ProcessFunctionOverData((i, j) =>
                this[i, j] = value - data[i, j]);
        }
    }
}
