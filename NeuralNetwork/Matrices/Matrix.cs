using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tyzegt.NN.Matrices
{
    public class Matrix
    {
        private float[,] data;

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
            data = new float[m, n];
        }

        public Matrix(float[] inputValues)
        {
            m = inputValues.Length;
            n = 1;
            data = new float[m, n];
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

        public float this[int x, int y]
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

        public static Matrix operator *(Matrix matrix, float value)
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

        public static Matrix operator *(float value, Matrix matrix)
        {
            return matrix * value;
        }

        public static Matrix operator +(Matrix matrix, float value)
        {
            var result = new Matrix(matrix.M, matrix.N);
            result.ProcessFunctionOverData((i, j) =>
                result[i, j] = matrix[i, j] + value);
            return result;
        }

        public static Matrix operator -(float value, Matrix matrix)
        {
            var result = new Matrix(matrix.M, matrix.N);
            result.ProcessFunctionOverData((i, j) =>
                result[i, j] = value - matrix[i, j]);
            return result;
        }

        public static Matrix Dot(Matrix matrix, Matrix matrix2, Accelerator accelerator = null)
        {
            var result = new Matrix(matrix.M, matrix2.N);

            if (accelerator != null && matrix.data.Length > 1000) // experimental
            {
                var acceleratedResult = MatrixMultiplyAccelerated(accelerator, matrix.data, matrix2.data);
                result.data = acceleratedResult;
                return result;
            }

            if (matrix.N != matrix2.M)
            {
                throw new ArgumentException("matrices can not be multiplied");
            }
            result.ProcessFunctionOverData((i, j) => {
                for (var k = 0; k < matrix.N; k++)
                {
                    result[i, j] += matrix[i, k] * matrix2[k, j];
                }
            });
            return result;
        }

        static float[,] MatrixMultiplyAccelerated(Accelerator accelerator, float[,] a, float[,] b)
        {
            var m = a.GetLength(0);
            var ka = a.GetLength(1);
            var kb = b.GetLength(0);
            var n = b.GetLength(1);

            if (ka != kb)
                throw new ArgumentException($"Cannot multiply {m}x{ka} matrix by {n}x{kb} matrix", nameof(b));

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView2D<float, Stride2D.DenseX>>(
                MatrixMultiplyAcceleratedKernel);

            using var aBuffer = accelerator.Allocate2DDenseX<float>(new Index2D(m, ka));
            using var bBuffer = accelerator.Allocate2DDenseX<float>(new Index2D(ka, n));
            using var cBuffer = accelerator.Allocate2DDenseX<float>(new Index2D(m, n));
            aBuffer.CopyFromCPU(a);
            bBuffer.CopyFromCPU(b);

            kernel(cBuffer.Extent.ToIntIndex(), aBuffer.View, bBuffer.View, cBuffer.View);

            // Reads data from the GPU buffer into a new CPU array.
            // Implicitly calls accelerator.DefaultStream.Synchronize() to ensure
            // that the kernel and memory copy are completed first.
            return cBuffer.GetAsArray2D();
        }


        static void MatrixMultiplyAcceleratedKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> aView,
            ArrayView2D<float, Stride2D.DenseX> bView,
            ArrayView2D<float, Stride2D.DenseX> cView)
        {
            var x = index.X;
            var y = index.Y;
            var sum = 0.0f;

            for (var i = 0; i < aView.IntExtent.Y; i++)
                sum += aView[new Index2D(x, i)] * bView[new Index2D(i, y)];

            cView[index] = sum;
        }

        public static Matrix Transpose(Matrix m)
        {
            var result = new Matrix(m.N, m.M);

            result.ProcessFunctionOverData((i, j) =>
                result[i, j] = m[j, i]);

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

        public float[] ToArray()
        {
            if (N > 1) throw new ArgumentException("cannot convert matrix with multiple columns");

            var result = new float[M];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = data[i, 0];
            }

            return result;
        }

        public void SubstractAllValuesFrom(float value)
        {
            this.ProcessFunctionOverData((i, j) =>
                this[i, j] = value - data[i, j]);
        }
    }
}
