using Tyzegt.NN.Matrices;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime;
using ILGPU;
using System.Diagnostics;

namespace Tyzegt.NN
{
    public class NeuralNetwork
    {
        private List<Matrix> weights;
        private float learningRate;
        Context context = Context.CreateDefault();
        IEnumerable<Device> cudaDevices;
        Accelerator accelerator;

        /// <summary>
        /// Topology represents amount of neurons in each layer, for example {4,2,2,1} means 4 neurons at input layer, 
        /// 2 neurons at first hidden layer, 2 neurons at second, and one neuron at output layer.
        /// </summary>
        public NeuralNetwork(float learningRate, params int[] topology)
        {
            cudaDevices = context.Devices.Where(x => x.AcceleratorType == AcceleratorType.Cuda);
            accelerator = cudaDevices.Any() ? cudaDevices.First().CreateAccelerator(context) : null;

            this.learningRate = learningRate;

            weights = new List<Matrix>();
            for (int i = 0; i < topology.Length - 1; i++)
            {
                var newMatrix = new Matrix(topology[i+1], topology[i]);
                SetRandomWeights(newMatrix);
                weights.Add(newMatrix);
            }
        }

        public void Train(float[] inputsList, float[] targetsList)
        {
            var inputs = new Matrix(inputsList);
            var targets = new Matrix(targetsList);

            var outputs = FeedForward(inputsList);

            var currentErrorList = targetsList.Zip(outputs.Last().ToArray(), (x, y) => x - y).ToArray();

            var currentErrors = new Matrix(currentErrorList);

            for (int i = weights.Count - 1; i >= 0; i--)
            {
                var deltas = Matrix.Dot(currentErrors * outputs[i+1] * (1 - outputs[i + 1]), Matrix.Transpose(outputs[i]), accelerator) * learningRate;
                var newErrors = Matrix.Dot(Matrix.Transpose(weights[i]), currentErrors, accelerator);
                weights[i] += deltas;
                currentErrors = newErrors;
            }

        }

        public float[] Query(params float[] inputValues)
        {
            List<Matrix> outputs = FeedForward(inputValues);

            return outputs.Last().ToArray();
        }

        private List<Matrix> FeedForward(float[] inputValues)
        {
            List<Matrix> outputs = new List<Matrix>();
            outputs.Add(new Matrix(inputValues));         // input layer

            foreach (var m in weights)
            {
                var currentOutputs = Matrix.Dot(m, outputs.Last(), accelerator);
                ApplyActivationFunction(currentOutputs);
                outputs.Add(currentOutputs);
            }

            return outputs;
        }

        private void SetRandomWeights(Matrix m)
        {
            var r = new Random();

            m.ProcessFunctionOverData((i, j) =>
                m[i,j] = (float)((float)r.NextDouble() - 0.5));
        }

        private void ApplyActivationFunction(Matrix m)
        {
            m.ProcessFunctionOverData((i, j) =>
                m[i, j] = ActivationFunction(m[i, j]));
        }

        private float ActivationFunction(float x)
        {
            return (float)(1.0 / (1.0 + Math.Pow(Math.E, -x)));
        }
    }
}
