using Tyzegt.NN.Matrices;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime;
using ILGPU;
using System.Diagnostics;
using System.Text.Json.Serialization;

namespace Tyzegt.NN
{
    public class NeuralNetwork
    {
        public List<Matrix> Weights { get; set; }
        public float LearningRate { get; set; }
        static Context context = Context.CreateDefault();
        static IEnumerable<Device> cudaDevices = context.Devices.Where(x => x.AcceleratorType == AcceleratorType.Cuda);
        static Accelerator accelerator = cudaDevices.Any() ? cudaDevices.First().CreateAccelerator(context) : null;
        public int[] Topology { get; set; }

        /// <summary>
        /// Topology represents amount of neurons in each layer, for example {4,2,2,1} means 4 neurons at input layer, 
        /// 2 neurons at first hidden layer, 2 neurons at second, and one neuron at output layer.
        /// </summary>
        public NeuralNetwork(float learningRate, params int[] topology)
        {
            Init(learningRate, topology);
            for (int i = 0; i < topology.Length - 1; i++)
            {
                var newMatrix = new Matrix(topology[i + 1], topology[i]);
                SetRandomWeights(newMatrix);
                Weights.Add(newMatrix);
            }
        }

        [JsonConstructor]
        public NeuralNetwork()
        {
        }

        /// <summary>
        /// Create new neural network and copy weights and learning rate from template network
        /// </summary>
        public NeuralNetwork(NeuralNetwork template)
        {
            Init(template.LearningRate, template.Topology);
            for (int i = 0; i < Topology.Length - 1; i++)
            {
                var newMatrix = new Matrix(Topology[i + 1], Topology[i]);
                
                for (int m = 0; m < newMatrix.M; m++)
                {
                    for (int n = 0; n < newMatrix.N; n++)
                    {
                        newMatrix[m,n] = template.Weights[i][m,n];
                    }
                }

                Weights.Add(newMatrix);
            }
        }

        private void Init(float learningRate, params int[] topology)
        {
            this.Topology = topology;

            this.LearningRate = learningRate;

            Weights = new List<Matrix>();

        }

        public void Mutate(float mutationRate, float mutationStrength)
        {
            Random r = new Random();
            foreach (var matrix in Weights) {
                for (int i = 0; i < matrix.M; i++)
                {
                    for (int j = 0; j < matrix.N; j++)
                    {
                        if(r.NextSingle() < mutationRate)
                        {
                            var value = r.NextSingle() * mutationStrength;
                            if (r.Next(2) == 0) value = -value;
                            matrix[i,j] += value;
                        }
                    }
                }
            }
        }

        public void Train(float[] inputsList, float[] targetsList)
        {
            var inputs = new Matrix(inputsList);
            var targets = new Matrix(targetsList);

            var outputs = FeedForward(inputsList);

            var currentErrorList = targetsList.Zip(outputs.Last().ToArray(), (x, y) => x - y).ToArray();

            var currentErrors = new Matrix(currentErrorList);

            for (int i = Weights.Count - 1; i >= 0; i--)
            {
                var deltas = Matrix.Dot(currentErrors * outputs[i+1] * (1 - outputs[i + 1]), Matrix.Transpose(outputs[i]), accelerator) * LearningRate;
                var newErrors = Matrix.Dot(Matrix.Transpose(Weights[i]), currentErrors, accelerator);
                Weights[i] += deltas;
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

            foreach (var m in Weights)
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
