using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tyzegt.NN.Tests
{
    [TestClass()]
    public class NeuralNetworkTests
    {
        [TestMethod()]
        public void TrainTestNumbersShort()
        {
            var nn = new NeuralNetwork(0.1, 784, 100, 10);
            var trainDataList = File.ReadAllLines("TestData/mnist_train_100.csv");
            var testDataList = File.ReadAllLines("TestData/mnist_test_10.csv");

            int correctCount = TrainAndTestNeuralNetwork(nn, trainDataList, testDataList, 10);

            Assert.IsTrue(correctCount >= 6);
        }

        [TestMethod()]
        public void TrainTestNumbersLong() // Warning! Very Long! Unzip "TestData/TestData.7z" before run.
        {
            var nn = new NeuralNetwork(0.1, 784, 100, 10);
            var trainDataList = File.ReadAllLines("TestData/mnist_train.csv");
            var testDataList = File.ReadAllLines("TestData/mnist_test.csv");

            int correctCount = TrainAndTestNeuralNetwork(nn, trainDataList, testDataList);

            Assert.IsTrue(correctCount >= 9000);
        }

        [TestMethod()]
        public void TrainSimple() 
        {
            var nn = new NeuralNetwork(0.1, 4, 2, 1);
            var trainDataList = File.ReadAllLines("TestData/SimpleDataset.txt");

            for (int j = 1; j <= 20000; j++)
            {
                for (int i = 0; i < trainDataList.Length; i++)
                {
                    var allValues = trainDataList[i].Split(',').Select(x => Convert.ToDouble(x)).ToArray();
                    var inputs = allValues[1..].Select(x => (x * 0.99) + 0.01).ToArray();
                    var target = new double[] { (allValues[0]* 0.99) + 0.01 };

                    nn.Train(inputs, target);
                }
            }

            var correctCount = 0;
            foreach (var item in trainDataList)
            {
                var allValues = item.Split(',').Select(x => Convert.ToDouble(x)).ToArray();
                var inputs = allValues[1..].Select(x => (x * 0.99) + 0.01).ToArray();
                var target = new double[] { (allValues[0] * 0.99) + 0.01 };

                var result = nn.Query(inputs);
                if(Math.Round(target[0], 1) == Math.Round(result[0], 1)) correctCount++;
            }
            Assert.IsTrue(correctCount == 16);
        }

        [TestMethod()]
        public void TrainXOR()
        {
            var nn = new NeuralNetwork(0.1, 2, 3, 3, 1);
            var trainDataList = new double[][] { 
                new double[] { 0.01, 0.01, 0.01 },
                new double[] { 1, 1, 0.01 },
                new double[] { 1, 0.01, 1 },
                new double[] { 0.01, 1, 1 }
            };

            for (int j = 1; j <= 200000; j++)
            {
                foreach (var item in trainDataList)
                {
                    var inputs = item[1..];
                    var target = new double[] { item[0] };

                    nn.Train(inputs, target);
                }
            }

            var correctCount = 0;
            foreach (var item in trainDataList)
            {
                var inputs = item[1..];
                var target = new double[] { item[0] };

                var result = nn.Query(inputs);
                if (Math.Round(target[0], 1) == Math.Round(result[0], 1)) correctCount++;
            }
            Assert.IsTrue(correctCount == 4);
        }


        [TestMethod()]
        public void MatrixDotTest()
        {
            var m1 = new Matrices.Matrix(2, 2);
            m1[0, 0] = 1;
            m1[0, 1] = 2;
            m1[1, 0] = 3;
            m1[1, 1] = 4;

            var m2 = new Matrices.Matrix(2, 2);
            m2[0, 0] = 5;
            m2[0, 1] = 6;
            m2[1, 0] = 7;
            m2[1, 1] = 8;

            var d1 = Matrices.Matrix.Dot(m1, m2);

            Assert.AreEqual(d1[0, 0], 19);
            Assert.AreEqual(d1[0, 1], 22);
            Assert.AreEqual(d1[1, 0], 43);
            Assert.AreEqual(d1[1, 1], 50);


            var m3 = new Matrices.Matrix(3, 3);
            m3[0, 0] = 0.9;
            m3[0, 1] = 0.3;
            m3[0, 2] = 0.4;
            m3[1, 0] = 0.2;
            m3[1, 1] = 0.8;
            m3[1, 2] = 0.2;
            m3[2, 0] = 0.1;
            m3[2, 1] = 0.5;
            m3[2, 2] = 0.6;

            var m4 = new Matrices.Matrix(3, 1);
            m4[0, 0] = 0.9;
            m4[1, 0] = 0.1;
            m4[2, 0] = 0.8;

            var d2 = Matrices.Matrix.Dot(m3, m4);
            Assert.AreEqual(Math.Round(d2[0, 0], 2), 1.16);
            Assert.AreEqual(Math.Round(d2[1, 0], 2), 0.42);
            Assert.AreEqual(Math.Round(d2[2, 0], 2), 0.62);
        }

        private int TrainAndTestNeuralNetwork(NeuralNetwork nn, string[] trainDataList, string[] testDataList, int epochsCount = 1)
        {

            for (int j = 1; j <= epochsCount; j++)
            {
                for (int i = 0; i < trainDataList.Length; i++)
                {
                    var data = GetNormalizedTrainingData(trainDataList[i]);
                    Debug.WriteLine($"Training | epoch: {j} {i+1}/{trainDataList.Length}");
                    nn.Train(data.Item1, data.Item2);
                }
            }
            
            var correctCount = 0;
            foreach (var dataString in testDataList)
            {
                var data = GetNormalizedTestData(dataString);
                var result = nn.Query(data.Item2);

                var maxValue = result.Max();
                int maxIndex = result.ToList().IndexOf(maxValue);

                if (data.Item1 == result.ToList().IndexOf(result.Max()))
                {
                    correctCount++;
                }
            }

            return correctCount;
        }

        private Tuple<double[], double[]> GetNormalizedTrainingData (string s)
        {
            var allValues = s.Split(',').Select(x => Convert.ToDouble(x)).ToArray();
            var inputs = allValues[1..].Select(x => (x / 255 * 0.99) + 0.01).ToArray();
            var targets = new double[10].Select(x => x += 0.01).ToArray();
            targets[(int)allValues[0]] = 1;
            return new Tuple<double[], double[]> (inputs, targets);
        }

        private Tuple<int, double[]> GetNormalizedTestData (string s)
        {
            var allValues = s.Split(',').Select(x => Convert.ToDouble(x)).ToArray();
            var inputs = allValues[1..].Select(x => (x / 255 * 0.99) + 0.01).ToArray();
            return new Tuple<int, double[]> ((int)allValues[0], inputs);
        }
    }
}