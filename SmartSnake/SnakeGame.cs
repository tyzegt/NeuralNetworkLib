using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using Tyzegt.NN;

namespace SmartSnake
{
    internal class SnakeGame
    {
        public Point FieldSize { get; set; }
        public List<Point> GameField { get; set; } = new List<Point>();
        public List<Point> Snake { get; set; } = new List<Point>();
        public Point FoodPosition { get; set; } 
        public List<float> VisionData { get; set; } = new List<float>();
        public NeuralNetwork Brain { get; set; }
        public bool IsAlive { get; set; } = true;
        public List<string> GameLog { get; set; } = new List<string>();
        public int LastOutputIndex { get; set; }
        public float[] LastOutputs { get; set; }
        private int life = 100;
        public int Score { get; set; }

        public SnakeGame(Point fieldSize, Point startPosition, NeuralNetwork brain)
        {
            InitGameField(fieldSize);
            Snake.Add(new Point(startPosition.X, startPosition.Y));
            Brain = brain;
            FoodPosition = GetEmptySpacePosition();
        }

        private void InitGameField(Point fieldSize)
        {
            FieldSize = fieldSize;
            for (int x = 0; x < fieldSize.X; x++)
            {
                for (int y = 0; y < fieldSize.Y; y++)
                {
                    GameField.Add(new Point(x, y));
                }
            }
        }

        public SnakeGame Play()
        {
            while(IsAlive)
            {
                LookAround();
                Move();
                Render();
            }
            return this;
        }

        private void Move()
        {
            life--;
            Point direction = GetDirection();

            var nextPosition = new Point(Snake.Last().X + direction.X, Snake.Last().Y + direction.Y);

            if (life <= 0 ||
                nextPosition.X >= FieldSize.X ||
                nextPosition.X < 0 ||
                nextPosition.Y >= FieldSize.Y ||
                nextPosition.Y < 0 ||
                Snake.Where(x => x.X == nextPosition.X && x.Y == nextPosition.Y).Any())
            {
                IsAlive = false;
            }
            else
            {
                Snake.Add(nextPosition);
                if (FoodPosition != nextPosition)
                {
                    Snake.Remove(Snake.First());
                }
                else
                {
                    life = 100;
                    Score += 1;
                    FoodPosition = GetEmptySpacePosition();
                }
            }
        }

        private Point GetDirection()
        {
            LastOutputs = Brain.Query(VisionData.ToArray());
            LastOutputIndex = LastOutputs.ToList().IndexOf(LastOutputs.Max());
            switch (LastOutputIndex)
            {
                case 0:
                    return new Point(-1, 0);
                case 1:
                    return new Point(0, 1);
                case 2:
                    return new Point(1, 0); 
                default:
                    return new Point(0, -1); 
            }
        }

        private void LookAround()
        {
            VisionData.Clear();
            LookForFood();
            LookForBodyParts();
            LookForWalls();

            NormalizeVisionData();
        }

        private void NormalizeVisionData()
        {
            var max = FieldSize.X > FieldSize.Y ? FieldSize.X : FieldSize.Y;
            for (int i = 0; i < VisionData.Count; i++)
            {
                VisionData[i] /= max;
                VisionData[i] *= 0.999999f;
                VisionData[i] += 0.000001f;
            }
        }

        private void LookForFood()
        {
            VisionData.Add(Snake.Last().X > FoodPosition.X && Snake.Last().Y > FoodPosition.Y ? Snake.Last().X - FoodPosition.X : 0);   // ↖
            VisionData.Add(Snake.Last().X == FoodPosition.X && Snake.Last().Y > FoodPosition.Y ? Snake.Last().Y - FoodPosition.Y : 0);  // ↑
            VisionData.Add(Snake.Last().X < FoodPosition.X && Snake.Last().Y > FoodPosition.Y ? Snake.Last().Y - FoodPosition.Y : 0);   // ↗
            VisionData.Add(Snake.Last().X > FoodPosition.X && Snake.Last().Y == FoodPosition.Y ? Snake.Last().X - FoodPosition.X : 0);  // ←
            VisionData.Add(Snake.Last().X < FoodPosition.X && Snake.Last().Y == FoodPosition.Y ? FoodPosition.X - Snake.Last().X : 0);  // →
            VisionData.Add(Snake.Last().X > FoodPosition.X && Snake.Last().Y < FoodPosition.Y ? FoodPosition.Y - Snake.Last().Y : 0);   // ↙
            VisionData.Add(Snake.Last().X == FoodPosition.X && Snake.Last().Y < FoodPosition.Y ? FoodPosition.Y - Snake.Last().Y : 0);  // ↓
            VisionData.Add(Snake.Last().X < FoodPosition.X && Snake.Last().Y < FoodPosition.Y ? FoodPosition.Y - Snake.Last().Y : 0);   // ↘
        }

        private void LookForBodyParts() // TODO: consider lighter algorythm
        {
            var list = new List<Point>();
            Point nearest;
            for (int i = 0; i < Snake.Last().Y; i++)
            {
                if (Snake.Where(x => x.X == Snake.Last().X && x.Y == i).Any())
                    list.Add(new Point(Snake.Last().X, i));
            }
            if (list.Any())
            {
                VisionData.Add(Snake.Last().Y - list.OrderByDescending(x => x.Y).First().Y);    // ↑
                list.Clear();
            }
            else
            {
                VisionData.Add(0);
            }
            for (int i = 1; i < FieldSize.X - Snake.Last().X && i < Snake.Last().Y; i++)
            {
                if (Snake.Where(x => x.X == Snake.Last().X + i && x.Y == Snake.Last().Y - i).Any())
                    list.Add(new Point(Snake.Last().X + i, Snake.Last().Y - i));
            }
            if (list.Any())
            {
                nearest = list.OrderByDescending(x => x.Y).First();
                VisionData.Add(nearest.X - Snake.Last().X);                                     // ↗
                list.Clear();
            }
            else
            {
                VisionData.Add(0);
            }
            list = Snake.Where(x => x.Y == Snake.Last().Y && x.X > Snake.Last().X).ToList();
            if (list.Any())
            {
                nearest = list.OrderBy(x => x.X).First();
                VisionData.Add(nearest.X - Snake.Last().X);                                     // →
                list.Clear();

            }
            else
            {
                VisionData.Add(0);
            }
            for (int i = 1; i < FieldSize.Y - Snake.Last().Y && i < FieldSize.X - Snake.Last().X; i++)
            {
                if (Snake.Where(x => x.X == Snake.Last().X + i && x.Y == Snake.Last().Y + i).Any())
                    list.Add(new Point(Snake.Last().X + i, Snake.Last().Y + i));
            }
            if (list.Any())
            {
                nearest = list.OrderBy(x => x.X).First();
                VisionData.Add(nearest.X - Snake.Last().X);                                     // ↘
                list.Clear();
            }
            else
            {
                VisionData.Add(0);
            }
            for (int i = 1; i < FieldSize.Y - Snake.Last().Y; i++)
            {
                if (Snake.Where(x => x.Y == Snake.Last().Y + i).Any())
                    list.Add(new Point(Snake.Last().X, Snake.Last().Y + i));
            }
            if (list.Any())
            {
                nearest = list.OrderBy(x => x.Y).First();
                VisionData.Add(nearest.Y - Snake.Last().Y);                                     // ↓
                list.Clear();
            }
            else
            {
                VisionData.Add(0);
            }
            for (int i = 1; i < FieldSize.Y - Snake.Last().Y && i <= Snake.Last().X; i++)
            {
                if (Snake.Where(x => x.X == Snake.Last().X - i && x.Y == Snake.Last().Y + i).Any())
                    list.Add(new Point(Snake.Last().X - i, Snake.Last().Y + i));
            }
            if (list.Any())
            {
                nearest = list.OrderBy(x => x.Y).First();
                VisionData.Add(nearest.Y - Snake.Last().Y);                                     // ↙
                list.Clear();
            }
            else
            {
                VisionData.Add(0);
            }
            list = Snake.Where(x => x.Y == Snake.Last().Y && x.X < Snake.Last().X).ToList();
            if (list.Any())
            {
                nearest = list.OrderByDescending(x => x.X).First();
                VisionData.Add(Snake.Last().X - nearest.X);
                list.Clear();                                                                           // ←
            }
            else
            {
                VisionData.Add(0);
            }
            for (int i = 1; i <= Snake.Last().X && i <= Snake.Last().Y; i++)
            {
                if (Snake.Where(x => x.X == Snake.Last().X - i && x.Y == Snake.Last().Y - i).Any())
                    list.Add(new Point(Snake.Last().X - i, Snake.Last().Y - i));
            }
            if (list.Any())
            {
                nearest = list.OrderByDescending(x => x.X).First();
                VisionData.Add(Snake.Last().X - nearest.X);
            }
            else
            {
                VisionData.Add(0);
            }
        }

        private void LookForWalls()
        {
            VisionData.Add(Snake.Last().Y);                                                             // ↑
            var distances = new Point(FieldSize.X - Snake.Last().X - 1, Snake.Last().Y);                // ↗
            VisionData.Add(distances.X < distances.Y ? distances.X : distances.Y);
            VisionData.Add(FieldSize.X - Snake.Last().X - 1);                                           // →
            distances = new Point(FieldSize.X - Snake.Last().X - 1, FieldSize.Y - Snake.Last().Y - 1);  // ↘
            VisionData.Add(distances.X < distances.Y ? distances.X : distances.Y);
            VisionData.Add(FieldSize.Y - Snake.Last().Y - 1);                                           // ↓
            distances = new Point(Snake.Last().X, FieldSize.Y - Snake.Last().Y - 1);                    // ↙
            VisionData.Add(distances.X < distances.Y ? distances.X : distances.Y);
            VisionData.Add(Snake.Last().X);                                                             // ←
            distances = new Point(Snake.Last().X, Snake.Last().Y);                                      // ↖
            VisionData.Add(distances.X < distances.Y ? distances.X : distances.Y);
        }

        private Point GetEmptySpacePosition()
        {
            List<Point> list = new List<Point>();
            for (int x = 0; x < FieldSize.X; x++)
            {
                for (int y = 0; y < FieldSize.Y; y++)
                {
                    list.Add(new Point(x, y));
                }
            }
            var allEmptySpaces = new List<Point>();
            var goodEmptySpaces = new List<Point>();

            foreach (Point space in list)
            {
                if (!Snake.Contains(space)) allEmptySpaces.Add(space);
                if (space.X != Snake.Last().X && space.Y != Snake.Last().Y) goodEmptySpaces.Add(space);
            }

            Random random = new Random();
            if (goodEmptySpaces.Count > 0)
                return goodEmptySpaces[random.Next(0, goodEmptySpaces.Count)];
            else
                return allEmptySpaces[random.Next(0, allEmptySpaces.Count)];
        }

        void Render()
        {
            var sb = new StringBuilder();
            for (int i = 0; i < FieldSize.X + 2; i++) sb.Append("█");
            sb.AppendLine();

            for (int y = 0; y < FieldSize.Y; y++)
            {
                sb.Append("█");
                for (int x = 0; x < FieldSize.X; x++)
                {
                    if (Snake.Last() == new Point(x, y)) sb.Append("O");
                    else if (Snake.Contains(new Point(x, y))) sb.Append("o");
                    else if (FoodPosition == new Point(x, y)) sb.Append("*");
                    else sb.Append(" ");
                }
                sb.Append("█");
                sb.AppendLine();
            }
            for (int i = 0; i < FieldSize.X + 2; i++) sb.Append("█");
            sb.AppendLine();
            sb.AppendLine($"life - {life}          ");
            GameLog.Add(sb.ToString());
        }
    }
}
