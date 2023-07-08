using SmartSnake;
using System.Drawing;
using Tyzegt.NN;

Console.CursorVisible = false;

int winnersToReproduce = 10;
int winnersOffspringsCount = 1000;
int randomSpecimensCount = 1000;

int gamesCount = winnersToReproduce * winnersOffspringsCount + randomSpecimensCount + winnersToReproduce;

float learnRate = 0.1f;
int[] topology = new int[] { 24, 18, 18, 4};
Point fieldSize = new Point(15, 15);
Point startPosition = new Point(fieldSize.X/2, fieldSize.Y/2);
List<string>? lastGameLog = null;

List<NeuralNetwork> population = new List<NeuralNetwork>();
float mutationRate = 1f;
float mutationStrength = 2f;

int gen = 0;
int bestScore = 0;

// initial population
for (int i = 0; i < gamesCount; i++)
{
    population.Add(new NeuralNetwork(learnRate, topology));
}

while (true)
{
    Task render;
    if (lastGameLog != null) render = Task.Run(() => Render(lastGameLog));
    else render = Task.CompletedTask;

    Round();

    await render;
    gen++;
}

void Round()
{
    var tasks = new List<Task<SnakeGame>>();
    foreach (NeuralNetwork network in population)
    {
        tasks.Add(Task.Run(() => new SnakeGame(fieldSize, startPosition, network).Play()));
    }
    Task.WaitAll(tasks.ToArray());
    var games = tasks.Select(t => t.Result).OrderByDescending(x => x.Score).Take(winnersToReproduce).ToList();
    lastGameLog = games.First().GameLog;
    if (bestScore < games.First().Score)
    {
        bestScore = games.First().Score;
    }

    Repopulate(games.Select(x => x.Brain).ToList());
}

void Repopulate(List<NeuralNetwork> species)
{
    population.Clear();
    foreach (var brain in species)
    {
        population.Add(new NeuralNetwork(brain));

        for (int i = 0; i < winnersOffspringsCount; i++)
        {
            var newBrain = new NeuralNetwork(brain);
            newBrain.Mutate(mutationRate, mutationStrength);
            population.Add(newBrain);
        }

    }
    for (int i = 0;i < randomSpecimensCount; i++)
    {
        population.Add(new NeuralNetwork(learnRate, topology));
    }
}

async Task Render(List<string> log)
{
    Console.Clear();
    foreach (var item in log)
    {
        Console.WriteLine(item);
        Console.WriteLine($"generation - {gen}");
        Console.WriteLine($"bestScore - {bestScore}");
        Console.SetCursorPosition(0, 0);
        await Task.Delay(20);
    }
}

