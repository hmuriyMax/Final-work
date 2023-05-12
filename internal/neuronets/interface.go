package neuronets

import (
	"errors"
	"github.com/fxsjy/gonn/gonn"
	"github.com/kirill-scherba/nnhelper"
	"log"
	"os"
	"rwGo/internal/csv"
	"rwGo/internal/entities"
	"sync"
)

const defaultNeuroFile = "/nn"

type Neural interface {
	CreateNN(bool, int, int, int, int)
	GetDirPath() string
	GetResult([]float64) string
	Forward([]float64) []float64
	Generate(len int, num int) (inp entities.Data, out entities.Data)
}

type BaseNN struct {
	additionalPath string
	filepath       string
	nn             *gonn.NeuralNetwork
}

func (base *BaseNN) GetDirPath() string {
	return base.filepath
}

func (base *BaseNN) Forward(input []float64) []float64 {
	nn := gonn.LoadNN(base.filepath + base.additionalPath + defaultNeuroFile)
	return nn.Forward(input)
}

func countRes(data []float64, resMap map[int]string) (string, bool) {
	pos := nnhelper.MaxFloatPosition(data)
	res, ok := resMap[pos]
	return res, ok
}

func createNN(path string, regen bool, param []int, iterCnt int, addPath string, dataBatchSize int) *gonn.NeuralNetwork {
	if addPath != "" {
		err := os.Mkdir(path+addPath, 0660)
		if err != nil && !os.IsExist(err) {
			log.Fatalln(err)
		}
	}
	if _, err := os.Open(path + addPath + defaultNeuroFile); !errors.Is(err, os.ErrNotExist) && !regen {
		// Загружем НС из файла.
		return gonn.LoadNN(path + addPath + defaultNeuroFile)
	}

	input, target, err := csv.ParseAllData(path)
	if err != nil {
		log.Fatalln(err)
	}
	result := gonn.DefaultNetwork(param[0], param[1], param[2], false)

	var (
		min   int
		count int
		mu    = sync.Mutex{}
		wg    = sync.WaitGroup{}
	)

	for min != len(input) {
		max := getRight(min, dataBatchSize, len(input))
		wg.Add(1)
		go func(min, max int) {
			defer wg.Done()

			nn := gonn.DefaultNetwork(param[0], param[1], param[2], false)
			// Начинаем обучать нашу НС.
			nn.Train(input[min:max].ConvertToFloat64(), target[min:max].ConvertToFloat64(), iterCnt)

			// Добавляем веса в итоговую НС
			mu.Lock()
			result.InputLayer = entities.Sum(result.InputLayer, nn.InputLayer)
			result.HiddenLayer = entities.Sum(result.HiddenLayer, nn.HiddenLayer)
			result.WeightHidden = entities.SumDouble(result.WeightHidden, nn.WeightHidden)
			result.OutputLayer = entities.Sum(result.OutputLayer, nn.OutputLayer)
			result.WeightOutput = entities.SumDouble(result.WeightOutput, nn.WeightOutput)
			mu.Unlock()
			count++
		}(min, max)
		min = max
	}
	wg.Wait()
	// Делим на количество батчей
	divideFunc := func(i int, v float64) float64 {
		return v / float64(count)
	}
	divideFuncDouble := func(i int, j int, v float64) float64 {
		return v / float64(count)
	}
	result.InputLayer = entities.ForEach(result.InputLayer, divideFunc)
	result.HiddenLayer = entities.ForEach(result.HiddenLayer, divideFunc)
	result.WeightHidden = entities.ForEachDouble(result.WeightHidden, divideFuncDouble)
	result.OutputLayer = entities.ForEach(result.OutputLayer, divideFunc)
	result.WeightOutput = entities.ForEachDouble(result.WeightOutput, divideFuncDouble)

	// Сохраняем готовую НС в файл.
	gonn.DumpNN(path+addPath+defaultNeuroFile, result)

	return result
}

func getRight(left, batch, len int) int {
	if left+batch > len {
		return len
	} else {
		return left + batch
	}
}
