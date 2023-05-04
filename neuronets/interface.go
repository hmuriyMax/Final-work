package neuronets

import (
	"errors"
	"github.com/fxsjy/gonn/gonn"
	"github.com/kirill-scherba/nnhelper"
	"log"
	"os"
	"rwGo/internal/csv"
)

const defaultNeuroFile = "/nn"

type Neural interface {
	CreateNN(bool, int, int, int)
	GetDirPath() string
	GetResult([]float64) string
	Forward([]float64) []float64
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

func createNN(path string, regen bool, param []int, iterCnt int, addPath string) *gonn.NeuralNetwork {
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

	nn := gonn.DefaultNetwork(param[0], param[1], param[2], false)

	input, target, err := csv.ParseAllData(path)
	if err != nil {
		log.Fatalln(err)
	}

	// Начинаем обучать нашу НС.
	nn.Train(input.ConvertToFloat64(), target.ConvertToFloat64(), iterCnt)

	// Сохраняем готовую НС в файл.
	gonn.DumpNN(path+addPath+defaultNeuroFile, nn)
	return nn
}
