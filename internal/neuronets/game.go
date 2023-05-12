package neuronets

import (
	"fmt"
	"log"
)

type GameNet struct {
	BaseNN
	inputCount, hiddenCount, outputCount int
}

const dirGame = "datasets/game"

func NewGameNet() GameNet {
	return GameNet{
		BaseNN{filepath: dirGame},
		3, 1, 4,
	}
}

func (g *GameNet) CreateNN(regenerate bool, middleCount int, iterations int, id int, teachBatchSize int) {
	addPath := ""
	if id > 0 {
		addPath = fmt.Sprintf("/%d", id)
	}
	g.additionalPath = addPath
	g.nn = createNN(g.filepath, regenerate, []int{g.inputCount, middleCount, g.outputCount}, iterations, addPath, teachBatchSize)
}

func (g *GameNet) GetResult(data []float64) string {
	outPut, found := countRes(data, map[int]string{
		0: "Атаковать",
		1: "Красться",
		2: "Убегать",
		3: "Ничего не делать",
	})
	if !found {
		log.Println("error parsing")
	}
	return outPut
}
