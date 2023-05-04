package neuronets

import (
	"fmt"
	"log"
	"math/rand"
	"rwGo/internal/entities"
)

type MinusNet struct {
	BaseNN
	inputCount, hiddenCount, outputCount int
}

const dirMinus = "datasets/minusDetect"

func NewMinusNet() MinusNet {
	return MinusNet{
		BaseNN{filepath: dirMinus},
		2, 1, 2,
	}
}

func (g *MinusNet) CreateNN(regenerate bool, middleCount int, iterations int, id int) {
	addPath := ""
	if id > 0 {
		addPath = fmt.Sprintf("/%d", id)
	}
	g.additionalPath = addPath
	g.nn = createNN(g.filepath, regenerate, []int{g.inputCount, middleCount, g.outputCount}, iterations, addPath)
}

func (g *MinusNet) GetResult(data []float64) string {
	outPut, found := countRes(data, map[int]string{
		0: "Положительное",
		1: "Отрицательное",
	})
	if !found {
		log.Println("error parsing")
	}
	return outPut
}

func (g *MinusNet) Generate(len int, num int) (inp entities.Data, out entities.Data) {

	for i := 0; i < len; i++ {
		a := rand.Intn(num*2) - num
		b := rand.Intn(num*2) - num
		inp = append(inp, []any{a, b})
		var outExp []any
		if a*b < 0 {
			outExp = append(outExp, 0, 1)
		} else {
			outExp = append(outExp, 1, 0)
		}
		out = append(out, outExp)
	}
	return
}
