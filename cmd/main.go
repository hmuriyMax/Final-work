package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"runtime"
	"rwGo/internal/csv"
	"rwGo/internal/neuronets"
	"sync"
	"time"
)

type Result struct {
	TotalTime   int
	Experiments []struct {
		Neurons    int
		Iterations int
		Accuracy   int
	}
}

func PerformNS(nn neuronets.Neural, mC int, it int, id int, teachBatchSize int) int {
	fmt.Printf("Neuro params: middleneuros: %d, steps: %d\n", mC, it)
	nn.CreateNN(true, mC, it, id, teachBatchSize)

	in, out := nn.Generate(testTries, stdDev)
	correctCount := 0
	inputFloats := in.ConvertToFloat64()
	outputFloats := out.ConvertToFloat64()
	for i := 0; i < testTries; i++ {
		if nn.GetResult(outputFloats[i]) == nn.GetResult(nn.Forward(inputFloats[i])) {
			correctCount++
		}
	}
	percent := correctCount * 100 / testTries
	fmt.Printf("Correct items: %v \n", correctCount)
	return percent
}

const (
	testTries  = 100
	learnTries = 1000
	outputFile = "output.json"
	attempts   = 3
	stdDev     = 1000
)

func app(procNum int) int {
	var res Result
	fmt.Printf("Running on %v cores (default: %v)\n", procNum, runtime.NumCPU())
	runtime.GOMAXPROCS(procNum)
	nn := neuronets.NewMinusNet()
	i, o := nn.Generate(learnTries, stdDev)
	err := csv.WriteCSV(nn.GetDirPath()+csv.InputName, i)
	if err != nil {
		log.Fatalln(err)
	}
	err = csv.WriteCSV(nn.GetDirPath()+csv.OutputName, o)
	if err != nil {
		log.Fatalln(err)
	}
	log.Printf("started performing experiments")
	start := time.Now()
	wg := sync.WaitGroup{}
	mu := sync.Mutex{}
	for n := 1; n <= 20; n++ {
		for iterNumPow := 0; iterNumPow < 7; iterNumPow++ {
			wg.Add(1)
			go func(n int, iterNumPow int) {
				defer wg.Done()
				newN := neuronets.NewMinusNet()
				iterNum := int(math.Pow(2, float64(iterNumPow)))
				accur := 0
				for i := 0; i < attempts; i++ {
					accur += PerformNS(&newN, n, iterNum, n*10+iterNumPow, learnTries)
				}
				accur /= attempts
				mu.Lock()
				res.Experiments = append(res.Experiments, struct {
					Neurons    int
					Iterations int
					Accuracy   int
				}{Neurons: n, Iterations: iterNum, Accuracy: accur})
				mu.Unlock()
			}(n, iterNumPow)
		}
	}
	wg.Wait()
	res.TotalTime = int(time.Since(start).Milliseconds())
	log.Printf("finished performing experiments")
	outPut, err := os.OpenFile(outputFile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		log.Fatalf("open output.txt error: %v", err)
	}
	bytes, err := json.MarshalIndent(res, "", "\t")
	if err != nil {
		log.Fatalf("marshal error: %v", err)
	}
	numBytes, err := outPut.Write(bytes)
	if err != nil {
		log.Fatalf("write error: %v", err)
	}
	log.Printf("wrote %d bytes to %s", numBytes, outputFile)
	return res.TotalTime
}

func main() {
	err := os.RemoveAll("./datasets/minusDetect")
	if err != nil {
		log.Fatalf("remove dataset error: %v", err)
	}
	_ = os.Mkdir("./datasets/minusDetect", 0660)
	fmt.Printf("clear dataset done")
	fmt.Printf("total time: %v\n", app(8))
}
