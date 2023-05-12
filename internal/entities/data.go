package entities

import (
	"fmt"
	"log"
	"strconv"
	"strings"
)

type Data [][]any

func ConvertToData(strData [][]string) (Data, error) {
	res := make(Data, 0, len(strData))
	for i, line := range strData {
		floatLine := make([]any, 0, len(line))
		for j, el := range line {
			float, err := strconv.ParseFloat(strings.Trim(el, " "), 64)
			if err != nil {
				return nil,
					fmt.Errorf("failed to parse element \"%v\" on line %d pos %d: %v", el, i+1, j+1, err)
			}
			floatLine = append(floatLine, float)
		}
		res = append(res, floatLine)
	}
	return res, nil
}

func (d Data) ConvertToStrings() [][]string {
	res := make([][]string, 0, len(d))
	for _, line := range d {
		strLine := make([]string, 0, len(line))
		for _, el := range line {
			str := fmt.Sprint(el)
			strLine = append(strLine, str)
		}
		res = append(res, strLine)
	}
	return res
}

func (d Data) ConvertToFloat64() [][]float64 {
	res := make([][]float64, 0, len(d))
	for _, line := range d {
		strLine := make([]float64, 0, len(line))
		for _, el := range line {
			str, ok := el.(float64)
			if ok {
				strLine = append(strLine, str)
				continue
			}
			intVal, ok := el.(int)
			if ok {
				strLine = append(strLine, float64(intVal))
				continue
			}
			log.Fatalf("convert float64 error: %v", el)
		}
		res = append(res, strLine)
	}
	return res
}
