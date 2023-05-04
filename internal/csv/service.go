package csv

import (
	"encoding/csv"
	"os"
	"rwGo/internal/entities"
)

const (
	InputName  = "/input.csv"
	OutputName = "/output.csv"
)

func ParseCSV(filepath string) (entities.Data, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()
	r := csv.NewReader(file)
	strData, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	return entities.ConvertToData(strData)
}

func WriteCSV(filepath string, data entities.Data) error {
	file, err := os.OpenFile(filepath, os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		return err
	}
	defer func() { _ = file.Close() }()
	w := csv.NewWriter(file)
	return w.WriteAll(data.ConvertToStrings())
}

func ParseAllData(dirPath string) (entities.Data, entities.Data, error) {
	input, err := ParseCSV(dirPath + InputName)
	if err != nil {
		return nil, nil, err
	}
	output, err := ParseCSV(dirPath + OutputName)
	if err != nil {
		return nil, nil, err
	}
	return input, output, nil
}
