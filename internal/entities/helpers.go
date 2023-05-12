package entities

type (
	Float64Slice      []float64
	Float64SliceSlice [][]float64
)

func Sum(lhs, rhs Float64Slice) Float64Slice {
	var res Float64Slice
	if len(lhs) > len(rhs) {
		res = lhs
		for i, v := range rhs {
			res[i] += v
		}
	} else {
		res = rhs
		for i, v := range lhs {
			res[i] += v
		}
	}
	return res
}

func SumDouble(lhs, rhs Float64SliceSlice) Float64SliceSlice {
	var res Float64SliceSlice
	if len(lhs) > len(rhs) {
		res = lhs
		for i, v := range rhs {
			res[i] = Sum(v, res[i])
		}
	} else {
		res = rhs
		for i, v := range lhs {
			res[i] = Sum(v, res[i])
		}
	}
	return res
}

func ForEach(s Float64Slice, f func(i int, val float64) float64) Float64Slice {
	for i, v := range s {
		v = f(i, v)
	}
	return s
}

func ForEachDouble(s Float64SliceSlice, f func(row, col int, val float64) float64) Float64SliceSlice {
	for row, slice := range s {
		for col, v := range slice {
			v = f(row, col, v)
		}
	}
	return s
}
