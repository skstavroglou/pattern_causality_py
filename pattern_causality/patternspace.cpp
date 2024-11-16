#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <cmath>
#include <limits>

static double hashing(const std::vector<int>& vec) {
    double hash_val = 0.0;
    for (size_t i = 0; i < vec.size(); i++) {
        hash_val += vec[i] * std::tgamma(i + 4); // i+3 since R's 1-based indexing used i+2
    }
    return hash_val;
}

static std::vector<double> cpp_pattern_vector_difference(const std::vector<double>& sVec, int E) {
    std::vector<double> result(E - 1);
    
    // Check for None/NaN values
    for (const auto& val : sVec) {
        if (std::isnan(val)) {
            std::fill(result.begin(), result.end(), std::numeric_limits<double>::quiet_NaN());
            return result;
        }
    }
    
    // Convert to pattern values
    std::vector<int> p_vec;
    for (const auto& val : sVec) {
        if (val > 0) {
            p_vec.push_back(3);
        } else if (val < 0) {
            p_vec.push_back(1);
        } else {
            p_vec.push_back(2);
        }
    }
    
    result[0] = hashing(p_vec);
    return result;
}

static PyObject* patternspace(PyObject* self, PyObject* args) {
    PyObject* sm_obj;
    int E;
    
    if (!PyArg_ParseTuple(args, "Oi", &sm_obj, &E)) {
        return NULL;
    }
    
    PyArrayObject* sm_array = (PyArrayObject*)PyArray_FROM_OTF(sm_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (sm_array == NULL) {
        return NULL;
    }
    
    if (PyArray_NDIM(sm_array) != 2) {
        Py_DECREF(sm_array);
        PyErr_SetString(PyExc_ValueError, "Input must be a 2D array");
        return NULL;
    }
    
    npy_intp* dims = PyArray_DIMS(sm_array);
    npy_intp num_rows = dims[0];
    npy_intp num_cols = dims[1];
    
    npy_intp out_dims[2] = {num_rows, 1};
    PyObject* result = PyArray_SimpleNew(2, out_dims, NPY_DOUBLE);
    if (result == NULL) {
        Py_DECREF(sm_array);
        return NULL;
    }
    
    double* sm_data = (double*)PyArray_DATA(sm_array);
    double* result_data = (double*)PyArray_DATA((PyArrayObject*)result);
    
    for (npy_intp i = 0; i < num_rows; i++) {
        std::vector<double> row(sm_data + i * num_cols, sm_data + (i + 1) * num_cols);
        std::vector<double> pattern = cpp_pattern_vector_difference(row, E);
        result_data[i] = pattern[0];
    }
    
    Py_DECREF(sm_array);
    return result;
}
extern "C" {
    PyObject* pattern_vector_difference(PyObject* self, PyObject* args) {
        PyObject* input_array;
        int E;
        
        if (!PyArg_ParseTuple(args, "Oi", &input_array, &E)) {
            return NULL;
        }

        if (PyFloat_Check(input_array) || PyLong_Check(input_array)) {
            double value;
            if (PyFloat_Check(input_array)) {
                value = PyFloat_AsDouble(input_array);
            } else {
                value = (double)PyLong_AsLong(input_array);
            }
            
            npy_intp dims[1] = {1};
            PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
            if (result_array == NULL) {
                return NULL;
            }
            
            double* output_data = (double*)PyArray_DATA(result_array);
            
            std::vector<int> p_vec;
            if (value > 0) {
                p_vec.push_back(3);
            } else if (value < 0) {
                p_vec.push_back(1);
            } else {
                p_vec.push_back(2);
            }
            
            output_data[0] = hashing(p_vec);
            return (PyObject*)result_array;
        }

        PyArrayObject* array = (PyArrayObject*)PyArray_FROM_OTF(input_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        if (array == NULL) {
            PyErr_SetString(PyExc_TypeError, "Could not convert input to numpy array");
            return NULL;
        }

        if (PyArray_NDIM(array) != 1) {
            Py_DECREF(array);
            PyErr_SetString(PyExc_ValueError, "Input must be a 1D array");
            return NULL;
        }

        npy_intp length = PyArray_DIM(array, 0);
        
        npy_intp dims[1] = {1};
        PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if (result_array == NULL) {
            Py_DECREF(array);
            return NULL;
        }

        double* input_data = (double*)PyArray_DATA(array);
        double* output_data = (double*)PyArray_DATA(result_array);

        for (npy_intp i = 0; i < length; i++) {
            if (std::isnan(input_data[i])) {
                output_data[0] = std::numeric_limits<double>::quiet_NaN();
                Py_DECREF(array);
                return (PyObject*)result_array;
            }
        }

        std::vector<int> p_vec;
        for (npy_intp i = 0; i < length; i++) {
            if (input_data[i] > 0) {
                p_vec.push_back(3);
            } else if (input_data[i] < 0) {
                p_vec.push_back(1);
            } else {
                p_vec.push_back(2);
            }
        }

        output_data[0] = hashing(p_vec);

        Py_DECREF(array);
        return (PyObject*)result_array;
    }
}

static PyMethodDef PatternSpaceMethods[] = {
    {"patternspace", patternspace, METH_VARARGS, "Calculate pattern space matrix from signature matrix"},
    {"pattern_vector_difference", (PyCFunction)pattern_vector_difference, METH_VARARGS,
     "Calculate pattern vector difference"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef patternspacemodule = {
    PyModuleDef_HEAD_INIT,
    "patternspace",
    "Pattern space calculation module",
    -1,
    PatternSpaceMethods
};

PyMODINIT_FUNC PyInit_patternspace(void) {
    import_array();
    return PyModule_Create(&patternspacemodule);
}
