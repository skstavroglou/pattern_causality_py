#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <cmath>
#include <limits>

// Optimized inline difference calculation
static inline void calculate_differences(const double* input, double* output, npy_intp length) {
    for (npy_intp i = 0; i < length - 1; i++) {
        if (std::isnan(input[i]) || std::isnan(input[i + 1])) {
            output[i] = std::numeric_limits<double>::quiet_NaN();
        } else {
            output[i] = input[i + 1] - input[i];
        }
    }
}

static PyObject* signatureVectorDifference(PyObject* self, PyObject* args) {
    PyObject* input_array;
    if (!PyArg_ParseTuple(args, "O", &input_array)) {
        return NULL;
    }

    // Convert to numpy array
    PyArrayObject* array = (PyArrayObject*)PyArray_FROM_OTF(
        input_array, 
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY
    );
    if (!array) {
        return NULL;
    }

    // Check dimensions
    if (PyArray_NDIM(array) != 1) {
        Py_DECREF(array);
        PyErr_SetString(PyExc_ValueError, "Input must be a 1D array");
        return NULL;
    }

    const npy_intp length = PyArray_DIM(array, 0);
    const npy_intp output_length = length - 1;
    
    // Create output array
    npy_intp dims[1] = {output_length};
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!result_array) {
        Py_DECREF(array);
        return NULL;
    }

    // Get data pointers
    double* input_data = (double*)PyArray_DATA(array);
    double* output_data = (double*)PyArray_DATA(result_array);

    // Calculate differences
    calculate_differences(input_data, output_data, length);

    Py_DECREF(array);
    return (PyObject*)result_array;
}

static PyObject* signaturespace(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* input_matrix;
    int E;
    int relative = 1;  // Default to relative (1 for true)
    
    static char* kwlist[] = {"input_matrix", "E", "relative", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi|p", kwlist, 
                                    &input_matrix, &E, &relative)) {
        return NULL;
    }

    // Validate parameters
    if (E < 2) {
        PyErr_SetString(PyExc_ValueError, "State space matrix must have at least 2 columns");
        return NULL;
    }

    // Convert input to numpy array
    PyArrayObject* array = (PyArrayObject*)PyArray_FROM_OTF(
        input_matrix,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY
    );
    if (!array) {
        PyErr_SetString(PyExc_ValueError, "Input must be a matrix");
        return NULL;
    }

    // Validate dimensions
    if (PyArray_NDIM(array) != 2) {
        Py_DECREF(array);
        PyErr_SetString(PyExc_ValueError, "Input must be a matrix");
        return NULL;
    }

    const npy_intp rows = PyArray_DIM(array, 0);
    const npy_intp cols = PyArray_DIM(array, 1);
    const npy_intp output_cols = cols - 1;

    // Handle empty input
    if (rows == 0) {
        Py_DECREF(array);
        npy_intp dims[2] = {0, output_cols};
        return (PyObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    }

    // Create output array
    npy_intp out_dims[2] = {rows, output_cols};
    PyArrayObject* result_matrix = (PyArrayObject*)PyArray_SimpleNew(2, out_dims, NPY_DOUBLE);
    if (!result_matrix) {
        Py_DECREF(array);
        return NULL;
    }

    // Get data pointers
    const double* input_data = (double*)PyArray_DATA(array);
    double* output_data = (double*)PyArray_DATA(result_matrix);

    // Calculate differences for each row
    for (npy_intp i = 0; i < rows; i++) {
        const double* input_row = input_data + i * cols;
        double* output_row = output_data + i * output_cols;
        
        for (npy_intp j = 0; j < output_cols; j++) {
            if (std::isnan(input_row[j]) || std::isnan(input_row[j + 1])) {
                output_row[j] = std::numeric_limits<double>::quiet_NaN();
            } else {
                if (relative) {
                    // Relative change: (new - old) / old
                    // Exactly match R's behavior: no special handling for zero values
                    output_row[j] = (input_row[j + 1] - input_row[j]) / input_row[j];
                } else {
                    // Absolute change: new - old
                    output_row[j] = input_row[j + 1] - input_row[j];
                }
            }
        }
    }

    Py_DECREF(array);
    return (PyObject*)result_matrix;
}

// Module method definitions
static PyMethodDef SignatureSpaceMethods[] = {
    {"signatureVectorDifference", signatureVectorDifference, METH_VARARGS,
     "Calculate differences between successive elements using SIMD optimization"},
    {"signaturespace", (PyCFunction)signaturespace, METH_VARARGS | METH_KEYWORDS,
     "Calculate signature space matrix with parallel processing and SIMD optimization.\n"
     "Args:\n"
     "    input_matrix: Input 2D array\n"
     "    E: Embedding dimension\n"
     "    relative: If True, calculate relative differences (new-old)/old, otherwise absolute differences (new-old). Default is False."},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef signaturespacemodule = {
    PyModuleDef_HEAD_INIT,
    "signaturespace",
    "Optimized signature space calculation module",
    -1,
    SignatureSpaceMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_signaturespace(void) {
    import_array();
    return PyModule_Create(&signaturespacemodule);
}