#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>
#include <string>

// Helper functions for different distance metrics
static inline double euclideanDistance(const double* vec1, const double* vec2, size_t size) {
    double sum = 0.0;
    for (size_t i = 0; i < size; i++) {
        double diff = vec1[i] - vec2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

static inline double manhattanDistance(const double* vec1, const double* vec2, size_t size) {
    double sum = 0.0;
    for (size_t i = 0; i < size; i++) {
        sum += fabs(vec1[i] - vec2[i]);
    }
    return sum;
}

static inline double minkowskiDistance(const double* vec1, const double* vec2, size_t size, int n) {
    double sum = 0.0;
    for (size_t i = 0; i < size; i++) {
        sum += pow(fabs(vec1[i] - vec2[i]), n);
    }
    return pow(sum, 1.0/n);
}

static inline double calculateDistance(const double* vec1, const double* vec2, size_t size,
                                       const std::string& metric, int n = 2) {
    if (metric == "euclidean") {
        return euclideanDistance(vec1, vec2, size);
    } else if (metric == "manhattan") {
        return manhattanDistance(vec1, vec2, size);
    } else if (metric == "minkowski") {
        return minkowskiDistance(vec1, vec2, size, n);
    }
    return euclideanDistance(vec1, vec2, size);
}

static PyObject* distanceMatrix(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* matrix_obj;
    const char* metric_str = "euclidean";
    int n = 2;
    
    static char* kwlist[] = {"matrix", "metric", "n", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|si", kwlist, 
                                    &matrix_obj, &metric_str, &n)) {
        return NULL;
    }

    PyArrayObject* matrix_array = (PyArrayObject*)PyArray_FROM_OTF(matrix_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (matrix_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Could not convert input to numpy array");
        return NULL;
    }

    if (PyArray_NDIM(matrix_array) != 2) {
        Py_DECREF(matrix_array);
        PyErr_SetString(PyExc_ValueError, "Input must be a 2D array");
        return NULL;
    }

    npy_intp num_rows = PyArray_DIM(matrix_array, 0);
    npy_intp vec_size = PyArray_DIM(matrix_array, 1);

    npy_intp dims[2] = {num_rows, num_rows};
    PyObject* result_matrix = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (result_matrix == NULL) {
        Py_DECREF(matrix_array);
        return NULL;
    }

    double* matrix_data = (double*)PyArray_DATA(matrix_array);
    double* result_data = (double*)PyArray_DATA((PyArrayObject*)result_matrix);

    std::string metric(metric_str);

    for (npy_intp i = 0; i < num_rows; i++) {
        const double* vec1 = matrix_data + i * vec_size;
        result_data[i * num_rows + i] = 0.0;
        
        for (npy_intp j = i + 1; j < num_rows; j++) {
            const double* vec2 = matrix_data + j * vec_size;
            double dist = calculateDistance(vec1, vec2, vec_size, metric, n);
            
            result_data[i * num_rows + j] = dist;
            result_data[j * num_rows + i] = dist;
        }
    }

    Py_DECREF(matrix_array);

    return result_matrix;
}

static PyMethodDef DistanceMatrixMethods[] = {
    {"distancematrix", (PyCFunction)distanceMatrix, METH_VARARGS | METH_KEYWORDS, 
     "Calculate distance matrix for a set of vectors"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef distancematrixmodule = {
    PyModuleDef_HEAD_INIT,
    "distancematrix",
    "Distance calculation module",
    -1,
    DistanceMatrixMethods
};

PyMODINIT_FUNC PyInit_distancematrix(void) {
    import_array();
    return PyModule_Create(&distancematrixmodule);
}