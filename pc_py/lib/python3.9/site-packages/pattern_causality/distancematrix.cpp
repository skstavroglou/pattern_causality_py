#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <cmath>
#include <string>

// Helper functions for different distance metrics
static double euclideanDistance(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    double sum = 0.0;
    for (size_t i = 0; i < vec1.size(); i++) {
        double diff = vec1[i] - vec2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

static double manhattanDistance(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    double sum = 0.0;
    for (size_t i = 0; i < vec1.size(); i++) {
        sum += fabs(vec1[i] - vec2[i]);
    }
    return sum;
}

static double minkowskiDistance(const std::vector<double>& vec1, const std::vector<double>& vec2, int n) {
    double sum = 0.0;
    for (size_t i = 0; i < vec1.size(); i++) {
        sum += pow(fabs(vec1[i] - vec2[i]), n);
    }
    return pow(sum, 1.0/n);
}

static double calculateDistance(const std::vector<double>& vec1, const std::vector<double>& vec2, 
                              const std::string& metric, int n = 2) {
    if (metric == "euclidean") {
        return euclideanDistance(vec1, vec2);
    } else if (metric == "manhattan") {
        return manhattanDistance(vec1, vec2);
    } else if (metric == "minkowski") {
        return minkowskiDistance(vec1, vec2, n);
    }
    // Default to Euclidean
    return euclideanDistance(vec1, vec2);
}

static PyObject* distanceVector(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* point_obj;
    PyObject* candidates_obj;
    const char* metric_str = "euclidean";
    int n = 2;
    
    static char* kwlist[] = {"point", "candidates", "metric", "n", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|si", kwlist, 
                                    &point_obj, &candidates_obj, &metric_str, &n)) {
        return NULL;
    }

    PyArrayObject* point_array = (PyArrayObject*)PyArray_FROM_OTF(point_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* candidates_array = (PyArrayObject*)PyArray_FROM_OTF(candidates_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    
    if (point_array == NULL || candidates_array == NULL) {
        Py_XDECREF(point_array);
        Py_XDECREF(candidates_array);
        PyErr_SetString(PyExc_TypeError, "Could not convert input to numpy array");
        return NULL;
    }

    if (PyArray_NDIM(point_array) != 1 || PyArray_NDIM(candidates_array) != 2) {
        Py_DECREF(point_array);
        Py_DECREF(candidates_array);
        PyErr_SetString(PyExc_ValueError, "Invalid input dimensions");
        return NULL;
    }

    npy_intp point_size = PyArray_DIM(point_array, 0);
    npy_intp num_candidates = PyArray_DIM(candidates_array, 0);
    npy_intp vec_size = PyArray_DIM(candidates_array, 1);

    if (point_size != vec_size) {
        Py_DECREF(point_array);
        Py_DECREF(candidates_array);
        PyErr_SetString(PyExc_ValueError, "Dimension mismatch");
        return NULL;
    }

    npy_intp dims[1] = {num_candidates};
    PyObject* result = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (result == NULL) {
        Py_DECREF(point_array);
        Py_DECREF(candidates_array);
        return NULL;
    }

    double* point_data = (double*)PyArray_DATA(point_array);
    double* candidates_data = (double*)PyArray_DATA(candidates_array);
    double* result_data = (double*)PyArray_DATA((PyArrayObject*)result);

    std::string metric(metric_str);
    std::vector<double> point(point_data, point_data + point_size);

    for (npy_intp i = 0; i < num_candidates; i++) {
        std::vector<double> vec(candidates_data + i * vec_size, 
                              candidates_data + (i + 1) * vec_size);
        result_data[i] = calculateDistance(point, vec, metric, n);
    }

    Py_DECREF(point_array);
    Py_DECREF(candidates_array);

    return result;
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
        std::vector<double> vec1(matrix_data + i * vec_size, 
                               matrix_data + (i + 1) * vec_size);
        
        for (npy_intp j = 0; j < num_rows; j++) {
            std::vector<double> vec2(matrix_data + j * vec_size, 
                                   matrix_data + (j + 1) * vec_size);
            
            result_data[i * num_rows + j] = calculateDistance(vec1, vec2, metric, n);
        }
    }

    Py_DECREF(matrix_array);

    return result_matrix;
}

static PyMethodDef DistanceMatrixMethods[] = {
    {"distancevector", (PyCFunction)distanceVector, METH_VARARGS | METH_KEYWORDS, 
     "Calculate distances between a point and multiple candidates"},
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