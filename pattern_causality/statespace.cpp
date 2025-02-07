#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <vector>
#include <numpy/arrayobject.h>
#include <limits>
#include <cmath>
#include <algorithm>
#include <memory>

// Thread-local storage for reusable buffers
thread_local std::vector<double> ts_buffer;

// Optimized conversion from Python object to double
static inline double convert_to_double(PyObject* item, bool& success) {
    if (PyFloat_Check(item)) {
        success = true;
        return PyFloat_AS_DOUBLE(item);
    } else if (PyLong_Check(item)) {
        success = true;
        return (double)PyLong_AsLongLong(item);
    } else {
        PyObject* float_obj = PyNumber_Float(item);
        if (!float_obj) {
            success = false;
            return 0.0;
        }
        double result = PyFloat_AS_DOUBLE(float_obj);
        Py_DECREF(float_obj);
        success = true;
        return result;
    }
}

// Fast check for numpy array contiguity and type
static inline bool check_array_valid(PyArrayObject* arr) {
    return (PyArray_ISCARRAY_RO(arr) && 
            (PyArray_TYPE(arr) == NPY_DOUBLE || 
             PyArray_TYPE(arr) == NPY_FLOAT ||
             PyArray_TYPE(arr) == NPY_INT64 ||
             PyArray_TYPE(arr) == NPY_INT32));
}

static PyObject* stateSpace(PyObject* self, PyObject* args) {
    PyObject* ts_obj;
    int E, tau;
    
    // Parse Python arguments
    if (!PyArg_ParseTuple(args, "Oii", &ts_obj, &E, &tau)) {
        return NULL;
    }

    // Quick parameter validation
    if (E < 2 || tau < 1) {
        PyErr_SetString(PyExc_ValueError, "E must be >= 2 and tau must be >= 1");
        return NULL;
    }

    // Get input type and length
    const bool is_list = PyList_Check(ts_obj);
    const bool is_array = PyArray_Check(ts_obj);
    if (!is_list && !is_array) {
        PyErr_SetString(PyExc_TypeError, "Input must be a list or numpy array");
        return NULL;
    }

    // Get length of input time series
    const Py_ssize_t ts_len = is_list ? PyList_Size(ts_obj) : PyArray_SIZE((PyArrayObject*)ts_obj);

    // Check minimum length requirement
    if (ts_len < (E - 1) * tau + 1) {
        PyErr_SetString(PyExc_ValueError, "Time series too short for given E and tau");
        return NULL;
    }

    // Calculate output dimensions
    const npy_intp rows = ts_len - (E - 1) * tau;
    const npy_intp cols = E;
    npy_intp dims[2] = {rows, cols};

    // Create output array with alignment
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!result_array) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create output array");
        return NULL;
    }

    // Resize thread-local buffer if needed
    if (static_cast<Py_ssize_t>(ts_buffer.size()) < ts_len) {
        ts_buffer.resize(static_cast<size_t>(ts_len));
    }

    // Get data pointers
    double* const data = (double*)PyArray_DATA(result_array);
    double* const ts_data = ts_buffer.data();

    // Convert input to double array using the most efficient method
    if (is_list) {
        #pragma omp parallel for schedule(static)
        for (Py_ssize_t i = 0; i < ts_len; i++) {
            PyObject* item = PyList_GET_ITEM(ts_obj, i);
            bool success = true;
            ts_data[i] = convert_to_double(item, success);
            if (!success) {
                PyErr_SetString(PyExc_TypeError, "All elements must be numeric");
                // Note: Cannot return NULL here due to OpenMP, error will be checked later
            }
        }
        if (PyErr_Occurred()) {
            Py_DECREF(result_array);
            return NULL;
        }
    } else {
        PyArrayObject* arr = (PyArrayObject*)ts_obj;
        if (!check_array_valid(arr)) {
            arr = (PyArrayObject*)PyArray_FROM_OTF(ts_obj, NPY_DOUBLE, 
                NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ALIGNED | NPY_ARRAY_FORCECAST);
            if (!arr) {
                Py_DECREF(result_array);
                return NULL;
            }
            memcpy(ts_data, PyArray_DATA(arr), ts_len * sizeof(double));
            Py_DECREF(arr);
        } else {
            memcpy(ts_data, PyArray_DATA(arr), ts_len * sizeof(double));
        }
    }

    // Fill state space matrix using optimized parallel processing
    const npy_intp block_size = std::max<npy_intp>(1, 1024 / E); // Optimize cache usage
    #pragma omp parallel
    {
        #pragma omp for schedule(static) collapse(2)
        for (npy_intp i = 0; i < rows; i += block_size) {
            for (npy_intp j = 0; j < E; j++) {
                const npy_intp block_end = std::min<npy_intp>(i + block_size, rows);
                #pragma omp simd
                for (npy_intp k = i; k < block_end; k++) {
                    const npy_intp idx = k + j * tau;
                    const double val = ts_data[idx];
                    data[k * E + j] = std::isfinite(val) ? val : std::numeric_limits<double>::quiet_NaN();
                }
            }
        }
    }

    return (PyObject*)result_array;
}

// Method definition
static PyMethodDef StateSpaceMethods[] = {
    {"statespace", stateSpace, METH_VARARGS, 
     "Create state space matrix from time series using embedding parameters E and tau"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef statespacemodule = {
    PyModuleDef_HEAD_INIT,
    "statespace",
    "Optimized state space embedding module",
    -1,
    StateSpaceMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_statespace(void) {
    import_array();
    return PyModule_Create(&statespacemodule);
}

