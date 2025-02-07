#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>

// Helper function for factorial calculation - must match R implementation exactly
static double factorial(int n) {
    if (n <= 1) return 1.0;
    double result = 1.0;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

// Helper function for hashing - must match R implementation exactly
static double hashing(const std::vector<int>& vec) {
    double hash = 0.0;
    for (size_t i = 0; i < vec.size(); i++) {
        hash += static_cast<double>(vec[i]) * factorial(i + 2);
    }
    return hash;
}

// Pre-allocated vectors to avoid repeated allocation
thread_local std::vector<double> result_buffer;
thread_local std::vector<int> p_vec_buffer;

static double pattern_vector_difference(const std::vector<double>& sVec) {
    // Quick check for NaN
    for (const auto& val : sVec) {
        if (std::isnan(val)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }
    
    // Reuse pre-allocated vector
    if (p_vec_buffer.capacity() < sVec.size()) {
        p_vec_buffer.reserve(sVec.size());
    }
    p_vec_buffer.clear();
    
    const double eps = std::numeric_limits<double>::epsilon();
    
    // Pattern calculation - must match R implementation exactly
    for (const auto& val : sVec) {
        if (std::abs(val) < eps) {
            p_vec_buffer.push_back(2);  // zero
        } else if (val > 0) {
            p_vec_buffer.push_back(3);  // positive
        } else {
            p_vec_buffer.push_back(1);  // negative
        }
    }
    
    return hashing(p_vec_buffer);
}

static PyObject* patternspace(PyObject* self, PyObject* args) {
    PyObject* sm_obj;
    int E;
    
    if (!PyArg_ParseTuple(args, "Oi", &sm_obj, &E)) {
        return NULL;
    }
    
    PyArrayObject* sm_array = (PyArrayObject*)PyArray_FROM_OTF(sm_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!sm_array || PyArray_NDIM(sm_array) != 2) {
        Py_XDECREF(sm_array);
        PyErr_SetString(PyExc_ValueError, "Input must be a 2D array");
        return NULL;
    }
    
    npy_intp* dims = PyArray_DIMS(sm_array);
    npy_intp num_rows = dims[0];
    npy_intp num_cols = dims[1];
    
    npy_intp out_dims[2] = {num_rows, 1};
    PyObject* result = PyArray_SimpleNew(2, out_dims, NPY_DOUBLE);
    if (!result) {
        Py_DECREF(sm_array);
        return NULL;
    }
    
    double* sm_data = (double*)PyArray_DATA(sm_array);
    double* result_data = (double*)PyArray_DATA((PyArrayObject*)result);
    
    // Pre-allocate vector for row data
    std::vector<double> row_buffer(num_cols);
    
    // Process each row sequentially to ensure consistent results
    for (npy_intp i = 0; i < num_rows; i++) {
        // Copy row data
        std::copy(sm_data + i * num_cols, sm_data + (i + 1) * num_cols, row_buffer.begin());
        result_data[i] = pattern_vector_difference(row_buffer);
    }
    
    Py_DECREF(sm_array);
    return result;
}

static PyMethodDef PatternSpaceMethods[] = {
    {"patternspace", patternspace, METH_VARARGS, "Calculate pattern space matrix from signature matrix"},
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
