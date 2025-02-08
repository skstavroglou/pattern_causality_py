#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <limits.h>

static PyObject* fcp(PyObject* self, PyObject* args) {
    int E, tau, h;
    PyObject* X;
    
    // Parse input arguments
    if (!PyArg_ParseTuple(args, "iiiO", &E, &tau, &h, &X)) {
        return NULL;
    }

    // Validate input types
    if (!PyList_Check(X) && !PyArray_Check(X)) {
        PyErr_SetString(PyExc_TypeError, "X must be a list or numpy array");
        return NULL;
    }

    // Validate input values
    if (E < 2) {
        PyErr_SetString(PyExc_ValueError, "E must be >= 2");
        return NULL;
    }
    if (tau < 1) {
        PyErr_SetString(PyExc_ValueError, "tau must be >= 1");
        return NULL;
    }
    if (h < 0) {
        PyErr_SetString(PyExc_ValueError, "h must be >= 0");
        return NULL;
    }

    // Get length of input
    Py_ssize_t X_len;
    if (PyList_Check(X)) {
        X_len = PyList_Size(X);
    } else {
        PyArrayObject* arr = (PyArrayObject*)X;
        X_len = PyArray_SIZE(arr);
    }

    if (X_len < 1) {
        PyErr_SetString(PyExc_ValueError, "Input X cannot be empty");
        return NULL;
    }

    // Calculate constants with overflow checking
    if (E > (INT_MAX - 1) || tau > INT_MAX / (E - 1)) {
        PyErr_SetString(PyExc_OverflowError, "Parameters too large");
        return NULL;
    }

    int NNSPAN = E + 1;  // Former NN | Reserves a minimum number of nearest neighbors
    int CCSPAN = (E - 1) * tau;  // This will remove the common coordinate NNs  
    int PredSPAN = h;
    
    // Check for integer overflow in final calculation
    if (NNSPAN > INT_MAX - CCSPAN || 
        NNSPAN + CCSPAN > INT_MAX - PredSPAN || 
        NNSPAN + CCSPAN + PredSPAN > INT_MAX - 1) {
        PyErr_SetString(PyExc_OverflowError, "Integer overflow in FCP calculation");
        return NULL;
    }
    
    int FCP = 1 + NNSPAN + CCSPAN + PredSPAN;

    // Validate sufficient data points
    if (NNSPAN + CCSPAN + PredSPAN >= X_len - CCSPAN) {
        PyErr_SetString(PyExc_ValueError, 
            "The First Point to consider for Causality does not have sufficient "
            "Nearest Neighbors. Please Check parameters: "
            "E, lag, p as well as the length of X and Y");
        return NULL;
    }

    return PyLong_FromLong((long)FCP);
}

static PyMethodDef FcpMethods[] = {
    {"fcp", fcp, METH_VARARGS, "Calculate first causality point"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fcpmodule = {
    PyModuleDef_HEAD_INIT,
    "utils.fcp",  // Changed back to "utils.fcp"
    "First causality point calculation module",
    -1,
    FcpMethods
};

PyMODINIT_FUNC PyInit_fcp(void) {
    import_array();  // Initialize NumPy
    
    PyObject* m = PyModule_Create(&fcpmodule);
    if (m == NULL) {
        return NULL;
    }
    
    return m;
}
