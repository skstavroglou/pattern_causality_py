#include <Python.h>
#include <vector>
#include <numpy/arrayobject.h>
#include <limits>

static PyObject* stateSpace(PyObject* self, PyObject* args) {
    PyObject* ts_obj;
    int E, tau;
    
    // Parse Python arguments
    if (!PyArg_ParseTuple(args, "Oii", &ts_obj, &E, &tau)) {
        return NULL;
    }

    // Check if input is a list
    if (!PyList_Check(ts_obj)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be a list");
        return NULL;
    }

    // Check if E and tau are positive
    if (E <= 0 || tau <= 0) {
        PyErr_SetString(PyExc_ValueError, "E and tau must be positive integers");
        return NULL;
    }

    // Get length of input time series
    Py_ssize_t ts_len = PyList_Size(ts_obj);
    
    // Check if input list is long enough
    if (ts_len < E * tau) {
        PyErr_SetString(PyExc_ValueError, "Input list is too short for given E and tau");
        return NULL;
    }
    
    // Calculate dimensions of output matrix
    int rows = (ts_len - (E-1)*tau);
    int cols = E;
    
    // Create NumPy array
    npy_intp dims[2] = {rows, cols};
    PyObject* result_array = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (result_array == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create output array");
        return NULL;
    }
    
    double* data = (double*)PyArray_DATA((PyArrayObject*)result_array);
    
    // Fill the array
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < E; j++) {
            int idx = i + j * tau;
            PyObject* val = PyList_GetItem(ts_obj, idx);
            
            // Check if value is numeric
            if (!PyNumber_Check(val)) {
                Py_DECREF(result_array);
                PyErr_SetString(PyExc_TypeError, "All elements must be numbers");
                return NULL;
            }
            
            PyObject* float_val = PyNumber_Float(val);
            if (float_val == NULL) {
                Py_DECREF(result_array);
                return NULL;
            }
            
            data[i * cols + j] = PyFloat_AsDouble(float_val);
            Py_DECREF(float_val);
        }
    }
    
    return result_array;
}

// Method definition
static PyMethodDef StateSpaceMethods[] = {
    {"statespace", stateSpace, METH_VARARGS, "Compute state space matrix from time series"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef statespacemodule = {
    PyModuleDef_HEAD_INIT,
    "statespace",
    NULL,
    -1,
    StateSpaceMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_statespace(void) {
    import_array();
    PyObject* m = PyModule_Create(&statespacemodule);
    if (m == NULL) {
        return NULL;
    }
    return m;
}

