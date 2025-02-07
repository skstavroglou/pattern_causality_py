#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>
#include <limits>

static PyObject* natureOfCausality(PyObject* self, PyObject* args) {
    PyObject *pc_obj, *dur_obj, *hashed_obj, *x_obj;
    PyObject* weighted_obj;
    
    if (!PyArg_ParseTuple(args, "OOOOO", &pc_obj, &dur_obj, &hashed_obj, &x_obj, &weighted_obj)) {
        return NULL;
    }
    
    // Convert inputs to numpy arrays
    PyArrayObject* pc_arr = (PyArrayObject*)PyArray_FROM_OTF(pc_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* dur_arr = (PyArrayObject*)PyArray_FROM_OTF(dur_obj, NPY_LONG, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* hashed_arr = (PyArrayObject*)PyArray_FROM_OTF(hashed_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    
    if (!pc_arr || !dur_arr || !hashed_arr || !x_arr) {
        Py_XDECREF(pc_arr);
        Py_XDECREF(dur_arr);
        Py_XDECREF(hashed_arr);
        Py_XDECREF(x_arr);
        return NULL;
    }
    
    const bool weighted = PyObject_IsTrue(weighted_obj);
    
    // Get array dimensions
    const npy_intp* pc_dims = PyArray_DIMS(pc_arr);
    const npy_intp pc_stride_row = PyArray_STRIDE(pc_arr, 0) / sizeof(double);
    const npy_intp pc_stride_col = PyArray_STRIDE(pc_arr, 1) / sizeof(double);
    const npy_intp x_size = PyArray_SIZE(x_arr);
    
    // Create output arrays
    npy_intp dims[] = {x_size};
    PyArrayObject* positive_causality = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyArrayObject* negative_causality = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyArrayObject* dark_causality = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyArrayObject* no_causality = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    
    if (!positive_causality || !negative_causality || !dark_causality || !no_causality) {
        Py_XDECREF(pc_arr);
        Py_XDECREF(dur_arr);
        Py_XDECREF(hashed_arr);
        Py_XDECREF(x_arr);
        Py_XDECREF(positive_causality);
        Py_XDECREF(negative_causality);
        Py_XDECREF(dark_causality);
        Py_XDECREF(no_causality);
        return NULL;
    }
    
    // Get data pointers for direct memory access
    double* pos_data = (double*)PyArray_DATA(positive_causality);
    double* neg_data = (double*)PyArray_DATA(negative_causality);
    double* dark_data = (double*)PyArray_DATA(dark_causality);
    double* no_data = (double*)PyArray_DATA(no_causality);
    double* pc_data = (double*)PyArray_DATA(pc_arr);
    long* dur_data = (long*)PyArray_DATA(dur_arr);
    
    // Initialize all arrays with NaN
    const double nan_value = std::numeric_limits<double>::quiet_NaN();
    for(npy_intp i = 0; i < x_size; i++) {
        pos_data[i] = neg_data[i] = dark_data[i] = no_data[i] = nan_value;
    }
    
    const npy_intp dur_size = PyArray_SIZE(dur_arr);
    const npy_intp hashed_size = PyArray_SIZE(hashed_arr);
    const npy_intp mean_pattern = hashed_size / 2;
    const double eps = std::numeric_limits<double>::epsilon();
    
    // Main computation loop
    for (npy_intp d = 0; d < dur_size; d++) {
        const long i = dur_data[d];
        
        bool found_valid = false;
        bool has_causality = false;
        double pos_val = 0.0;
        double neg_val = 0.0;
        double dark_val = 0.0;
        int valid_count = 0;
        
        // First pass: check if we have any valid values and count total non-NaN values
        for (npy_intp row = 0; row < pc_dims[0]; row++) {
            for (npy_intp col = 0; col < pc_dims[1]; col++) {
                const double pc_val = pc_data[row * pc_stride_row + col * pc_stride_col + i];
                if (!std::isnan(pc_val)) {
                    found_valid = true;
                    valid_count++;
                }
            }
        }
        
        // Only proceed with causality calculation if we found valid values
        if (found_valid) {
            // Second pass: calculate causalities
            for (npy_intp row = 0; row < pc_dims[0]; row++) {
                for (npy_intp col = 0; col < pc_dims[1]; col++) {
                    const double pc_val = pc_data[row * pc_stride_row + col * pc_stride_col + i];
                    
                    if (!std::isnan(pc_val) && std::abs(pc_val) > eps) {
                        has_causality = true;
                        
                        // Center diagonal element contributes to dark causality
                        if (row == col && row == mean_pattern) {
                            dark_val += weighted ? pc_val : 1.0;
                        }
                        // Other diagonal elements contribute to positive causality
                        else if (row == col) {
                            pos_val += weighted ? pc_val : 1.0;
                        }
                        // Anti-diagonal elements contribute to negative causality
                        else if (row + col == hashed_size - 1) {
                            neg_val += weighted ? pc_val : 1.0;
                        }
                        // All other elements contribute to dark causality
                        else {
                            dark_val += weighted ? pc_val : 1.0;
                        }
                    }
                }
            }
            
            // Set values only if we found valid data
            if (valid_count > 0) {
                no_data[i] = has_causality ? 0.0 : 1.0;
                pos_data[i] = pos_val;
                neg_data[i] = neg_val;
                dark_data[i] = dark_val;
            }
        }
    }
    
    // Create return dictionary
    PyObject* result = PyDict_New();
    if (!result) {
        Py_XDECREF(pc_arr);
        Py_XDECREF(dur_arr);
        Py_XDECREF(hashed_arr);
        Py_XDECREF(x_arr);
        Py_XDECREF(positive_causality);
        Py_XDECREF(negative_causality);
        Py_XDECREF(dark_causality);
        Py_XDECREF(no_causality);
        return NULL;
    }
    
    PyDict_SetItemString(result, "noCausality", (PyObject*)no_causality);
    PyDict_SetItemString(result, "Positive", (PyObject*)positive_causality);
    PyDict_SetItemString(result, "Negative", (PyObject*)negative_causality);
    PyDict_SetItemString(result, "Dark", (PyObject*)dark_causality);
    
    Py_DECREF(pc_arr);
    Py_DECREF(dur_arr);
    Py_DECREF(hashed_arr);
    Py_DECREF(x_arr);
    Py_DECREF(positive_causality);
    Py_DECREF(negative_causality);
    Py_DECREF(dark_causality);
    Py_DECREF(no_causality);
    
    return result;
}

static PyMethodDef NatureOfCausalityMethods[] = {
    {"natureOfCausality", natureOfCausality, METH_VARARGS,
     "Calculate nature of causality from PC matrix"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef natureOfCausalitymodule = {
    PyModuleDef_HEAD_INIT,
    "natureOfCausality",
    NULL,
    -1,
    NatureOfCausalityMethods
};

PyMODINIT_FUNC PyInit_natureOfCausality(void) {
    import_array();
    return PyModule_Create(&natureOfCausalitymodule);
}
