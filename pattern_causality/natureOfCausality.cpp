#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <limits>

static PyObject* natureOfCausality(PyObject* self, PyObject* args) {
    PyArrayObject *PC, *dur, *hashedpatterns, *X;
    PyObject* weighted_obj;

    // Parse input arguments
    if (!PyArg_ParseTuple(args, "OOOOO", &PC, &dur, &hashedpatterns, &X, &weighted_obj)) {
        return NULL;
    }

    bool weighted = PyObject_IsTrue(weighted_obj);

    // Convert inputs to numpy arrays with optimization flags
    PyArrayObject* pc_arr = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)PC, NPY_DOUBLE, 
                           NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ALIGNED);
    PyArrayObject* dur_arr = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)dur, NPY_LONG, 
                           NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ALIGNED);
    PyArrayObject* hashed_arr = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)hashedpatterns, NPY_DOUBLE, 
                               NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ALIGNED);
    PyArrayObject* x_arr = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)X, NPY_DOUBLE, 
                          NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ALIGNED);

    if (!pc_arr || !dur_arr || !hashed_arr || !x_arr) {
        Py_XDECREF(pc_arr);
        Py_XDECREF(dur_arr);
        Py_XDECREF(hashed_arr);
        Py_XDECREF(x_arr);
        return NULL;
    }

    // Get array dimensions
    npy_intp x_size = PyArray_SIZE(x_arr);
    npy_intp* pc_dims = PyArray_DIMS(pc_arr);
    
    // Pre-calculate array strides for faster access
    npy_intp pc_stride_row = pc_dims[1] * pc_dims[2];
    npy_intp pc_stride_col = pc_dims[2];

    // Create output arrays initialized with NaN
    npy_intp dims[1] = {x_size};
    PyArrayObject* positive_causality = (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    PyArrayObject* negative_causality = (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    PyArrayObject* dark_causality = (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    PyArrayObject* no_causality = (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);

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
    
    // Initialize with NaN
    const double nan_value = std::numeric_limits<double>::quiet_NaN();
    for(npy_intp i = 0; i < x_size; i++) {
        pos_data[i] = neg_data[i] = dark_data[i] = no_data[i] = nan_value;
    }
    
    npy_intp dur_size = PyArray_SIZE(dur_arr);
    npy_intp hashed_size = PyArray_SIZE(hashed_arr);
    npy_intp mean_pattern = hashed_size / 2;  // Calculate mean pattern index

    // Main computation loop
    #pragma omp parallel for if(dur_size > 1000) // Enable OpenMP for large arrays
    for (npy_intp d = 0; d < dur_size; d++) {
        long i = dur_data[d];
        
        // Find first non-NaN cell in PC[:,:,i]
        bool found = false;
        npy_intp cell_row = 0, cell_col = 0;
        double pc_val = 0.0;
        
        // Optimized search for first non-NaN value
        for (npy_intp row = 0; row < pc_dims[0] && !found; row++) {
            const npy_intp row_offset = row * pc_stride_row;
            for (npy_intp col = 0; col < pc_dims[1] && !found; col++) {
                const double value = pc_data[row_offset + col * pc_stride_col + i];
                if (!isnan(value)) {
                    cell_row = row;
                    cell_col = col;
                    pc_val = value;
                    found = true;
                }
            }
        }
        
        if (found) {
            // Positive causality
            if (cell_row == cell_col) {
                if (cell_row != mean_pattern) {
                    if (pc_val == 0) {
                        no_data[i] = 1;
                        pos_data[i] = 0;
                    } else {
                        no_data[i] = 0;
                        pos_data[i] = weighted ? pc_val : 1;
                    }
                    neg_data[i] = 0;
                    dark_data[i] = 0;
                } else {  // Center of PC matrix
                    if (pc_val == 0) {
                        no_data[i] = 1;
                        dark_data[i] = 0;
                    } else {
                        no_data[i] = 0;
                        dark_data[i] = weighted ? pc_val : 1;
                    }
                    neg_data[i] = 0;
                    pos_data[i] = 0;
                }
            }
            // Negative causality
            else if ((cell_row + cell_col) == (hashed_size - 1)) {
                if (pc_val == 0) {
                    no_data[i] = 1;
                    neg_data[i] = 0;
                } else {
                    no_data[i] = 0;
                    neg_data[i] = weighted ? pc_val : 1;
                }
                pos_data[i] = 0;
                dark_data[i] = 0;
            }
            // Dark causality
            else {
                if (pc_val == 0) {
                    no_data[i] = 1;
                    dark_data[i] = 0;
                } else {
                    no_data[i] = 0;
                    dark_data[i] = weighted ? pc_val : 1;
                }
                neg_data[i] = 0;
                pos_data[i] = 0;
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
