#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>

static double norm_vec(PyObject* x) {
    PyArrayObject* arr = (PyArrayObject*)PyArray_FROM_OTF(x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!arr) {
        return 0.0;
    }
    
    double sum = 0.0;
    double* data = (double*)PyArray_DATA(arr);
    npy_intp size = PyArray_SIZE(arr);
    
    for(npy_intp i = 0; i < size; i++) {
        sum += data[i] * data[i];
    }
    
    Py_DECREF(arr);
    return sqrt(sum);
}

static PyObject* fillPCMatrix(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *predictedPatternY_obj, *realPatternY_obj, *predictedSignatureY_obj;
    PyObject *realSignatureY_obj, *patternX_obj, *signatureX_obj;
    PyObject* weighted_obj;

    static char* kwlist[] = {
        "weighted", "predictedPatternY", "realPatternY",
        "predictedSignatureY", "realSignatureY",
        "patternX", "signatureX", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOOOO", kwlist,
                                    &weighted_obj,
                                    &predictedPatternY_obj, &realPatternY_obj,
                                    &predictedSignatureY_obj, &realSignatureY_obj,
                                    &patternX_obj, &signatureX_obj)) {
        return NULL;
    }

    bool weighted = PyObject_IsTrue(weighted_obj);

    // Convert inputs to numpy arrays if they aren't already
    PyArrayObject* pred_pattern_arr = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)predictedPatternY_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* real_pattern_arr = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)realPatternY_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* pattern_x_arr = (PyArrayObject*)PyArray_FROM_OTF((PyObject*)patternX_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    
    if (!pred_pattern_arr || !real_pattern_arr || !pattern_x_arr) {
        Py_XDECREF(pred_pattern_arr);
        Py_XDECREF(real_pattern_arr);
        Py_XDECREF(pattern_x_arr);
        PyErr_SetString(PyExc_TypeError, "Could not convert input to numpy array");
        return NULL;
    }

    // Check for NaN values
    double *pred_pattern = (double*)PyArray_DATA(pred_pattern_arr);
    double *real_pattern = (double*)PyArray_DATA(real_pattern_arr);
    double *pattern_x = (double*)PyArray_DATA(pattern_x_arr);
    
    npy_intp size_pred = PyArray_SIZE(pred_pattern_arr);
    npy_intp size_real = PyArray_SIZE(real_pattern_arr);
    npy_intp size_x = PyArray_SIZE(pattern_x_arr);

    // First check if input vectors have length > 0
    if (size_pred == 0) {
        PyErr_SetString(PyExc_ValueError, "The length of the predicted pattern of Y is ZERO");
        return NULL;
    }
    if (size_x == 0) {
        PyErr_SetString(PyExc_ValueError, "The length of the causal pattern of X is ZERO");
        return NULL;
    }

    // Check for NaN values - combine all checks into one loop
    bool has_nan = false;
    for(npy_intp i = 0; i < size_pred; i++) {
        if(std::isnan(pred_pattern[i])) {
            has_nan = true;
            break;
        }
    }
    for(npy_intp i = 0; i < size_real && !has_nan; i++) {
        if(std::isnan(real_pattern[i])) {
            has_nan = true;
            break;
        }
    }
    for(npy_intp i = 0; i < size_x && !has_nan; i++) {
        if(std::isnan(pattern_x[i])) {
            has_nan = true;
            break;
        }
    }

    if (has_nan) {
        Py_DECREF(pred_pattern_arr);
        Py_DECREF(real_pattern_arr);
        Py_DECREF(pattern_x_arr);
        return Py_BuildValue("{s:d,s:d}", "real", NAN, "predicted", NAN);
    }

    double predictedCausalityStrength, realCausalityStrength;

    // Check if patterns are equal
    bool patterns_equal = true;
    if(size_pred == size_real) {
        for(npy_intp i = 0; i < size_pred; i++) {
            if(pred_pattern[i] != real_pattern[i]) {
                patterns_equal = false;
                break;
            }
        }
    } else {
        patterns_equal = false;
    }

    if(patterns_equal) {
        if(weighted) {
            // For weighted case, calculate using norm ratios
            double pred_ratio = norm_vec(predictedSignatureY_obj) / norm_vec(signatureX_obj);
            double real_ratio = norm_vec(realSignatureY_obj) / norm_vec(signatureX_obj);
            
            predictedCausalityStrength = std::erf(pred_ratio);
            realCausalityStrength = std::erf(real_ratio);
        } else {
            // For unweighted case, just return 1.0 when patterns match
            predictedCausalityStrength = 1.0;
            realCausalityStrength = 1.0;
        }
    } else {
        predictedCausalityStrength = 0.0;
        realCausalityStrength = 0.0;
    }

    // Clean up
    Py_DECREF(pred_pattern_arr);
    Py_DECREF(real_pattern_arr);
    Py_DECREF(pattern_x_arr);

    // Return results as a dictionary
    return Py_BuildValue("{s:d,s:d}", 
                        "real", realCausalityStrength, 
                        "predicted", predictedCausalityStrength);
}

static PyMethodDef FillPCMatrixMethods[] = {
    {"fillPCMatrix", (PyCFunction)fillPCMatrix, METH_VARARGS | METH_KEYWORDS,
     "Fill pattern causality matrix with causality strengths"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fillpcmatrixmodule = {
    PyModuleDef_HEAD_INIT,
    "fillPCMatrix",
    "Fill pattern causality matrix module",
    -1,
    FillPCMatrixMethods
};

PyMODINIT_FUNC PyInit_fillPCMatrix(void) {
    import_array();
    return PyModule_Create(&fillpcmatrixmodule);
}