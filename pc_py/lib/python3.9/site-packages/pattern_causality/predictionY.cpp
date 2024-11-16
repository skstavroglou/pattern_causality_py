#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <cmath>


static double hashing(const std::vector<int>& vec) {
    double hash_val = 0.0;
    for (size_t i = 0; i < vec.size(); i++) {
        hash_val += vec[i] * std::tgamma(i + 4);
    }
    return hash_val;
}

static PyObject* calculate_pattern(double value) {
    npy_intp dims[1] = {1};
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!result_array) return NULL;

    double* output_data = (double*)PyArray_DATA(result_array);
    std::vector<int> p_vec;
    
    if (value > 0) {
        p_vec.push_back(3);
    } else if (value < 0) {
        p_vec.push_back(1);
    } else {
        p_vec.push_back(2);
    }
    
    output_data[0] = hashing(p_vec);
    return (PyObject*)result_array;
}

static PyObject* calculate_pattern_array(PyArrayObject* array) {
    npy_intp dims[1] = {1};
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!result_array) return NULL;

    double* input_data = (double*)PyArray_DATA(array);
    double* output_data = (double*)PyArray_DATA(result_array);
    npy_intp length = PyArray_SIZE(array);

    std::vector<int> p_vec;
    for (npy_intp i = 0; i < length; i++) {
        if (input_data[i] > 0) {
            p_vec.push_back(3);
        } else if (input_data[i] < 0) {
            p_vec.push_back(1);
        } else {
            p_vec.push_back(2);
        }
    }

    output_data[0] = hashing(p_vec);
    return (PyObject*)result_array;
}

static PyObject* predictionY(PyObject* self, PyObject* args, PyObject* kwargs) {
    long E;
    PyObject* projNNy;
    PyObject* weighted_obj;
    PyObject* zeroTolerance_obj = Py_None;
    
    static char* kwlist[] = {"E", "projNNy", "weighted", "zeroTolerance", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "lOO|O", kwlist, 
                                    &E, &projNNy, &weighted_obj, &zeroTolerance_obj)) {
        return NULL;
    }

    bool weighted = PyObject_IsTrue(weighted_obj);

    double zeroTolerance;
    if (zeroTolerance_obj == Py_None) {
        zeroTolerance = (E + 1.0) / 2.0;
    } else {
        zeroTolerance = PyFloat_AsDouble(zeroTolerance_obj);
        if (PyErr_Occurred()) return NULL;
    }

    PyObject* signatures = PyDict_GetItemString(projNNy, "signatures");
    PyObject* weights = PyDict_GetItemString(projNNy, "weights");
    
    if (!signatures || !weights) {
        PyErr_SetString(PyExc_KeyError, "projNNy must contain 'signatures' and 'weights' keys");
        return NULL;
    }

    PyArrayObject* signatures_array = (PyArrayObject*)PyArray_FROM_OTF(signatures, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* weights_array = (PyArrayObject*)PyArray_FROM_OTF(weights, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    
    if (!signatures_array || !weights_array) {
        Py_XDECREF(signatures_array);
        Py_XDECREF(weights_array);
        PyErr_SetString(PyExc_TypeError, "Failed to convert signatures or weights to numpy array");
        return NULL;
    }

    PyObject* predictedSignatureY = NULL;
    
    if (E >= 3) {
        npy_intp dims[] = {E - 1};
        predictedSignatureY = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
        if (!predictedSignatureY) {
            Py_DECREF(signatures_array);
            Py_DECREF(weights_array);
            return NULL;
        }

        double* predSig_data = (double*)PyArray_DATA((PyArrayObject*)predictedSignatureY);
        double* sig_data = (double*)PyArray_DATA(signatures_array);
        double* weights_data = (double*)PyArray_DATA(weights_array);
        npy_intp* sig_dims = PyArray_DIMS(signatures_array);
        
        for(long part = 0; part < E - 1; part++) {
            double zeros_count = 0;
            for(npy_intp i = 0; i < sig_dims[0]; i++) {
                if(fabs(sig_data[i * sig_dims[1] + part]) < 1e-10) {
                    zeros_count += 1.0;
                }
            }
            
            if(zeros_count > zeroTolerance) {
                predSig_data[part] = 0.0;
            } else {
                double sum = 0.0;
                if(weighted) {
                    // Use weighted calculation
                    for(npy_intp i = 0; i < sig_dims[0]; i++) {
                        sum += sig_data[i * sig_dims[1] + part] * weights_data[i];
                    }
                } else {
                    // Use unweighted calculation - simple average
                    for(npy_intp i = 0; i < sig_dims[0]; i++) {
                        sum += sig_data[i * sig_dims[1] + part];
                    }
                    sum /= sig_dims[0];  // Take average
                }
                predSig_data[part] = sum;
            }
        }
    } else {
        npy_intp dims[] = {1};
        predictedSignatureY = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if (!predictedSignatureY) {
            Py_DECREF(signatures_array);
            Py_DECREF(weights_array);
            return NULL;
        }
        
        double* predSig_data = (double*)PyArray_DATA((PyArrayObject*)predictedSignatureY);
        double zeros_count = 0;
        double* sig_data = (double*)PyArray_DATA(signatures_array);
        double* weights_data = (double*)PyArray_DATA(weights_array);
        npy_intp total_elements = PyArray_SIZE(signatures_array);
        
        for(npy_intp i = 0; i < total_elements; i++) {
            if(sig_data[i] == 0.0) {
                zeros_count += 1.0;
            }
        }
        
        if(zeros_count > zeroTolerance) {
            predSig_data[0] = 0.0;
        } else {
            double sum = 0.0;
            if(weighted) {
                // Use weighted calculation
                for(npy_intp i = 0; i < total_elements; i++) {
                    sum += sig_data[i] * weights_data[i];
                }
            } else {
                // Use unweighted calculation - simple average
                for(npy_intp i = 0; i < total_elements; i++) {
                    sum += sig_data[i];
                }
                sum /= total_elements;  // Take average
            }
            predSig_data[0] = sum;
        }
    }

    if (!predictedSignatureY) {
        Py_DECREF(signatures_array);
        Py_DECREF(weights_array);
        return NULL;
    }

    PyObject* predictedPatternY;
    if (PyFloat_Check(predictedSignatureY)) {
        predictedPatternY = calculate_pattern(PyFloat_AsDouble(predictedSignatureY));
    } else {
        predictedPatternY = calculate_pattern_array((PyArrayObject*)predictedSignatureY);
    }

    if (!predictedPatternY) {
        Py_DECREF(signatures_array);
        Py_DECREF(weights_array);
        Py_DECREF(predictedSignatureY);
        return NULL;
    }

    PyObject* return_dict = PyDict_New();
    if (!return_dict) {
        Py_DECREF(signatures_array);
        Py_DECREF(weights_array);
        Py_DECREF(predictedSignatureY);
        Py_DECREF(predictedPatternY);
        return NULL;
    }

    if (PyDict_SetItemString(return_dict, "predictedSignatureY", predictedSignatureY) < 0 ||
        PyDict_SetItemString(return_dict, "predictedPatternY", predictedPatternY) < 0) {
        Py_DECREF(return_dict);
        Py_DECREF(signatures_array);
        Py_DECREF(weights_array);
        Py_DECREF(predictedSignatureY);
        Py_DECREF(predictedPatternY);
        return NULL;
    }

    Py_DECREF(signatures_array);
    Py_DECREF(weights_array);
    Py_DECREF(predictedSignatureY);
    Py_DECREF(predictedPatternY);

    return return_dict;
}

static PyMethodDef PredictionYMethods[] = {
    {"predictionY", (PyCFunction)predictionY, METH_VARARGS | METH_KEYWORDS,
     "Predict Y signature and pattern based on projected nearest neighbors"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef predictionymodule = {
    PyModuleDef_HEAD_INIT,
    "predictionY",
    "Prediction Y calculation module",
    -1,
    PredictionYMethods
};

PyMODINIT_FUNC PyInit_predictionY(void) {
    import_array();
    return PyModule_Create(&predictionymodule);
}
