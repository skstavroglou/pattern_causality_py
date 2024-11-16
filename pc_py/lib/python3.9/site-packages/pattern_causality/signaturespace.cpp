#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <stdexcept>

static PyObject* signatureVectorDifference(PyObject* self, PyObject* args) {
    PyObject* input_array;
    if (!PyArg_ParseTuple(args, "O", &input_array)) {
        return NULL;
    }

    PyArrayObject* array = (PyArrayObject*)PyArray_FROM_OTF(input_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Could not convert input to numpy array");
        return NULL;
    }

    if (PyArray_NDIM(array) != 1) {
        Py_DECREF(array);
        PyErr_SetString(PyExc_ValueError, "Input must be a 1D array");
        return NULL;
    }

    npy_intp length = PyArray_DIM(array, 0);
    
    npy_intp dims[1] = {length - 1};
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (result_array == NULL) {
        Py_DECREF(array);
        return NULL;
    }

    double* input_data = (double*)PyArray_DATA(array);
    double* output_data = (double*)PyArray_DATA(result_array);

    for (npy_intp i = 0; i < length - 1; i++) {
        output_data[i] = input_data[i + 1] - input_data[i];
    }

    Py_DECREF(array);
    return (PyObject*)result_array;
}

static PyObject* signaturespace(PyObject* self, PyObject* args) {
    PyObject* input_matrix;
    int E;
    if (!PyArg_ParseTuple(args, "Oi", &input_matrix, &E)) {
        return NULL;
    }

    if (E < 2) {
        PyErr_SetString(PyExc_ValueError, "E must be >= 2");
        return NULL;
    }

    PyArrayObject* array = (PyArrayObject*)PyArray_FROM_OTF(input_matrix, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Could not convert input to numpy array");
        return NULL;
    }

    if (PyArray_NDIM(array) != 2) {
        Py_DECREF(array);
        PyErr_SetString(PyExc_ValueError, "Input must be a 2D array");
        return NULL;
    }

    npy_intp rows = PyArray_DIM(array, 0);
    npy_intp cols = PyArray_DIM(array, 1);

    if (rows == 0) {
        Py_DECREF(array);
        npy_intp dims[2] = {0, cols-1};
        return (PyObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    }

    npy_intp out_dims[2] = {rows, cols-1};
    PyArrayObject* result_matrix = (PyArrayObject*)PyArray_SimpleNew(2, out_dims, NPY_DOUBLE);
    if (result_matrix == NULL) {
        Py_DECREF(array);
        return NULL;
    }

    double* input_data = (double*)PyArray_DATA(array);
    double* output_data = (double*)PyArray_DATA(result_matrix);


    for (npy_intp i = 0; i < rows; i++) {
        for (npy_intp j = 0; j < cols-1; j++) {
            output_data[i * (cols-1) + j] = input_data[i * cols + (j+1)] - input_data[i * cols + j];
        }
    }

    Py_DECREF(array);
    return (PyObject*)result_matrix;
}

static PyMethodDef SignatureSpaceMethods[] = {
    {"signatureVectorDifference", signatureVectorDifference, METH_VARARGS, "Calculate differences between successive elements"},
    {"signaturespace", signaturespace, METH_VARARGS, "Calculate signature space matrix"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef signaturespacemodule = {
    PyModuleDef_HEAD_INIT,
    "signaturespace",
    "Signature space calculation module",
    -1,
    SignatureSpaceMethods
};

PyMODINIT_FUNC PyInit_signaturespace(void) {
    import_array(); 
    return PyModule_Create(&signaturespacemodule);
}