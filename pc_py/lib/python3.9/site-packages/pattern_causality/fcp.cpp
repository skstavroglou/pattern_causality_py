#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject* fcp(PyObject* self, PyObject* args) {
    int E, tau, h;
    PyObject* X;
    
    if (!PyArg_ParseTuple(args, "iiiO", &E, &tau, &h, &X)) {
        return NULL;
    }

    if (!PyList_Check(X)) {
        PyErr_SetString(PyExc_TypeError, "X must be a list");
        return NULL;
    }

    int NNSPAN = E + 1;  // Former NN | Reserves a minimum number of nearest neighbors
    int CCSPAN = (E - 1) * tau;  // This will remove the common coordinate NNs  
    int PredSPAN = h;
    int FCP = 1 + NNSPAN + CCSPAN + PredSPAN;

    Py_ssize_t X_len = PyList_Size(X);

    if (NNSPAN + CCSPAN + PredSPAN >= X_len - CCSPAN) {
        PyErr_SetString(PyExc_ValueError, 
            "The First Point to consider for Causality does not have sufficient "
            "Nearest Neighbors. Please Check parameters: "
            "E, lag, p as well as the length of X and Y");
        return NULL;
    }

    return PyLong_FromLong(FCP);
}

static PyMethodDef FcpMethods[] = {
    {"fcp", fcp, METH_VARARGS, "Calculate first causality point"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fcpmodule = {
    PyModuleDef_HEAD_INIT,
    "utils.fcp",
    "First causality point calculation module",
    -1,
    FcpMethods
};

PyMODINIT_FUNC PyInit_fcp(void) {
    return PyModule_Create(&fcpmodule);
}
