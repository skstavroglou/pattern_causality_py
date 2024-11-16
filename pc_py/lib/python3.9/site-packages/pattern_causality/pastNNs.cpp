#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <algorithm>

static PyObject* pastNNs(PyObject* self, PyObject* args, PyObject* kwargs) {
    int ccspan, nnspan, i, h;
    PyObject *mx_obj, *dx_obj, *smx_obj, *psmx_obj;
    
    static char* kwlist[] = {
        "ccspan", "nnspan", "mx", "dx", "smx", "psmx", "i", "h", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiOOOOii", kwlist,
                                    &ccspan, &nnspan, &mx_obj, &dx_obj, 
                                    &smx_obj, &psmx_obj, &i, &h)) {
        return NULL;
    }

    // Convert numpy arrays
    PyArrayObject *mx_array = (PyArrayObject*)PyArray_FROM_OTF(mx_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *dx_array = (PyArrayObject*)PyArray_FROM_OTF(dx_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *smx_array = (PyArrayObject*)PyArray_FROM_OTF(smx_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *psmx_array = (PyArrayObject*)PyArray_FROM_OTF(psmx_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    if (!mx_array || !dx_array || !smx_array || !psmx_array) {
        Py_XDECREF(mx_array);
        Py_XDECREF(dx_array);
        Py_XDECREF(smx_array);
        Py_XDECREF(psmx_array);
        return NULL;
    }

    // Get candidate NNs up to i-ccspan-h
    int candidate_length = i - ccspan - h;
    std::vector<std::pair<double, int>> candidate_nns;
    double* dx_data = (double*)PyArray_DATA(dx_array);
    
    for(int j = 0; j < candidate_length; j++) {
        candidate_nns.push_back({dx_data[i * PyArray_SHAPE(dx_array)[1] + j], j});
    }

    // Sort by distance and get top nnspan indices
    std::sort(candidate_nns.begin(), candidate_nns.end());
    std::vector<int> nn_indices;
    std::vector<double> dists;
    
    for(int j = 0; j < nnspan && j < candidate_nns.size(); j++) {
        nn_indices.push_back(candidate_nns[j].second);
        dists.push_back(candidate_nns[j].first);
    }

    // Create numpy arrays for return values
    npy_intp times_dims[1] = {(npy_intp)nn_indices.size()};
    PyObject* times_array = PyArray_SimpleNew(1, times_dims, NPY_INT);
    PyObject* dists_array = PyArray_SimpleNew(1, times_dims, NPY_DOUBLE);
    
    // Copy data to arrays
    int* times_data = (int*)PyArray_DATA((PyArrayObject*)times_array);
    double* dists_data = (double*)PyArray_DATA((PyArrayObject*)dists_array);
    
    for(size_t j = 0; j < nn_indices.size(); j++) {
        times_data[j] = nn_indices[j];
        dists_data[j] = dists[j];
    }

    // Get signatures, patterns and coordinates for those times
    npy_intp* smx_dims = PyArray_SHAPE(smx_array);
    npy_intp* psmx_dims = PyArray_SHAPE(psmx_array);
    npy_intp* mx_dims = PyArray_SHAPE(mx_array);
    
    npy_intp signatures_dims[2] = {(npy_intp)nn_indices.size(), smx_dims[1]};
    npy_intp patterns_dims[2] = {(npy_intp)nn_indices.size(), psmx_dims[1]};
    npy_intp coordinates_dims[2] = {(npy_intp)nn_indices.size(), mx_dims[1]};
    
    PyObject* signatures_array = PyArray_SimpleNew(2, signatures_dims, NPY_DOUBLE);
    PyObject* patterns_array = PyArray_SimpleNew(2, patterns_dims, NPY_DOUBLE);
    PyObject* coordinates_array = PyArray_SimpleNew(2, coordinates_dims, NPY_DOUBLE);

    double* smx_data = (double*)PyArray_DATA(smx_array);
    double* psmx_data = (double*)PyArray_DATA(psmx_array);
    double* mx_data = (double*)PyArray_DATA(mx_array);
    double* signatures_data = (double*)PyArray_DATA((PyArrayObject*)signatures_array);
    double* patterns_data = (double*)PyArray_DATA((PyArrayObject*)patterns_array);
    double* coordinates_data = (double*)PyArray_DATA((PyArrayObject*)coordinates_array);

    for(size_t j = 0; j < nn_indices.size(); j++) {
        int idx = nn_indices[j];
        
        // Copy signatures
        for(npy_intp k = 0; k < smx_dims[1]; k++) {
            signatures_data[j * smx_dims[1] + k] = smx_data[idx * smx_dims[1] + k];
        }
        
        // Copy patterns
        for(npy_intp k = 0; k < psmx_dims[1]; k++) {
            patterns_data[j * psmx_dims[1] + k] = psmx_data[idx * psmx_dims[1] + k];
        }
        
        // Copy coordinates
        for(npy_intp k = 0; k < mx_dims[1]; k++) {
            coordinates_data[j * mx_dims[1] + k] = mx_data[idx * mx_dims[1] + k];
        }
    }

    // Build return dictionary
    PyObject* return_dict = PyDict_New();
    PyDict_SetItemString(return_dict, "i", PyLong_FromLong(i));
    PyDict_SetItemString(return_dict, "times", times_array);
    PyDict_SetItemString(return_dict, "dists", dists_array);
    PyDict_SetItemString(return_dict, "signatures", signatures_array);
    PyDict_SetItemString(return_dict, "patterns", patterns_array);
    PyDict_SetItemString(return_dict, "coordinates", coordinates_array);

    // Cleanup
    Py_DECREF(mx_array);
    Py_DECREF(dx_array);
    Py_DECREF(smx_array);
    Py_DECREF(psmx_array);

    return return_dict;
}

static PyMethodDef PastNNsMethods[] = {
    {"pastNNs", (PyCFunction)pastNNs, METH_VARARGS | METH_KEYWORDS,
     "Get information about past nearest neighbors"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef pastnnsmodule = {
    PyModuleDef_HEAD_INIT,
    "pastNNs",
    "Past nearest neighbors calculation module",
    -1,
    PastNNsMethods
};

PyMODINIT_FUNC PyInit_pastNNs(void) {
    import_array();
    return PyModule_Create(&pastnnsmodule);
}
