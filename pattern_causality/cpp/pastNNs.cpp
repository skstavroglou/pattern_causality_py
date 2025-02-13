#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

// Pre-allocated buffer size
constexpr size_t INITIAL_BUFFER_SIZE = 1024;

// Reusable buffer
static std::vector<std::pair<double, int>> candidate_buffer;
static std::vector<int> nn_indices_buffer;
static std::vector<double> dists_buffer;

static PyObject* pastNNs(PyObject* self, PyObject* args) {
    int ccspan, nnspan, i, h;
    PyObject *mx_obj, *dx_obj, *smx_obj, *psmx_obj;
    
    if (!PyArg_ParseTuple(args, "iiOOOOii", &ccspan, &nnspan, 
                         &mx_obj, &dx_obj, &smx_obj, &psmx_obj, &i, &h)) {
        return NULL;
    }
    
    // Convert inputs to numpy arrays
    PyArrayObject* mx_array = (PyArrayObject*)PyArray_FROM_OTF(mx_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* dx_array = (PyArrayObject*)PyArray_FROM_OTF(dx_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* smx_array = (PyArrayObject*)PyArray_FROM_OTF(smx_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* psmx_array = (PyArrayObject*)PyArray_FROM_OTF(psmx_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    
    if (!mx_array || !dx_array || !smx_array || !psmx_array) {
        Py_XDECREF(mx_array);
        Py_XDECREF(dx_array);
        Py_XDECREF(smx_array);
        Py_XDECREF(psmx_array);
        return NULL;
    }
    
    // Get array dimensions
    const npy_intp* mx_dims = PyArray_DIMS(mx_array);
    const npy_intp* smx_dims = PyArray_DIMS(smx_array);
    const npy_intp mx_stride_0 = PyArray_STRIDE(mx_array, 0) / sizeof(double);
    const npy_intp smx_stride_0 = PyArray_STRIDE(smx_array, 0) / sizeof(double);
    const npy_intp psmx_stride = PyArray_STRIDE(psmx_array, 0) / sizeof(double);
    
    // Get data pointers
    double* mx_data = (double*)PyArray_DATA(mx_array);
    double* dx_data = (double*)PyArray_DATA(dx_array);
    double* smx_data = (double*)PyArray_DATA(smx_array);
    double* psmx_data = (double*)PyArray_DATA(psmx_array);
    
    // Find valid indices
    std::vector<int> valid_indices;
    valid_indices.reserve(mx_dims[0]);
    
    for (npy_intp j = 0; j < i - ccspan; j++) {
        bool valid = true;
        // Check for NaN in state space
        for (npy_intp k = 0; k < mx_dims[1]; k++) {
            if (std::isnan(mx_data[j * mx_stride_0 + k])) {
                valid = false;
                break;
            }
        }
        // Check for NaN in distance matrix
        if (valid && std::isnan(dx_data[i * mx_dims[0] + j])) {
            valid = false;
        }
        if (valid) {
            valid_indices.push_back(j);
        }
    }
    
    // Sort indices by distance
    std::vector<std::pair<double, int>> distances;
    distances.reserve(valid_indices.size());
    
    for (int idx : valid_indices) {
        distances.push_back({dx_data[i * mx_dims[0] + idx], idx});
    }
    
    std::sort(distances.begin(), distances.end());
    
    // Take only nnspan nearest neighbors
    const size_t sort_size = std::min(static_cast<size_t>(nnspan), distances.size());
    
    // Create output arrays
    npy_intp out_dims[] = {static_cast<npy_intp>(sort_size)};
    PyObject* times = PyArray_SimpleNew(1, out_dims, NPY_LONG);
    PyObject* dists = PyArray_SimpleNew(1, out_dims, NPY_DOUBLE);
    
    npy_intp sig_dims[] = {static_cast<npy_intp>(sort_size), smx_dims[1]};
    PyObject* signatures = PyArray_SimpleNew(2, sig_dims, NPY_DOUBLE);
    
    npy_intp pat_dims[] = {static_cast<npy_intp>(sort_size), 1};
    PyObject* patterns = PyArray_SimpleNew(2, pat_dims, NPY_DOUBLE);
    
    npy_intp coord_dims[] = {static_cast<npy_intp>(sort_size), mx_dims[1]};
    PyObject* coordinates = PyArray_SimpleNew(2, coord_dims, NPY_DOUBLE);
    
    if (!times || !dists || !signatures || !patterns || !coordinates) {
        Py_XDECREF(times);
        Py_XDECREF(dists);
        Py_XDECREF(signatures);
        Py_XDECREF(patterns);
        Py_XDECREF(coordinates);
        Py_DECREF(mx_array);
        Py_DECREF(dx_array);
        Py_DECREF(smx_array);
        Py_DECREF(psmx_array);
        return NULL;
    }
    
    // Fill output arrays
    long* times_data = (long*)PyArray_DATA((PyArrayObject*)times);
    double* dists_data = (double*)PyArray_DATA((PyArrayObject*)dists);
    double* signatures_data = (double*)PyArray_DATA((PyArrayObject*)signatures);
    double* patterns_data = (double*)PyArray_DATA((PyArrayObject*)patterns);
    double* coordinates_data = (double*)PyArray_DATA((PyArrayObject*)coordinates);
    
    for (size_t j = 0; j < sort_size; j++) {
        const int idx = distances[j].second;
        times_data[j] = idx;
        dists_data[j] = distances[j].first;
        
        // Copy signatures
        for (npy_intp k = 0; k < smx_dims[1]; k++) {
            signatures_data[j * smx_dims[1] + k] = smx_data[idx * smx_stride_0 + k];
        }
        
        // Copy pattern
        patterns_data[j] = psmx_data[idx * psmx_stride];
        
        // Copy coordinates
        for (npy_intp k = 0; k < mx_dims[1]; k++) {
            coordinates_data[j * mx_dims[1] + k] = mx_data[idx * mx_stride_0 + k];
        }
    }
    
    // Create return dictionary
    PyObject* result = PyDict_New();
    if (!result) {
        Py_DECREF(times);
        Py_DECREF(dists);
        Py_DECREF(signatures);
        Py_DECREF(patterns);
        Py_DECREF(coordinates);
        Py_DECREF(mx_array);
        Py_DECREF(dx_array);
        Py_DECREF(smx_array);
        Py_DECREF(psmx_array);
        return NULL;
    }
    
    PyDict_SetItemString(result, "times", times);
    PyDict_SetItemString(result, "dists", dists);
    PyDict_SetItemString(result, "signatures", signatures);
    PyDict_SetItemString(result, "patterns", patterns);
    PyDict_SetItemString(result, "coordinates", coordinates);
    
    Py_DECREF(times);
    Py_DECREF(dists);
    Py_DECREF(signatures);
    Py_DECREF(patterns);
    Py_DECREF(coordinates);
    Py_DECREF(mx_array);
    Py_DECREF(dx_array);
    Py_DECREF(smx_array);
    Py_DECREF(psmx_array);
    
    return result;
}

static PyMethodDef PastNNsMethods[] = {
    {"pastNNs", (PyCFunction)pastNNs, METH_VARARGS,
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
