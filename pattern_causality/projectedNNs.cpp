#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <cmath>

static PyObject* weights_relative_to_distance(PyObject* dists_vec_obj) {
    PyArrayObject* dists_vec = (PyArrayObject*)PyArray_FROM_OTF(dists_vec_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!dists_vec) return NULL;

    npy_intp n = PyArray_SIZE(dists_vec);
    double* dists_data = (double*)PyArray_DATA(dists_vec);
    
    // Calculate sum
    double w_total = 0;
    for(npy_intp i = 0; i < n; i++) {
        w_total += dists_data[i];
    }
    
    if(w_total == 0) {
        w_total = 0.0001;
    }
    
    // Calculate weights_2
    std::vector<double> weights_2(n);
    for(npy_intp i = 0; i < n; i++) {
        weights_2[i] = dists_data[i] / w_total;
    }
    
    // Calculate final weights
    double exp_sum = 0;
    std::vector<double> exp_weights(n);
    for(npy_intp i = 0; i < n; i++) {
        exp_weights[i] = exp(-weights_2[i]);
        exp_sum += exp_weights[i];
    }
    
    npy_intp dims[] = {n};
    PyObject* weights = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    double* weights_data = (double*)PyArray_DATA((PyArrayObject*)weights);
    
    for(npy_intp i = 0; i < n; i++) {
        weights_data[i] = exp_weights[i] / exp_sum;
    }
    
    Py_DECREF(dists_vec);
    return weights;
}

static PyObject* projectedNNs(PyObject* self, PyObject* args) {
    PyObject *my_obj, *dy_obj, *smy_obj, *psmy_obj, *times_x_obj;
    int i, h;
    
    if (!PyArg_ParseTuple(args, "OOOOOii", &my_obj, &dy_obj, &smy_obj, 
                         &psmy_obj, &times_x_obj, &i, &h)) {
        return NULL;
    }
    
    PyArrayObject *my_array = (PyArrayObject*)PyArray_FROM_OTF(my_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *dy_array = (PyArrayObject*)PyArray_FROM_OTF(dy_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *smy_array = (PyArrayObject*)PyArray_FROM_OTF(smy_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *psmy_array = (PyArrayObject*)PyArray_FROM_OTF(psmy_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *times_x_array = (PyArrayObject*)PyArray_FROM_OTF(times_x_obj, NPY_LONG, NPY_ARRAY_IN_ARRAY);
    
    if (!my_array || !dy_array || !smy_array || !psmy_array || !times_x_array) {
        Py_XDECREF(my_array);
        Py_XDECREF(dy_array);
        Py_XDECREF(smy_array);
        Py_XDECREF(psmy_array);
        Py_XDECREF(times_x_array);
        return NULL;
    }
    
    // Calculate projected times
    npy_intp n_times = PyArray_SIZE(times_x_array);
    npy_intp dims[] = {n_times};
    PyObject* projected_times = PyArray_SimpleNew(1, dims, NPY_LONG);
    long* times_data = (long*)PyArray_DATA(times_x_array);
    long* proj_times_data = (long*)PyArray_DATA((PyArrayObject*)projected_times);
    
    for(npy_intp j = 0; j < n_times; j++) {
        proj_times_data[j] = times_data[j] + h;
    }
    
    // Get distances
    PyObject* distances = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    double* dy_data = (double*)PyArray_DATA(dy_array);
    double* dist_data = (double*)PyArray_DATA((PyArrayObject*)distances);
    npy_intp dy_cols = PyArray_SHAPE(dy_array)[1];
    
    for(npy_intp j = 0; j < n_times; j++) {
        dist_data[j] = dy_data[i * dy_cols + proj_times_data[j]];
    }
    
    // Calculate weights
    PyObject* weights = weights_relative_to_distance(distances);
    
    // Get signatures, patterns and coordinates for projected times
    npy_intp sig_dims[] = {n_times, PyArray_SHAPE(smy_array)[1]};
    npy_intp pat_dims[] = {n_times, PyArray_SHAPE(psmy_array)[1]};
    npy_intp coord_dims[] = {n_times, PyArray_SHAPE(my_array)[1]};
    
    PyObject* signatures = PyArray_SimpleNew(2, sig_dims, NPY_DOUBLE);
    PyObject* patterns = PyArray_SimpleNew(2, pat_dims, NPY_DOUBLE);
    PyObject* coordinates = PyArray_SimpleNew(2, coord_dims, NPY_DOUBLE);
    
    double* smy_data = (double*)PyArray_DATA(smy_array);
    double* psmy_data = (double*)PyArray_DATA(psmy_array);
    double* my_data = (double*)PyArray_DATA(my_array);
    double* sig_data = (double*)PyArray_DATA((PyArrayObject*)signatures);
    double* pat_data = (double*)PyArray_DATA((PyArrayObject*)patterns);
    double* coord_data = (double*)PyArray_DATA((PyArrayObject*)coordinates);
    
    for(npy_intp j = 0; j < n_times; j++) {
        long proj_time = proj_times_data[j];
        
        for(npy_intp k = 0; k < sig_dims[1]; k++) {
            sig_data[j * sig_dims[1] + k] = smy_data[proj_time * sig_dims[1] + k];
        }
        
        for(npy_intp k = 0; k < pat_dims[1]; k++) {
            pat_data[j * pat_dims[1] + k] = psmy_data[proj_time * pat_dims[1] + k];
        }
        
        for(npy_intp k = 0; k < coord_dims[1]; k++) {
            coord_data[j * coord_dims[1] + k] = my_data[proj_time * coord_dims[1] + k];
        }
    }
    
    // Build return dictionary
    PyObject* return_dict = PyDict_New();
    PyDict_SetItemString(return_dict, "i", PyLong_FromLong(i));
    PyDict_SetItemString(return_dict, "times_projected", projected_times);
    PyDict_SetItemString(return_dict, "dists", distances);
    PyDict_SetItemString(return_dict, "weights", weights);
    PyDict_SetItemString(return_dict, "signatures", signatures);
    PyDict_SetItemString(return_dict, "patterns", patterns);
    PyDict_SetItemString(return_dict, "coordinates", coordinates);
    
    // Cleanup
    Py_DECREF(my_array);
    Py_DECREF(dy_array);
    Py_DECREF(smy_array);
    Py_DECREF(psmy_array);
    Py_DECREF(times_x_array);
    
    return return_dict;
}

static PyMethodDef ProjectedNNsMethods[] = {
    {"projectedNNs", projectedNNs, METH_VARARGS,
     "Get information about projected nearest neighbors"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef projectednnsmodule = {
    PyModuleDef_HEAD_INIT,
    "projectedNNs", 
    "Projected nearest neighbors calculation module",
    -1,
    ProjectedNNsMethods
};

PyMODINIT_FUNC PyInit_projectedNNs(void) {
    import_array();
    return PyModule_Create(&projectednnsmodule);
}
