#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

// Thread-local storage for reusable vectors
thread_local std::vector<double> weights_2_buffer;
thread_local std::vector<double> exp_weights_buffer;

// Optimized weights calculation with SIMD support
static PyObject* weights_relative_to_distance(PyObject* dists_vec_obj) {
    PyArrayObject* dists_vec = (PyArrayObject*)PyArray_FROM_OTF(dists_vec_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!dists_vec) return NULL;

    const npy_intp n = PyArray_SIZE(dists_vec);
    const double* dists_data = (double*)PyArray_DATA(dists_vec);
    
    // Calculate sum using SIMD
    double w_total = 0.0;
    #pragma omp simd reduction(+:w_total)
    for(npy_intp i = 0; i < n; i++) {
        w_total += dists_data[i];
    }
    
    // Handle zero case
    w_total = (w_total == 0.0) ? 0.0001 : w_total;
    const double w_total_inv = 1.0 / w_total;
    
    // Reuse pre-allocated vectors
    if (weights_2_buffer.size() < n) {
        weights_2_buffer.resize(n);
        exp_weights_buffer.resize(n);
    }
    
    // Calculate weights_2 using SIMD
    #pragma omp simd
    for(npy_intp i = 0; i < n; i++) {
        weights_2_buffer[i] = dists_data[i] * w_total_inv;
    }
    
    // Calculate exponentials using SIMD
    double exp_sum = 0.0;
    #pragma omp simd reduction(+:exp_sum)
    for(npy_intp i = 0; i < n; i++) {
        exp_weights_buffer[i] = std::exp(-weights_2_buffer[i]);
        exp_sum += exp_weights_buffer[i];
    }
    
    // Prepare output array
    npy_intp dims[] = {n};
    PyObject* weights = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!weights) {
        Py_DECREF(dists_vec);
        return NULL;
    }
    
    // Calculate final weights using SIMD
    const double exp_sum_inv = 1.0 / exp_sum;
    double* weights_data = (double*)PyArray_DATA((PyArrayObject*)weights);
    #pragma omp simd
    for(npy_intp i = 0; i < n; i++) {
        weights_data[i] = exp_weights_buffer[i] * exp_sum_inv;
    }
    
    Py_DECREF(dists_vec);
    return weights;
}

// Optimized projectedNNs function
static PyObject* projectedNNs(PyObject* self, PyObject* args) {
    PyObject *my_obj, *dy_obj, *smy_obj, *psmy_obj, *times_x_obj;
    int i, h;
    
    if (!PyArg_ParseTuple(args, "OOOOOii", &my_obj, &dy_obj, &smy_obj, 
                         &psmy_obj, &times_x_obj, &i, &h)) {
        return NULL;
    }
    
    // Convert input arrays with error checking
    PyArrayObject* arrays[] = {
        (PyArrayObject*)PyArray_FROM_OTF(my_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        (PyArrayObject*)PyArray_FROM_OTF(dy_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        (PyArrayObject*)PyArray_FROM_OTF(smy_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        (PyArrayObject*)PyArray_FROM_OTF(psmy_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY),
        (PyArrayObject*)PyArray_FROM_OTF(times_x_obj, NPY_LONG, NPY_ARRAY_IN_ARRAY)
    };
    
    // Check for conversion errors
    for (int j = 0; j < 5; j++) {
        if (!arrays[j]) {
            for (int k = 0; k < j; k++) {
                Py_DECREF(arrays[k]);
            }
            return NULL;
        }
    }
    
    // Get array dimensions once
    const npy_intp n_times = PyArray_SIZE(arrays[4]);
    const npy_intp dy_cols = PyArray_SHAPE(arrays[1])[1];
    const npy_intp sig_cols = PyArray_SHAPE(arrays[2])[1];
    const npy_intp pat_cols = PyArray_SHAPE(arrays[3])[1];
    const npy_intp coord_cols = PyArray_SHAPE(arrays[0])[1];
    
    // Pre-allocate all output arrays
    npy_intp dims[] = {n_times};
    PyObject* projected_times = PyArray_SimpleNew(1, dims, NPY_LONG);
    PyObject* distances = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    
    npy_intp sig_dims[] = {n_times, sig_cols};
    npy_intp pat_dims[] = {n_times, pat_cols};
    npy_intp coord_dims[] = {n_times, coord_cols};
    
    PyObject* signatures = PyArray_SimpleNew(2, sig_dims, NPY_DOUBLE);
    PyObject* patterns = PyArray_SimpleNew(2, pat_dims, NPY_DOUBLE);
    PyObject* coordinates = PyArray_SimpleNew(2, coord_dims, NPY_DOUBLE);
    
    // Check memory allocation
    if (!projected_times || !distances || !signatures || !patterns || !coordinates) {
        for (auto arr : arrays) Py_DECREF(arr);
        Py_XDECREF(projected_times);
        Py_XDECREF(distances);
        Py_XDECREF(signatures);
        Py_XDECREF(patterns);
        Py_XDECREF(coordinates);
        return NULL;
    }
    
    // Get data pointers
    long* times_data = (long*)PyArray_DATA(arrays[4]);
    double* dy_data = (double*)PyArray_DATA(arrays[1]);
    double* smy_data = (double*)PyArray_DATA(arrays[2]);
    double* psmy_data = (double*)PyArray_DATA(arrays[3]);
    double* my_data = (double*)PyArray_DATA(arrays[0]);
    
    long* proj_times_data = (long*)PyArray_DATA((PyArrayObject*)projected_times);
    double* dist_data = (double*)PyArray_DATA((PyArrayObject*)distances);
    double* sig_data = (double*)PyArray_DATA((PyArrayObject*)signatures);
    double* pat_data = (double*)PyArray_DATA((PyArrayObject*)patterns);
    double* coord_data = (double*)PyArray_DATA((PyArrayObject*)coordinates);
    
    // Calculate projected times and distances using SIMD
    #pragma omp parallel for simd schedule(static)
    for(npy_intp j = 0; j < n_times; j++) {
        const long proj_time = times_data[j] + h;
        proj_times_data[j] = proj_time;
        dist_data[j] = dy_data[i * dy_cols + proj_time];
    }
    
    // Calculate weights
    PyObject* weights = weights_relative_to_distance(distances);
    if (!weights) {
        for (auto arr : arrays) Py_DECREF(arr);
        Py_DECREF(projected_times);
        Py_DECREF(distances);
        Py_DECREF(signatures);
        Py_DECREF(patterns);
        Py_DECREF(coordinates);
        return NULL;
    }
    
    // Copy data using parallel processing where beneficial
    #pragma omp parallel for collapse(2) schedule(static)
    for(npy_intp j = 0; j < n_times; j++) {
        for(npy_intp k = 0; k < sig_cols; k++) {
            const long proj_time = proj_times_data[j];
            sig_data[j * sig_cols + k] = smy_data[proj_time * sig_cols + k];
        }
    }
    
    #pragma omp parallel for collapse(2) schedule(static)
    for(npy_intp j = 0; j < n_times; j++) {
        for(npy_intp k = 0; k < pat_cols; k++) {
            const long proj_time = proj_times_data[j];
            pat_data[j * pat_cols + k] = psmy_data[proj_time * pat_cols + k];
        }
    }
    
    #pragma omp parallel for collapse(2) schedule(static)
    for(npy_intp j = 0; j < n_times; j++) {
        for(npy_intp k = 0; k < coord_cols; k++) {
            const long proj_time = proj_times_data[j];
            coord_data[j * coord_cols + k] = my_data[proj_time * coord_cols + k];
        }
    }
    
    // Build return dictionary
    PyObject* return_dict = PyDict_New();
    if (!return_dict) {
        for (auto arr : arrays) Py_DECREF(arr);
        Py_DECREF(projected_times);
        Py_DECREF(distances);
        Py_DECREF(weights);
        Py_DECREF(signatures);
        Py_DECREF(patterns);
        Py_DECREF(coordinates);
        return NULL;
    }
    
    // Set dictionary items
    const char* keys[] = {"i", "times_projected", "dists", "weights", 
                         "signatures", "patterns", "coordinates"};
    PyObject* values[] = {PyLong_FromLong(i), projected_times, distances, 
                         weights, signatures, patterns, coordinates};
    
    for (int j = 0; j < 7; j++) {
        if (PyDict_SetItemString(return_dict, keys[j], values[j]) < 0) {
            for (auto arr : arrays) Py_DECREF(arr);
            for (auto val : values) Py_DECREF(val);
            Py_DECREF(return_dict);
            return NULL;
        }
        Py_DECREF(values[j]);
    }
    
    // Cleanup input arrays
    for (auto arr : arrays) {
        Py_DECREF(arr);
    }
    
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
