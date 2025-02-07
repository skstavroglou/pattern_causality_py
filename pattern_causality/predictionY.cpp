#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <cmath>
#include <limits>

// Pre-compute factorials for common cases
static constexpr size_t MAX_FACTORIAL_CACHE = 10;
static std::array<int, MAX_FACTORIAL_CACHE> factorial_cache = []() {
    std::array<int, MAX_FACTORIAL_CACHE> cache{};
    cache[0] = 1;
    for(size_t i = 1; i < MAX_FACTORIAL_CACHE; ++i) {
        cache[i] = cache[i-1] * i;
    }
    return cache;
}();

// Optimized factorial calculation with cache
static inline int factorial(int n) {
    if (n < 0) return 1;  // Handle error case
    if (n < MAX_FACTORIAL_CACHE) {
        return factorial_cache[n];
    }
    int result = factorial_cache[MAX_FACTORIAL_CACHE - 1];
    for(int i = MAX_FACTORIAL_CACHE; i <= n; ++i) {
        result *= i;
    }
    return result;
}

// Optimized hashing function with SIMD hints
static inline int hashing(const std::vector<int>& vec) {
    int hash = 0;
    const size_t size = vec.size();
    #pragma omp simd reduction(+:hash)
    for (size_t i = 0; i < size; i++) {
        hash += vec[i] * factorial(i + 2);
    }
    return hash;
}

// Thread-local storage for reusable vectors
thread_local std::vector<int> p_vec_buffer;

static inline int pattern_vector_difference(const std::vector<double>& sVec) {
    // Quick check for NaN values
    for (const auto& val : sVec) {
        if (std::isnan(val)) {
            return 0;
        }
    }
    
    // Reuse pre-allocated vector
    if (p_vec_buffer.capacity() < sVec.size()) {
        p_vec_buffer.reserve(sVec.size());
    }
    p_vec_buffer.clear();
    
    // Vectorized pattern calculation
    #pragma omp simd
    for (const auto& val : sVec) {
        p_vec_buffer.push_back(val > 0 ? 3 : (val < 0 ? 1 : 2));
    }
    
    return hashing(p_vec_buffer);
}

static PyObject* predictionY(PyObject* self, PyObject* args, PyObject* kwargs) {
    long E;
    PyObject* projNNy;
    PyObject* zeroTolerance_obj = Py_None;
    
    static char* kwlist[] = {"E", "projNNy", "zeroTolerance", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "lO|O", kwlist, 
                                    &E, &projNNy, &zeroTolerance_obj)) {
        return NULL;
    }

    // Optimize default value calculation
    const double zeroTolerance = (zeroTolerance_obj == Py_None) ? 
                                (E + 1.0) / 2.0 : 
                                PyFloat_AsDouble(zeroTolerance_obj);
    
    if (PyErr_Occurred()) return NULL;

    // Get dictionary items with error checking
    PyObject* signatures = PyDict_GetItemString(projNNy, "signatures");
    PyObject* weights = PyDict_GetItemString(projNNy, "weights");
    
    if (!signatures || !weights) {
        PyErr_SetString(PyExc_KeyError, "projNNy must contain 'signatures' and 'weights' keys");
        return NULL;
    }

    // Convert to numpy arrays with error checking
    PyArrayObject* signatures_array = (PyArrayObject*)PyArray_FROM_OTF(signatures, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject* weights_array = (PyArrayObject*)PyArray_FROM_OTF(weights, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    
    if (!signatures_array || !weights_array) {
        Py_XDECREF(signatures_array);
        Py_XDECREF(weights_array);
        PyErr_SetString(PyExc_TypeError, "Failed to convert signatures or weights to numpy array");
        return NULL;
    }

    // Pre-allocate vector with proper size
    std::vector<double> predictedSignatureY;
    predictedSignatureY.reserve(E >= 3 ? E - 1 : 1);
    
    double* sig_data = (double*)PyArray_DATA(signatures_array);
    double* weights_data = (double*)PyArray_DATA(weights_array);
    npy_intp* sig_dims = PyArray_DIMS(signatures_array);

    if (E >= 3) {
        predictedSignatureY.resize(E - 1, 0.0);
        const npy_intp rows = sig_dims[0];
        const npy_intp cols = sig_dims[1];
        
        // Optimize main calculation loop
        #pragma omp parallel for
        for(long part = 1; part <= E - 1; part++) {
            int zero_count = 0;
            double sum = 0.0;
            
            // Vectorized inner loop
            #pragma omp simd reduction(+:zero_count,sum)
            for(npy_intp i = 0; i < rows; i++) {
                const double sig_val = sig_data[i * cols + (part-1)];
                zero_count += (sig_val == 0.0);
                sum += sig_val * weights_data[i];
            }
            
            predictedSignatureY[part-1] = (zero_count > zeroTolerance) ? 0.0 : sum;
        }
    } else {
        predictedSignatureY.resize(1, 0.0);
        const npy_intp total_elements = PyArray_SIZE(signatures_array);
        
        int zero_count = 0;
        double sum = 0.0;
        
        // Vectorized calculation for E < 3 case
        #pragma omp simd reduction(+:zero_count,sum)
        for(npy_intp i = 0; i < total_elements; i++) {
            zero_count += (sig_data[i] == 0.0);
            sum += sig_data[i] * weights_data[i];
        }
        
        predictedSignatureY[0] = (zero_count > zeroTolerance) ? 0.0 : sum;
    }

    // Calculate pattern value
    const int pattern_value = pattern_vector_difference(predictedSignatureY);

    // Create return objects
    npy_intp sig_dims_out[] = {static_cast<npy_intp>(predictedSignatureY.size())};
    PyObject* predictedSignatureY_array = PyArray_SimpleNew(1, sig_dims_out, NPY_DOUBLE);
    if (!predictedSignatureY_array) {
        Py_DECREF(signatures_array);
        Py_DECREF(weights_array);
        return NULL;
    }
    
    // Fast memory copy
    memcpy(PyArray_DATA((PyArrayObject*)predictedSignatureY_array),
           predictedSignatureY.data(),
           predictedSignatureY.size() * sizeof(double));

    PyObject* predictedPatternY = PyLong_FromLong(pattern_value);
    if (!predictedPatternY) {
        Py_DECREF(signatures_array);
        Py_DECREF(weights_array);
        Py_DECREF(predictedSignatureY_array);
        return NULL;
    }

    // Create return dictionary
    PyObject* return_dict = PyDict_New();
    if (!return_dict || 
        PyDict_SetItemString(return_dict, "predictedSignatureY", predictedSignatureY_array) < 0 ||
        PyDict_SetItemString(return_dict, "predictedPatternY", predictedPatternY) < 0) {
        Py_XDECREF(return_dict);
        Py_DECREF(signatures_array);
        Py_DECREF(weights_array);
        Py_DECREF(predictedPatternY);
        Py_DECREF(predictedSignatureY_array);
        return NULL;
    }

    // Cleanup
    Py_DECREF(signatures_array);
    Py_DECREF(weights_array);
    Py_DECREF(predictedPatternY);
    Py_DECREF(predictedSignatureY_array);

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
