#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>
#include <limits>

// Include SIMD headers based on architecture
#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

// Optimized norm calculation using available SIMD instructions
static double norm_vec(PyObject* x) {
    PyArrayObject* arr = (PyArrayObject*)PyArray_FROM_OTF(x, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ALIGNED);
    if (!arr) {
        return 0.0;
    }
    
    double sum = 0.0;
    double* data = (double*)PyArray_DATA(arr);
    npy_intp size = PyArray_SIZE(arr);
    
    #ifdef __ARM_NEON
    // ARM NEON implementation (processes 2 doubles at a time)
    float64x2_t sum_vec = vdupq_n_f64(0.0);
    npy_intp i;
    
    for(i = 0; i <= size - 2; i += 2) {
        float64x2_t v = vld1q_f64(data + i);
        sum_vec = vfmaq_f64(sum_vec, v, v);
    }
    
    sum = vgetq_lane_f64(sum_vec, 0) + vgetq_lane_f64(sum_vec, 1);
    
    for(; i < size; i++) {
        sum += data[i] * data[i];
    }
    #elif defined(__AVX__)
    // x86 AVX implementation (processes 4 doubles at a time)
    __m256d sum_vec = _mm256_setzero_pd();
    npy_intp i;
    
    for(i = 0; i <= size - 4; i += 4) {
        __m256d v = _mm256_load_pd(data + i);
        sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(v, v));
    }
    
    // Horizontal sum
    __m128d sum128 = _mm_add_pd(_mm256_extractf128_pd(sum_vec, 0),
                                _mm256_extractf128_pd(sum_vec, 1));
    sum = _mm_cvtsd_f64(sum128) + _mm_cvtsd_f64(_mm_unpackhi_pd(sum128, sum128));
    
    for(; i < size; i++) {
        sum += data[i] * data[i];
    }
    #else
    // Fallback to scalar operations with OpenMP SIMD
    #pragma omp simd reduction(+:sum)
    for(npy_intp i = 0; i < size; i++) {
        sum += data[i] * data[i];
    }
    #endif
    
    Py_DECREF(arr);
    return sqrt(sum);
}

static PyObject* fillPCMatrix(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *predictedPatternY_obj, *realPatternY_obj, *predictedSignatureY_obj;
    PyObject *realSignatureY_obj, *patternX_obj, *signatureX_obj;
    PyObject* weighted_obj;

    static const char* const kwlist[] = {
        "weighted", "predictedPatternY", "realPatternY",
        "predictedSignatureY", "realSignatureY",
        "patternX", "signatureX", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOOOO", const_cast<char**>(kwlist),
                                    &weighted_obj,
                                    &predictedPatternY_obj, &realPatternY_obj,
                                    &predictedSignatureY_obj, &realSignatureY_obj,
                                    &patternX_obj, &signatureX_obj)) {
        return NULL;
    }

    const bool weighted = PyObject_IsTrue(weighted_obj);

    // Convert inputs to numpy arrays with optimization flags
    const int requirements = NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ALIGNED;
    PyArrayObject* pred_pattern_arr = (PyArrayObject*)PyArray_FROM_OTF(predictedPatternY_obj, NPY_DOUBLE, requirements);
    PyArrayObject* real_pattern_arr = (PyArrayObject*)PyArray_FROM_OTF(realPatternY_obj, NPY_DOUBLE, requirements);
    PyArrayObject* pattern_x_arr = (PyArrayObject*)PyArray_FROM_OTF(patternX_obj, NPY_DOUBLE, requirements);
    
    if (!pred_pattern_arr || !real_pattern_arr || !pattern_x_arr) {
        Py_XDECREF(pred_pattern_arr);
        Py_XDECREF(real_pattern_arr);
        Py_XDECREF(pattern_x_arr);
        PyErr_SetString(PyExc_TypeError, "Could not convert input to numpy array");
        return NULL;
    }

    // Get array data and sizes
    const double* const pred_pattern = (const double*)PyArray_DATA(pred_pattern_arr);
    const double* const real_pattern = (const double*)PyArray_DATA(real_pattern_arr);
    const double* const pattern_x = (const double*)PyArray_DATA(pattern_x_arr);
    
    const npy_intp size_pred = PyArray_SIZE(pred_pattern_arr);
    const npy_intp size_real = PyArray_SIZE(real_pattern_arr);
    const npy_intp size_x = PyArray_SIZE(pattern_x_arr);

    // Quick size checks first
    if (size_pred == 0 || size_x == 0) {
        Py_DECREF(pred_pattern_arr);
        Py_DECREF(real_pattern_arr);
        Py_DECREF(pattern_x_arr);
        PyErr_SetString(PyExc_ValueError, size_pred == 0 ? 
            "The length of the predicted pattern of Y is ZERO" :
            "The length of the causal pattern of X is ZERO");
        return NULL;
    }

    // Check for NaN values using available SIMD instructions
    bool has_nan = false;
    #ifdef __ARM_NEON
    npy_intp i;
    for(i = 0; i <= size_pred - 2 && !has_nan; i += 2) {
        float64x2_t v = vld1q_f64(pred_pattern + i);
        uint64x2_t cmp = vceqq_f64(v, v);
        if (vgetq_lane_u64(cmp, 0) == 0 || vgetq_lane_u64(cmp, 1) == 0) {
            has_nan = true;
            break;
        }
    }
    #elif defined(__AVX__)
    npy_intp i;
    for(i = 0; i <= size_pred - 4 && !has_nan; i += 4) {
        __m256d v = _mm256_load_pd(pred_pattern + i);
        if (_mm256_movemask_pd(_mm256_cmp_pd(v, v, _CMP_UNORD_Q))) {
            has_nan = true;
            break;
        }
    }
    #else
    npy_intp i = 0;
    #endif

    // Handle remaining elements and non-SIMD case
    for(; i < size_pred && !has_nan; i++) {
        if(std::isnan(pred_pattern[i])) {
            has_nan = true;
            break;
        }
    }

    if (!has_nan) {
        #ifdef __ARM_NEON
        for(i = 0; i <= size_real - 2 && !has_nan; i += 2) {
            float64x2_t v = vld1q_f64(real_pattern + i);
            uint64x2_t cmp = vceqq_f64(v, v);
            if (vgetq_lane_u64(cmp, 0) == 0 || vgetq_lane_u64(cmp, 1) == 0) {
                has_nan = true;
                break;
            }
        }
        #elif defined(__AVX__)
        for(i = 0; i <= size_real - 4 && !has_nan; i += 4) {
            __m256d v = _mm256_load_pd(real_pattern + i);
            if (_mm256_movemask_pd(_mm256_cmp_pd(v, v, _CMP_UNORD_Q))) {
                has_nan = true;
                break;
            }
        }
        #else
        i = 0;
        #endif

        for(; i < size_real && !has_nan; i++) {
            if(std::isnan(real_pattern[i])) {
                has_nan = true;
                break;
            }
        }
    }

    if (!has_nan) {
        #ifdef __ARM_NEON
        for(i = 0; i <= size_x - 2 && !has_nan; i += 2) {
            float64x2_t v = vld1q_f64(pattern_x + i);
            uint64x2_t cmp = vceqq_f64(v, v);
            if (vgetq_lane_u64(cmp, 0) == 0 || vgetq_lane_u64(cmp, 1) == 0) {
                has_nan = true;
                break;
            }
        }
        #elif defined(__AVX__)
        for(i = 0; i <= size_x - 4 && !has_nan; i += 4) {
            __m256d v = _mm256_load_pd(pattern_x + i);
            if (_mm256_movemask_pd(_mm256_cmp_pd(v, v, _CMP_UNORD_Q))) {
                has_nan = true;
                break;
            }
        }
        #else
        i = 0;
        #endif

        for(; i < size_x && !has_nan; i++) {
            if(std::isnan(pattern_x[i])) {
                has_nan = true;
                break;
            }
        }
    }

    if (has_nan) {
        Py_DECREF(pred_pattern_arr);
        Py_DECREF(real_pattern_arr);
        Py_DECREF(pattern_x_arr);
        return Py_BuildValue("{s:O,s:O}", "real", Py_None, "predicted", Py_None);
    }

    // Check if patterns are equal using available SIMD
    bool patterns_equal = (size_pred == size_real);
    if (patterns_equal) {
        #ifdef __ARM_NEON
        for(i = 0; i <= size_pred - 2 && patterns_equal; i += 2) {
            float64x2_t v1 = vld1q_f64(pred_pattern + i);
            float64x2_t v2 = vld1q_f64(real_pattern + i);
            uint64x2_t cmp = vceqq_f64(v1, v2);
            if (vgetq_lane_u64(cmp, 0) == 0 || vgetq_lane_u64(cmp, 1) == 0) {
                patterns_equal = false;
                break;
            }
        }
        #elif defined(__AVX__)
        for(i = 0; i <= size_pred - 4 && patterns_equal; i += 4) {
            __m256d v1 = _mm256_load_pd(pred_pattern + i);
            __m256d v2 = _mm256_load_pd(real_pattern + i);
            if (_mm256_movemask_pd(_mm256_cmp_pd(v1, v2, _CMP_NEQ_OQ))) {
                patterns_equal = false;
                break;
            }
        }
        #else
        i = 0;
        #endif

        for(; i < size_pred && patterns_equal; i++) {
            if(pred_pattern[i] != real_pattern[i]) {
                patterns_equal = false;
                break;
            }
        }
    }

    double predictedCausalityStrength, realCausalityStrength;

    if(patterns_equal) {
        if(weighted) {
            // Pre-calculate norms
            const double pred_norm = norm_vec(predictedSignatureY_obj);
            const double real_norm = norm_vec(realSignatureY_obj);
            const double sig_x_norm = norm_vec(signatureX_obj);
            
            if(sig_x_norm > std::numeric_limits<double>::epsilon()) {
                const double pred_ratio = pred_norm / sig_x_norm;
                const double real_ratio = real_norm / sig_x_norm;
                predictedCausalityStrength = std::erf(pred_ratio);
                realCausalityStrength = std::erf(real_ratio);
            } else {
                predictedCausalityStrength = realCausalityStrength = 1.0;
            }
        } else {
            predictedCausalityStrength = realCausalityStrength = 1.0;
        }
    } else {
        predictedCausalityStrength = realCausalityStrength = 0.0;
    }

    // Clean up
    Py_DECREF(pred_pattern_arr);
    Py_DECREF(real_pattern_arr);
    Py_DECREF(pattern_x_arr);

    // Return results
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