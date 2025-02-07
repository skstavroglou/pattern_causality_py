#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <vector>
#include <cmath>
#include <numpy/arrayobject.h>
#include <algorithm>
#include <set>

// Helper function for factorial calculation
static double factorial(int n) {
    if (n <= 1) return 1.0;
    double result = 1.0;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

// Helper function to generate possible patterns
static std::vector<std::vector<int>> possiblePatterns(int E) {
    if (E <= 1) {
        return std::vector<std::vector<int>>();
    }
    
    // Calculate total number of combinations
    const int numPatterns = pow(3, E-1);
    std::vector<std::vector<int>> patterns(numPatterns);
    
    // Generate patterns using R's expand.grid logic
    for (int i = 0; i < numPatterns; ++i) {
        std::vector<int> pattern(E-1);
        int temp = i;
        
        // Fill pattern from right to left (least significant to most significant)
        for (int j = E-2; j >= 0; --j) {
            pattern[j] = (temp % 3) + 1;  // Convert to 1, 2, 3
            temp /= 3;
        }
        
        patterns[i] = pattern;
    }
    
    return patterns;
}

// Helper function for hashing - must match R implementation exactly
static double hashing(const std::vector<int>& vec) {
    double hash = 0.0;
    for (size_t i = 0; i < vec.size(); i++) {
        hash += static_cast<double>(vec[i]) * factorial(i + 2);
    }
    return hash;
}

// Main function: patternHashing
static PyObject* patternHashing(PyObject* self, PyObject* args) {
    int E;
    if (!PyArg_ParseTuple(args, "i", &E)) {
        return NULL;
    }

    std::vector<std::vector<int>> patterns = possiblePatterns(E);
    
    // Handle E <= 1 case
    if (patterns.empty()) {
        npy_intp dims[] = {0};
        return (PyObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    }
    
    // Calculate hash values
    std::vector<double> hash_values;
    hash_values.reserve(patterns.size());
    
    for (const auto& pattern : patterns) {
        hash_values.push_back(hashing(pattern));
    }
    
    // Create numpy array for results
    npy_intp dims[] = {static_cast<npy_intp>(hash_values.size())};
    PyArrayObject* result_array = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!result_array) {
        return NULL;
    }
    
    // Copy hash values to output array
    double* data = (double*)PyArray_DATA(result_array);
    std::copy(hash_values.begin(), hash_values.end(), data);
    
    return (PyObject*)result_array;
}

static PyMethodDef PatternHashingMethods[] = {
    {"patternhashing", patternHashing, METH_VARARGS, "Calculate pattern hashing"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef patternhashing_module = {
    PyModuleDef_HEAD_INIT,
    "patternhashing",
    NULL,
    -1,
    PatternHashingMethods
};

PyMODINIT_FUNC PyInit_patternhashing(void) {
    import_array();
    return PyModule_Create(&patternhashing_module);
}