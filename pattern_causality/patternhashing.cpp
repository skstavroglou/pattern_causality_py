#include <Python.h>
#include <vector>
#include <cmath>

// Helper function to generate possible patterns
static std::vector<std::vector<int>> possiblePatterns(int E) {
    std::vector<std::vector<int>> patterns;
    
    if (E <= 1) {
        return patterns; // Return empty vector
    }
    
    // Generate all possible combinations
    int numPatterns = pow(3, E-1);
    patterns.resize(numPatterns);
    
    for (int i = 0; i < numPatterns; i++) {
        std::vector<int> pattern;
        int num = i;
        
        for (int j = 0; j < E-1; j++) {
            pattern.push_back((num % 3) + 1);
            num /= 3;
        }
        
        patterns[i] = pattern;
    }
    
    return patterns;
}

// Helper function for factorial calculation
static int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n-1);
}

// Helper function for hashing
static int hashing(const std::vector<int>& vec) {
    int hash = 0;
    for (size_t i = 0; i < vec.size(); i++) {
        hash += vec[i] * factorial(i + 3);
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
    
    // Create Python list for results
    PyObject* result_list;
    if (patterns.empty()) {
        Py_RETURN_NONE;
    } else {
        result_list = PyList_New(patterns.size());
        for (size_t i = 0; i < patterns.size(); i++) {
            int hash_value = hashing(patterns[i]);
            PyList_SET_ITEM(result_list, i, PyLong_FromLong(hash_value));
        }
    }
    
    return result_list;
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
    return PyModule_Create(&patternhashing_module);
}