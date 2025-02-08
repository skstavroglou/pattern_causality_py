#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <string>
#include <vector>
#include <limits>

static PyObject* databank(PyObject* self, PyObject* args) {
    const char* type_name;
    PyObject* dimensions_obj;

    // Parse arguments
    if (!PyArg_ParseTuple(args, "sO", &type_name, &dimensions_obj)) {
        return NULL;
    }

    // Convert dimensions to vector
    std::vector<npy_intp> dimensions;
    if (PyList_Check(dimensions_obj) || PyTuple_Check(dimensions_obj)) {
        Py_ssize_t size = PySequence_Size(dimensions_obj);
        dimensions.reserve(size);
        for (Py_ssize_t i = 0; i < size; i++) {
            PyObject* item = PySequence_GetItem(dimensions_obj, i);
            dimensions.push_back(PyLong_AsLong(item));
            Py_DECREF(item);
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "dimensions must be a list or tuple");
        return NULL;
    }

    std::string type(type_name);

    if (type == "array") {
        npy_intp* dims = dimensions.data();
        PyObject* arr = PyArray_EMPTY(dimensions.size(), dims, NPY_DOUBLE, 0);
        double* data = (double*)PyArray_DATA((PyArrayObject*)arr);
        for (npy_intp i = 0; i < PyArray_SIZE((PyArrayObject*)arr); i++) {
            data[i] = std::numeric_limits<double>::quiet_NaN();
        }
        return arr;
    }
    else if (type == "vector") {
        npy_intp dims[1] = {dimensions[0]};
        PyObject* arr = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
        double* data = (double*)PyArray_DATA((PyArrayObject*)arr);
        for (npy_intp i = 0; i < dimensions[0]; i++) {
            data[i] = std::numeric_limits<double>::quiet_NaN();
        }
        return arr;
    }
    else if (type == "matrix") {
        npy_intp dims[2] = {dimensions[0], dimensions[1]};
        PyObject* arr = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
        double* data = (double*)PyArray_DATA((PyArrayObject*)arr);
        for (npy_intp i = 0; i < dimensions[0] * dimensions[1]; i++) {
            data[i] = std::numeric_limits<double>::quiet_NaN();
        }
        return arr;
    }
    else if (type == "neighborhood memories") {
        // Validate dimensions
        npy_intp expected_cols = 1 + 4 * dimensions[2] + (dimensions[3] - 1) * dimensions[2] + dimensions[3] * dimensions[2];
        if (dimensions[1] != expected_cols) {
            PyErr_SetString(PyExc_ValueError, "The dimensions[1] is wrong!");
            return NULL;
        }

        // Create empty DataFrame equivalent (numpy array)
        npy_intp dims[2] = {dimensions[0], dimensions[1]};
        PyObject* arr = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
        if (!arr) return NULL;

        // Fill with NaN
        double* data = (double*)PyArray_DATA((PyArrayObject*)arr);
        for (npy_intp i = 0; i < dimensions[0] * dimensions[1]; i++) {
            data[i] = std::numeric_limits<double>::quiet_NaN();
        }

        // Create list for column names
        PyObject* col_names = PyList_New(dimensions[1]);
        if (!col_names) {
            Py_DECREF(arr);
            return NULL;
        }

        // Add column names
        int col_idx = 0;
        
        // "i" column
        PyList_SET_ITEM(col_names, col_idx++, PyUnicode_FromString("i"));

        // nn-times, nn-dists, nn-weights, nn-patt
        for (int j = 0; j < 4; j++) {
            const char* prefix;
            switch(j) {
                case 0: prefix = "nn-times"; break;
                case 1: prefix = "nn-dists"; break;
                case 2: prefix = "nn-weights"; break;
                case 3: prefix = "nn-patt"; break;
            }
            for (npy_intp k = 0; k < dimensions[2]; k++) {
                PyList_SET_ITEM(col_names, col_idx++, PyUnicode_FromString(prefix));
            }
        }

        // Signature component columns
        for (npy_intp nn = 1; nn <= dimensions[2]; nn++) {
            for (npy_intp comp = 1; comp < dimensions[3]; comp++) {
                char buf[100];
                snprintf(buf, sizeof(buf), "Sig-Comp.%ld of NN%ld", (long)comp, (long)nn);
                PyList_SET_ITEM(col_names, col_idx++, PyUnicode_FromString(buf));
            }
        }

        // Coordinate columns
        for (npy_intp nn = 1; nn <= dimensions[2]; nn++) {
            for (npy_intp coord = 1; coord <= dimensions[3]; coord++) {
                char buf[100];
                snprintf(buf, sizeof(buf), "Coord.%ld of NN%ld", (long)coord, (long)nn);
                PyList_SET_ITEM(col_names, col_idx++, PyUnicode_FromString(buf));
            }
        }

        // Import pandas
        PyObject* pandas = PyImport_ImportModule("pandas");
        if (!pandas) {
            Py_DECREF(arr);
            Py_DECREF(col_names);
            return NULL;
        }

        // Create DataFrame
        PyObject* df_class = PyObject_GetAttrString(pandas, "DataFrame");
        PyObject* df = PyObject_CallFunction(df_class, "OO", arr, col_names);
        
        Py_DECREF(pandas);
        Py_DECREF(df_class);
        Py_DECREF(arr);
        Py_DECREF(col_names);

        return df;
    }

    Py_RETURN_NONE;
}

static PyMethodDef DatabankMethods[] = {
    {"databank", databank, METH_VARARGS, "Create data structures based on type and dimensions"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef databankmodule = {
    PyModuleDef_HEAD_INIT,
    "databank",
    NULL,
    -1,
    DatabankMethods
};

PyMODINIT_FUNC PyInit_databank(void) {
    import_array();
    return PyModule_Create(&databankmodule);
}
