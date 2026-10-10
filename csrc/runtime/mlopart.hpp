#pragma once

#include <unistd.h>

#include <initializer_list>
#include <string>

#include <Python.h>
#include <cuda.h>
#include <deep_jit/backend/cuda/driver.hpp>

#include "../utils/exception.hpp"
#include "jit.hpp"

namespace deep_gemm::mlopart {

// We have to use IPC to allocate localized memory via MLOPart prior to CUDA 13.4
// For convenience this is done in Python, called through the CPython limited API
// NOTES: the extension does not link pybind11, and ops may run with the GIL released
class PyObjectRef {
    PyObject* ptr;

public:
    explicit PyObjectRef(PyObject* ptr): ptr(ptr) {}
    PyObjectRef(const PyObjectRef&) = delete;
    PyObjectRef& operator=(const PyObjectRef&) = delete;
    ~PyObjectRef() { Py_XDECREF(ptr); }

    PyObject* get() const { return ptr; }

    PyObject* release() {
        const auto released = ptr;
        ptr = nullptr;
        return released;
    }
};

class GILGuard {
    PyGILState_STATE state;

public:
    GILGuard(): state(PyGILState_Ensure()) {}
    GILGuard(const GILGuard&) = delete;
    GILGuard& operator=(const GILGuard&) = delete;
    ~GILGuard() { PyGILState_Release(state); }
};

// Rethrow the pending Python exception, if any, as a host error
static void check_python_error() {
    if (not PyErr_Occurred())
        return;
    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    const PyObjectRef type_ref(type), value_ref(value), traceback_ref(traceback);
    std::string message = "MLOPart Python call failed";
    if (value != nullptr) {
        const PyObjectRef str(PyObject_Str(value));
        if (const auto utf8 = str.get() != nullptr ? PyUnicode_AsUTF8AndSize(str.get(), nullptr) : nullptr)
            message += std::string(": ") + utf8;
    }
    PyErr_Clear();
    DG_HOST_UNREACHABLE(message);
}

static PyObjectRef checked(PyObject* ptr) {
    if (ptr == nullptr)
        check_python_error();
    DG_HOST_ASSERT(ptr != nullptr);
    return PyObjectRef(ptr);
}

// `deep_gemm.utils.mlopart.<name>(*args)`, stealing the references of `args`
// NOTES: the caller must hold the GIL
static PyObjectRef call(const char* name, const std::initializer_list<PyObject*>& args) {
    PyObjectRef tuple(PyTuple_New(static_cast<Py_ssize_t>(args.size())));
    Py_ssize_t idx = 0;
    for (const auto arg: args) {
        if (tuple.get() == nullptr or arg == nullptr) {
            Py_XDECREF(arg);
        } else {
            PyTuple_SetItem(tuple.get(), idx, arg);
        }
        ++ idx;
    }
    check_python_error();
    DG_HOST_ASSERT(tuple.get() != nullptr);

    const auto module = checked(PyImport_ImportModule("deep_gemm.utils.mlopart"));
    const auto func = checked(PyObject_GetAttrString(module.get(), name));
    return checked(PyObject_Call(func.get(), tuple.get(), nullptr));
}

static PyObject* get_device_uuid() {
    const auto& uuid = jit->device.get_prop().uuid.bytes;
    return PyBytes_FromStringAndSize(uuid, sizeof(uuid));
}

static bool is_available() {
    const GILGuard gil;
    const auto result = call("is_available", {get_device_uuid()});
    const auto available = PyObject_IsTrue(result.get());
    check_python_error();
    return available == 1;
}

static CUmemGenericAllocationHandle create_memory(const size_t& num_bytes, const int& domain_idx) {
    int fd;
    {
        const GILGuard gil;
        const auto result = call("create_memory", {get_device_uuid(), PyLong_FromSize_t(num_bytes), PyLong_FromLong(domain_idx)});
        fd = static_cast<int>(PyLong_AsLong(result.get()));
        check_python_error();
    }
    CUmemGenericAllocationHandle handle = 0;
    DJ_CUDA_DRIVER_CHECK(deep_jit::cuda::driver::lazy_cuMemImportFromShareableHandle(
        &handle, reinterpret_cast<void*>(static_cast<intptr_t>(fd)), CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    close(fd);
    return handle;
}

static void release() {
    const GILGuard gil;
    call("release", {});
}

}  // namespace deep_gemm::mlopart
