from setuptools import setup, find_packages, Extension
import os
import platform
import sys
import numpy as np
import sysconfig

def get_compiler_args():
    """Get platform-specific compiler arguments."""
    system = platform.system().lower()
    
    if system == "darwin":  # macOS
        return {
            "extra_compile_args": [
                "-O3",
                "-fPIC",
                "-std=c++11",
                "-stdlib=libc++",
                "-mmacosx-version-min=10.9",
                "-Wno-unused-function",
                "-Wno-unused-variable",
                "-Wno-deprecated-declarations",
                "-Wno-c++11-narrowing",
                "-v",
            ],
            "extra_link_args": [
                "-stdlib=libc++",
                "-mmacosx-version-min=10.9",
                "-v",
            ]
        }
    elif system == "linux":
        return {
            "extra_compile_args": [
                "-O3",
                "-fPIC",
                "-std=c++11",
                "-v",
            ],
            "extra_link_args": ["-v"]
        }
    elif system == "windows":
        return {
            "extra_compile_args": ["/O2", "/W3", "/EHsc", "/std:c++11", "/verbose"],
            "extra_link_args": ["/verbose"]
        }
    else:
        return {
            "extra_compile_args": ["-O3", "-fPIC", "-std=c++11", "-v"],
            "extra_link_args": ["-v"]
        }

def get_include_dirs():
    """Get include directories for compilation."""
    cpp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pattern_causality", "cpp")
    include_dirs = [
        np.get_include(),
        cpp_dir,
        sysconfig.get_path('include'),
    ]
    
    # Add platform-specific include directories
    if platform.system() == "Darwin":
        mac_dirs = [
            "/usr/local/include",
            "/usr/include",
            "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include",
        ]
        include_dirs.extend(d for d in mac_dirs if os.path.exists(d))
    
    return include_dirs

def get_extensions():
    """Get the list of C++ extensions to be built."""
    cpp_dir = os.path.join("pattern_causality", "cpp")
    include_dirs = get_include_dirs()
    
    compiler_args = get_compiler_args()
    
    extensions = []
    cpp_files = [
        "statespace",
        "patternhashing",
        "signaturespace",
        "distancematrix",
        "patternspace",
        "pastNNs",
        "projectedNNs",
        "predictionY",
        "fillPCMatrix",
        "natureOfCausality",
        "databank",
        "fcp"
    ]
    
    for cpp_file in cpp_files:
        ext = Extension(
            f"utils.{cpp_file}",
            sources=[f"pattern_causality/cpp/{cpp_file}.cpp"],
            language="c++",
            include_dirs=include_dirs,
            extra_compile_args=compiler_args["extra_compile_args"],
            extra_link_args=compiler_args["extra_link_args"]
        )
        extensions.append(ext)
            
    return extensions

# Print build environment information
print("\nBuild Environment:")
print(f"Platform: {platform.system()} {platform.machine()}")
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"Compiler: {sysconfig.get_config_var('CC')}")

# Read README
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pattern-causality",
    version="1.0.3",
    description="Pattern Causality Algorithm in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Stavros Stavroglou, Athanasios Pantelous, Hui Wang",
    author_email="huiw1128@gmail.com",
    url="https://github.com/skstavroglou/pattern_causality_py",
    packages=find_packages(),
    package_dir={"": "."},
    package_data={
        'pattern_causality': [
            'cpp/*.cpp',
            'cpp/*.h',
            'cpp/*.hpp',
            'cpp/*.so',
            'cpp/*.dylib',
            'data/*.csv'
        ],
    },
    ext_modules=get_extensions(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.0.0",
    ],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering",
    ],
)
