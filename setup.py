from setuptools import setup, find_packages, Extension
import numpy as np
import platform


def get_long_description():
    with open("README.md", "r", encoding="utf-8") as f:
        content = f.read()
        return content


long_description = get_long_description()

extensions = [
    Extension(
        "utils.statespace",
        [
            "pattern_causality/statespace.cpp",
        ],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11"],
    ),
    Extension(
        "utils.patternhashing",
        [
            "pattern_causality/patternhashing.cpp",
        ],
        language="c++",
        extra_compile_args=["-std=c++11"],
    ),
    Extension(
        "utils.signaturespace",
        [
            "pattern_causality/signaturespace.cpp",
        ],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11"],
    ),
    Extension(
        "utils.distancematrix",
        [
            "pattern_causality/distancematrix.cpp",
        ],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11"],
    ),
    Extension(
        "utils.patternspace",
        [
            "pattern_causality/patternspace.cpp",
        ],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11"],
    ),
    Extension(
        "utils.pastNNs",
        [
            "pattern_causality/pastNNs.cpp",
        ],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11"],
    ),
    Extension(
        "utils.projectedNNs",
        [
            "pattern_causality/projectedNNs.cpp",
        ],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11"],
    ),
    Extension(
        "utils.predictionY",
        ["pattern_causality/predictionY.cpp"],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11"],
        extra_link_args=(
            ["-undefined", "dynamic_lookup"] if platform.system() == "Darwin" else []
        ),
    ),
    Extension(
        "utils.fillPCMatrix",
        ["pattern_causality/fillPCMatrix.cpp"],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11"],
    ),
    Extension(
        "utils.natureOfCausality",
        ["pattern_causality/natureOfCausality.cpp"],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11"],
    ),
    Extension(
        "utils.databank",
        ["pattern_causality/databank.cpp"],
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11"],
    ),
    Extension(
        "utils.fcp",
        [
            "pattern_causality/fcp.cpp",
        ],
        language="c++",
        extra_compile_args=["-std=c++11"],
    ),
]

setup(
    name="pattern_causality",
    version="0.0.3",
    description="Pattern Causality Algorithm in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Stavros Stavroglou, Athanasios Pantelous, Hui Wang",
    author_email="stavros.k.stavroglou@gmail.com,Athanasios.Pantelous@monash.edu, huiw1128@gmail.com",
    maintainer="Hui Wang",
    maintainer_email="huiw1128@gmail.com",
    url="https://github.com/skstavroglou/pattern_causality_py",
    install_requires=["numpy", "pandas"],
    setup_requires=["numpy", "pandas"],
    test_suite="tests",
    tests_require=["pytest", "pytest-cov"],
    license="BSD License",
    packages=find_packages("."),
    package_data={
        "pattern_causality": ["data/*.csv"],
    },
    include_package_data=True,
    platforms=["all"],
    ext_modules=extensions,
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
