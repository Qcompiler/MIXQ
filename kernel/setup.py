import os
import torch
from pathlib import Path
from setuptools import setup, find_packages
from distutils.sysconfig import get_python_lib
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CUDAExtension

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

try:
    CUDA_VERSION = "".join(os.environ.get("CUDA_VERSION", torch.version.cuda).split("."))
except Exception as ex:
    raise RuntimeError("Your system must have an Nvidia GPU")

common_setup_kwargs = {
    "version": f"0.1.6+cu{CUDA_VERSION}",
    "name": "mixlib",
    "author": "Jidong Zhai; Yidong Chen; Tsinghua University",
    "license": "MIT",
    "python_requires": ">=3.8.0",
    "long_description_content_type": "text/markdown",
    "platforms": ["linux", "windows"],
    "classifiers": [
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
    ]
}

requirements = [
    "torch>=2.0.0",
    "transformers>=4.34.0",
    "tokenizers>=0.12.1",
    "accelerate",
    "sentencepiece",
    "lm_eval",
    "texttable",
    "toml",
    "attributedict",
    "protobuf",
    "torchvision",
    "tabulate"
]

def get_include_dirs():
    include_dirs = []

    conda_cuda_include_dir = os.path.join(get_python_lib(), "nvidia/cuda_runtime/include")
    if os.path.isdir(conda_cuda_include_dir):
        include_dirs.append(conda_cuda_include_dir)

    
    this_dir = os.path.dirname(os.path.abspath(__file__))
    include_dirs.append(this_dir)
    include_dirs.append(os.path.join(this_dir,"common"))
    include_dirs.append("cutlass/include")


    return include_dirs

def get_generator_flag():
    generator_flag = []
    torch_dir = torch.__path__[0]
    if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
        generator_flag = ["-DOLD_GENERATOR_PATH"]
    
    return generator_flag

def check_dependencies():
    if CUDA_HOME is None:
        raise RuntimeError(
            f"Cannot find CUDA_HOME. CUDA must be available to build the package.")

def get_compute_capabilities():
    # Collect the compute capabilities of all available GPUs.
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cc = major * 10 + minor

        if cc < 75:
            raise RuntimeError("GPUs with compute capability less than 7.5 are not supported.")

    # figure out compute capability
    compute_capabilities = { 80, 86, 89, 90}

    capability_flags = []
    for cap in compute_capabilities:
        capability_flags += ["-gencode", f"arch=compute_{cap},code=sm_{cap}"]

    capability_flags += ["-gencode", f"arch=compute_90a,code=sm_90a"]
    return capability_flags

check_dependencies()
include_dirs = get_include_dirs()
generator_flags = get_generator_flag()
arch_flags = get_compute_capabilities()

print(include_dirs)
if os.name == "nt":
    include_arch = os.getenv("INCLUDE_ARCH", "1") == "1"

    # Relaxed args on Windows
    if include_arch:
        extra_compile_args={"nvcc": arch_flags}
    else:
        extra_compile_args={}
else:
    extra_compile_args={
        "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"],
        "nvcc": [
            "-O3", 
            "-std=c++17",
            "-DENABLE_BF16",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
        ] + arch_flags + generator_flags
    }

extensions = [
    CUDAExtension(
        "mixlib",
        [
            "mix_cuda/pybind_mix.cpp",
            "mix_cuda/cult.cu",
             "mix_cuda/gemmdequant.cu",
            "mix_cuda/mma_permutated.cu",
            "mix_cuda/layernorm/layernorm.cu",
            #"mix_cuda/cutlassmix.cu",
        ], 

        
        
        extra_compile_args=extra_compile_args
    )
]



additional_setup_kwargs = {
    "ext_modules": extensions,
    "cmdclass": {'build_ext': BuildExtension}
}

common_setup_kwargs.update(additional_setup_kwargs)


setup(
    packages=find_packages(),
    install_requires=requirements,
    include_dirs=include_dirs,
    **common_setup_kwargs
)
