import matplotlib.pyplot as plt

# Data
data = {
    'ti.cpu': 79.20,
    'ti.vulkan': 1946.08,
    'ti.opengl': 3250.63,
    'ti.cuda': 3399.18,
    'sycl.openmp': 68.56,
    'sycl.cuda': 3824.29,
    'clang.serial': 21.32,
    'clang.serialsimd': 68.71,
    'clang.openmp': 88.31,
    'clang.openmpsimd': 401.22,
    'clang.tbb': 110.70,
    'clang.tbbsimd': 491.00,
    'mojo.serial': 20.60/20.68*21.32,
    'mojo.serialsimd': 201.26/99.83*68.71,
    'mojo.parasimd': 8.51/2492.72*401.22,
}

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(data.keys(), data.values())

# Formatting
plt.title('Iterations per Second')
plt.xlabel('Data Set')
plt.ylabel('Iterations per Second (it/s)')
plt.xticks(rotation=90)

# Displaying the histogram
plt.show()
