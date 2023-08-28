import multiprocessing as mp

def teston(arch):
    import tqdm
    import taichi as ti
    ti.init(arch=getattr(ti, arch))

    n = 1024
    pixels = ti.field(dtype=float, shape=(n, n))

    @ti.func
    def complex_sqr(z):
        return ti.Vector([z[0] ** 2 - z[1] ** 2, z[1] * z[0] * 2])

    @ti.kernel
    def paint():
        for i, j in pixels:
            c = ti.Vector([i / n, j / n])
            iterations = 0
            z = c
            while z.norm_sqr() < 4 and iterations < 50:
                z = complex_sqr(z) + c
                iterations += 1
            pixels[i, j] = 1 - iterations * 0.02

    for _ in tqdm.trange(100 if arch == 'cpu' else 1000, desc=repr(arch)):
        paint()
        pixels[0, 0]

p = mp.Process(target=teston, args=['cpu'])
p.start()
p.join()
p = mp.Process(target=teston, args=['vulkan'])
p.start()
p.join()
p = mp.Process(target=teston, args=['opengl'])
p.start()
p.join()
p = mp.Process(target=teston, args=['cuda'])
p.start()
p.join()
