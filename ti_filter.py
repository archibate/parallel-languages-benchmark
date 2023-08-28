import tqdm
import multiprocessing as mp

def teston(arch):
    import taichi as ti
    ti.init(arch=getattr(ti, arch), cpu_max_num_threads=1)

    n = 32*1024*1024

    dat = ti.field(shape=n, dtype=ti.f32)
    out = ti.field(shape=n, dtype=ti.f32)
    out_n = ti.field(shape=(), dtype=ti.i32)

    @ti.kernel
    def init():
        for i in dat:
            dat[i] = ti.random()

    @ti.kernel
    def filterabove(val: ti.f32, arch: ti.template()):
        if ti.static(arch == 'cpu'):
            out_n[None] = 0
            for i in dat:
                if dat[i] > val:
                    x = out_n[None]
                    out[x] = dat[i]
                    out_n[None] = x + 1
        else:
            out_n[None] = 0
            for i in dat:
                if dat[i] > val:
                    out[ti.atomic_add(out_n[None], 1)] = dat[i]

    init()
    filterabove(0.5, arch)

    for _ in tqdm.trange(30 if arch == 'cpu' else 1000, desc=repr(arch)):
        filterabove(0.5, arch)
        out_n[None]

    print(dat.to_numpy())
    print(out.to_numpy()[:out_n[None]])

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
