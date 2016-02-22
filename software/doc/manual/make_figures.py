import numpy, pylab
import dadi
import OutOfAfrica

data = dadi.Spectrum.from_file('YRI.CEU.CHB.fs')

params = numpy.array([1.68202    ,  0.287184   ,  0.129468   ,  3.73683    ,
                      0.0700931  ,  7.29205    ,  3.64657    ,  0.440549   ,  
                      0.280983   ,  1.39873    ,  0.211077   ,  0.337956   ,
                      0.0579698  ])

func_ex = dadi.Numerics.make_extrap_log_func(OutOfAfrica.OutOfAfrica)
#model = func_ex(params, data.sample_sizes, [40,50,60])

fig = pylab.figure(1)
fig.clear()
dadi.Plotting.plot_1d_comp_multinom(model.marginalize([1,2]), 
                                    data.marginalize([1,2]))
fig.savefig('1d_comp.pdf')

fig = pylab.figure(2)
fig.clear()
dadi.Plotting.plot_single_2d_sfs(data.marginalize([2]), vmin=1)
fig.savefig('2d_single.pdf')

fig = pylab.figure(3)
fig.clear()
dadi.Plotting.plot_2d_comp_multinom(model.marginalize([2]), 
                                    data.marginalize([2]), vmin=1)
fig.savefig('2d_comp.pdf')

fig = pylab.figure(4, figsize=(8,10))
fig.clear()
dadi.Plotting.plot_3d_comp_multinom(model, data, vmin=1)
fig.savefig('3d_comp.pdf')

pylab.show()
