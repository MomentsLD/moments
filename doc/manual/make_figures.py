import numpy, pylab
import moments
import OutOfAfrica

data = moments.Spectrum.from_file('YRI_CEU_CHB.fs')

params = numpy.array([2.10065897, 0.25066579, 0.22247642, 3.05297944,
					  0.09022469, 5.82773903, 3.79104318, 0.25730946,
					  0.12569788, 1.07182332, 0.36429414, 0.1108222, 
					  0.07072507])

model = OutOfAfrica.OutOfAfrica(params, data.sample_sizes)

fig = pylab.figure(1)
fig.clear()
moments.Plotting.plot_1d_comp_multinom(model.marginalize([1,2]), 
                                    data.marginalize([1,2]))
fig.savefig('1d_comp.pdf')

fig = pylab.figure(2)
fig.clear()
moments.Plotting.plot_single_2d_sfs(data.marginalize([2]), vmin=1)
fig.savefig('2d_single.pdf')

fig = pylab.figure(3)
fig.clear()
moments.Plotting.plot_2d_comp_multinom(model.marginalize([2]), 
                                    data.marginalize([2]), vmin=1)
fig.savefig('2d_comp.pdf')

fig = pylab.figure(4, figsize=(8,10))
fig.clear()
moments.Plotting.plot_3d_comp_multinom(model, data, vmin=1)
fig.savefig('3d_comp.pdf')

pylab.show()
