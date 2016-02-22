**Diffusion Approximation for Demographic Inference**

∂a∂i implements methods for demographic history and selection inference from genetic data, based on diffusion approximations to the allele frequency spectrum. One of ∂a∂i's main benefits is speed: fitting a two-population model typically takes around 10 minutes, and run time is independent of the number of SNPs in your data set. ∂a∂i is also flexible, handling up to three simultaneous populations, with arbitrary timecourses for population size and migration, plus the possibility of admixture and population-specific selection.

Originally ∂a∂i was initially developed by  Ryan Gutenkunst in the labs of Scott Williamson and Carlos Bustamante [http://www.bustamantelab.org] in Cornell's Department of Biological Statistics and Computational Biology. Ryan is now faculty in Molecular and Cellular Biology at the University of Arizona, and his group continues to work on ∂a∂i [http://gutengroup.mcb.arizona.edu].

If you use ∂a∂i in your research, please cite RN Gutenkunst, RD Hernandez, SH Williamson, CD Bustamante "Inferring the joint demographic history of multiple populations from multidimensional SNP data" PLoS Genetics 5:e1000695 (2009).

**Getting started**

Please see the wiki pages on Getting Started [https://bitbucket.org/GutenkunstLab/dadi/wiki/GettingStarted], Installation [https://bitbucket.org/GutenkunstLab/dadi/wiki/Installation], and our Data Format [ [https://bitbucket.org/GutenkunstLab/dadi/wiki/DataFormats].

Also, please sign up for our mailing list, where we help the community with ∂a∂i. Please do search the archives before posting. Many questions come up repeatedly. [http://groups.google.com/group/dadi-user]

**∂a∂i version 1.7.0 released**; Aug 14, 2015

After a long hiatus, a new release. The most important update is Godambe.py, which includes methods for computationally-efficient parameter uncertainty estimation and likelihood ratio tests. See the updated manual for more details.

**∂a∂i version 1.6.3 released**; Jul 12, 2012

This release improves `optimize_grid`, in response to a request by Xueqiu, and also adds the option to push optimization output to a file. It also includes a fix contributed by Simon Gravel for errors in extrapolation for very large spectra. Finally, spectra calculation for SNPs ascertained by sequencing a single individual has been added, in response to a request by Joe Pickrell.

[https://bitbucket.org/GutenkunstLab/dadi/wiki/OldNews]
