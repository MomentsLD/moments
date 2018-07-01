import moments

# 1D
# Case 1
fs = moments.Demographics1D.snm([5])
fs.integrate([100], 1)

# Case 2
fs = moments.Demographics1D.snm([50])
fs.integrate([100], 1)


# 2D
# Case 1
fs = moments.Demographics2D.snm([5, 8])
fs.integrate([2, 2], 1)

# Case 2
fs = moments.Demographics2D.snm([50, 80])
fs.integrate([2, 2], 1)


# 3D
fs = moments.Demographics2D.snm([40, 20])
fs = moments.Manips.split_2D_to_3D_2(fs, 10, 10)
fs.integrate([10, 10, 10], 1)

# 4D
fs = moments.Demographics2D.snm([20, 40])
fs = moments.Manips.split_2D_to_3D_2(fs, 20, 20)
fs = moments.Manips.split_3D_to_4D_3(fs, 10, 10)
fs.integrate([10,10,10,10], 1)


# 5D
fs = moments.Demographics2D.snm([20, 40])
fs = moments.Manips.split_2D_to_3D_2(fs, 20, 20)
fs = moments.Manips.split_3D_to_4D_3(fs, 10, 10)
fs = moments.Manips.split_4D_to_5D_3(fs, 5, 5)
fs.integrate([10,10,10,10, 10], 1)
