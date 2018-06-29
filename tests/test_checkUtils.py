import moments

# 1D
# Should report error
print "######################################### 1D #########################################"
print "*********************** n = {} ***********************".format(5)
fs = moments.Demographics1D.snm([5])
fs.integrate([100], 1)

print "*********************** n = {} ***********************".format(50)
fs = moments.Demographics1D.snm([50])
fs.integrate([100], 1)
print "If it doesn't print anything, there's no error above the predefined threshold during integration!"

print "######################################### 2D #########################################"
print "*********************** n1 = {}, n2 = {} ***********************".format(5, 8)
fs = moments.Demographics2D.snm([5, 8])
fs.integrate([2, 2], 1)

print "*********************** n1 = {}, n2 = {} ***********************".format(50, 80)
fs = moments.Demographics2D.snm([50, 80])
fs.integrate([2, 2], 1)
print "If it doesn't print anything, there's no error above the predefined threshold during integration!"
