imort datetime

s_time = datetime.datetime.now()
y = cupy.arange(2048, dtype=float)
cupy.fft.fft(y)
e_time = datetime.datetime.now()
print('the fft time stamp is: ', int((e_time - s_time).total_seconds() * 1000))
