import numpy as np
np.random.seed(42)
x_int8 = np.random.randint(-128, 127, size=(16, 16)).astype(np.int8)
W_q = np.random.randint(-128, 127, size=(16, 16)).astype(np.int8)
q_int32 = np.dot(x_int8.astype(np.int32), W_q.astype(np.int32))
print("Row 0:")
print([hex(int(x) & 0xFFFFFFFF) for x in q_int32[0]])
