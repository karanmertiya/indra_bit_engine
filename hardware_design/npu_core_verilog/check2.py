import golden_model_v2
t = golden_model_v2.Phase1InferenceTile()
x = golden_model_v2.to_int8(golden_model_v2.np.random.uniform(-5, 5, (16, 16)))
q_rope, k_int32 = t.forward(x)
q_int32 = golden_model_v2.np.dot(x.astype(golden_model_v2.np.int32), t.W_q.astype(golden_model_v2.np.int32))

def to_hex(arr):
    return [hex(int(v) & 0xFFFFFFFF) for v in arr]

print("Row 0 q_int32: ", to_hex(q_int32[0]))
print("Row 15 q_int32: ", to_hex(q_int32[15]))
print("Row 0 q_rope: ", to_hex(q_rope[0]))
print("Row 15 q_rope: ", to_hex(q_rope[15]))
