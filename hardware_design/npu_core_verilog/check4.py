import golden_model_v2
t = golden_model_v2.Phase1InferenceTile()
x = golden_model_v2.to_int8(golden_model_v2.np.random.uniform(-5, 5, (16, 16)))
q_rope, k = t.forward(x)

def to_hex(arr):
    return [hex(int(v) & 0xFFFFFFFF) for v in arr]

print("K Row 0: ", to_hex(k[0]))
