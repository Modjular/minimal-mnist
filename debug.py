import numpy as np

m = np.load("/Users/tony/Documents/Github/minimal-mnist/best-model.npy", allow_pickle=True)
m = m.item()

convolve_template = '''      <feConvolveMatrix
        in="SourceGraphic"
        result="fc0.0.weight_{}"
        bias="{}"
        order="28"
        targetX="0"
        targetY="0"
        preserveAlpha="true"
        kernelMatrix="{}"
      />'''

SCALE_FACTOR = 512

# Print a grid of weights normaized [0, 256]
for n in range(10):

    bias = m["fc1.0.bias"][n]
    weights = m["fc1.0.weight"][n]

    bias = str(round(bias * SCALE_FACTOR))
    weights = " ".join([str(round(w * SCALE_FACTOR)) for w in weights])
    
    print(convolve_template.format(n, bias, weights))