import numpy as np

svg_template = '''  <svg
    xmlns="http://www.w3.org/2000/svg"
    version="1.1"
    height="120"
    width="120"
  >
    <filter
      id="sobel1"
      x="0"
      y="0"
      width="100%"
      height="100%"
      color-interpolation-filters="sRGB"
    >
      <!-- === 01 === INPUT LAYER: (784, 10)-->
{}
      <!-- === 02 === HIDDEN LAYER: (10, 15)-->
{}
      <!-- === 01 === HIDDEN LAYER: (15, 10)-->
{}
      <!-- === 01 === OUTPUT LAYER: (10)-->

      <feBlend mode="normal" in="fc0.0.weight_0" in2="SourceGraphic" />
    </filter>
    <image
      href="https://datasets-server.huggingface.co/assets/mnist/--/mnist/train/7/image/image.jpg"
      width="28"
      height="28"
      style="filter: url(#sobel1)"
    />
  </svg>
'''

fc0_convolve_template = '''      <feConvolveMatrix
        in="SourceGraphic"
        result="fc0.0.weight_{}"
        bias="{}"
        order="28"
        targetX="0"
        targetY="0"
        preserveAlpha="true"
        kernelMatrix="{}"
      />'''

fc0_blend_template = '''
      <feConvolveMatrix
        in="SourceGraphic"
        result="fc0.0.weight_{}"
        bias="{}"
        order="28"
        targetX="0"
        targetY="0"
        preserveAlpha="true"
        kernelMatrix="{}"
      />
'''




DIVISOR = 512
layers = [
    ["fc0.0.weight", "fc0.0.bias"],
    ["fc1.0.weight", "fc1.0.bias"],
    ["fc2.weight", "fc2.bias"],
]

m = np.load("/Users/tony/Documents/Github/minimal-mnist/best-model.npy", allow_pickle=True)
m = m.item()

for w, b in layers:

    weights = m[w]
    biases = m[b]

    for n in range(weights.shape[0]):

        bias = biases[n]
        weights = weights[n]

        bias = str(round(bias * DIVISOR))
        weights = " ".join([str(round(w * DIVISOR)) for w in weights])
        
        print(fc0_convolve_template.format(n, bias, weights))