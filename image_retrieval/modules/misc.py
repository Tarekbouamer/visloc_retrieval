def mask_border(m, b: int, v):
  """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)

  """

  m[:, :b] = v
  m[:, :, :b] = v
  m[:, :, :, :b] = v
  m[:, :, :, :, :b] = v
  m[:, -b:0] = v
  m[:, :, -b:0] = v
  m[:, :, :, -b:0] = v
  m[:, :, :, :, -b:0] = v




def mask_border_with_padding(m, bd, v, p_m0, p_m1):
  m[:, :bd] = v
  m[:, :, :bd] = v
  m[:, :, :, :bd] = v
  m[:, :, :, :, :bd] = v

  h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
  h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
  
  for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
    m[b_idx, h0-bd:] = v
    m[b_idx, :, w0-bd:] = v
    m[b_idx, :, :, h1-bd:] = v
    m[b_idx, :, :, :, w1-bd:] = v