import torch


__all__ = [
    "histogram",
    "histogram2d",
]


def marginal_pdf(values, bins, sigma, epsilon=1e-10):
    """
        Function that calculates the marginal probability distribution function of the input tensor
        based on the number of histogram bins.
    """

    if not isinstance(values, torch.Tensor):
        raise TypeError("Input values type is not a torch.Tensor. Got {}"
                        .format(type(values)))

    if not isinstance(bins, torch.Tensor):
        raise TypeError("Input bins type is not a torch.Tensor. Got {}"
                        .format(type(bins)))

    if not isinstance(sigma, torch.Tensor):
        raise TypeError("Input sigma type is not a torch.Tensor. Got {}"
                        .format(type(sigma)))

    if not values.dim() == 3:
        raise ValueError("Input values must be a of the shape BxNx1."
                         " Got {}".format(values.shape))

    if not bins.dim() == 1:
        raise ValueError("Input bins must be a of the shape NUM_BINS"
                         " Got {}".format(bins.shape))

    if not sigma.dim() == 0:
        raise ValueError("Input sigma must be a of the shape 1"
                         " Got {}".format(sigma.shape))

    residuals = values - bins.unsqueeze(0).unsqueeze(0)
    kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))

    pdf = torch.mean(kernel_values, dim=1)
    normalization = torch.sum(pdf, dim=1).unsqueeze(1) + epsilon
    pdf = pdf / normalization

    return (pdf, kernel_values)


def joint_pdf(kernel_values1, kernel_values2, epsilon=1e-10):
    """
        Function that calculates the joint probability distribution function of the input tensors
        based on the number of histogram bins.
    """

    if not isinstance(kernel_values1, torch.Tensor):
        raise TypeError("Input kernel_values1 type is not a torch.Tensor. Got {}"
                        .format(type(kernel_values1)))

    if not isinstance(kernel_values2, torch.Tensor):
        raise TypeError("Input kernel_values2 type is not a torch.Tensor. Got {}"
                        .format(type(kernel_values2)))

    if not kernel_values1.dim() == 3:
        raise ValueError("Input kernel_values1 must be a of the shape BxN."
                         " Got {}".format(kernel_values1.shape))

    if not kernel_values2.dim() == 3:
        raise ValueError("Input kernel_values2 must be a of the shape BxN."
                         " Got {}".format(kernel_values2.shape))

    if kernel_values1.shape != kernel_values2.shape:
        raise ValueError("Inputs kernel_values1 and kernel_values2 must have the same shape."
                         " Got {} and {}".format(kernel_values1.shape, kernel_values2.shape))

    joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
    normalization = torch.sum(joint_kernel_values, dim=(1, 2)).view(-1, 1, 1) + epsilon
    pdf = joint_kernel_values / normalization

    return pdf


def histogram(x, bins, bandwidth, epsilon=1e-10):
    """
        Function that estimates the histogram of the input tensor.
        The calculation uses kernel density estimation which requires a bandwidth (smoothing) parameter.
    """

    pdf, _ = marginal_pdf(x.unsqueeze(2), bins, bandwidth, epsilon)

    return pdf


def histogram2d(x1, x2, bins, bandwidth, epsilon=1e-10):
    """
        Function that estimates the 2d histogram of the input tensor.
        The calculation uses kernel density estimation which requires a bandwidth (smoothing) parameter.

    """

    pdf1, kernel_values1 = marginal_pdf(x1.unsqueeze(2), bins, bandwidth, epsilon)
    pdf2, kernel_values2 = marginal_pdf(x2.unsqueeze(2), bins, bandwidth, epsilon)

    pdf = joint_pdf(kernel_values1, kernel_values2)

    return pdf