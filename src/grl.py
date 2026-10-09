import torch

from torch.autograd import Function


# ============================================================
# GRADIENT REVERSAL FUNCTION
# ============================================================

class _GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL).

    Forward:
        identity operation

            y = x

    Backward:
        reverses the gradient direction

            dL/dx = -strength * dL/dy

    This allows a domain classifier to minimize its own
    classification loss while encouraging the shared feature
    extractor to produce domain-confusing embeddings.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        strength,
    ):

        ctx.strength = float(
            strength
        )

        return x.view_as(
            x
        )

    @staticmethod
    def backward(
        ctx,
        grad_output,
    ):

        grad_input = (
            -ctx.strength
            * grad_output
        )

        return (
            grad_input,
            None,
        )


# ============================================================
# PUBLIC GRL FUNCTION
# ============================================================

def grad_reverse(
    x,
    strength=1.0,
):
    """
    Apply Gradient Reversal Layer.

    Parameters
    ----------
    x : torch.Tensor
        Input embedding.

    strength : float
        Gradient reversal strength.

        Forward output remains unchanged.

        Backward gradient becomes:

            -strength * gradient

    Returns
    -------
    torch.Tensor
        Tensor with identical forward values but reversed
        gradient during backpropagation.
    """

    if not torch.is_tensor(
        x
    ):

        raise TypeError(
            "x must be a torch.Tensor."
        )

    if strength < 0:

        raise ValueError(
            "GRL strength must be >= 0."
        )

    return (
        _GradientReversalFunction
        .apply(
            x,
            float(
                strength
            ),
        )
    )


# ============================================================
# OPTIONAL MODULE WRAPPER
# ============================================================

class GradientReversalLayer(
    torch.nn.Module
):
    """
    nn.Module wrapper around grad_reverse().

    This is optional but useful when GRL needs to be inserted
    directly inside a PyTorch model definition.
    """

    def __init__(
        self,
        strength=1.0,
    ):

        super().__init__()

        if strength < 0:

            raise ValueError(
                "GRL strength must be >= 0."
            )

        self.strength = float(
            strength
        )

    def forward(
        self,
        x,
    ):

        return grad_reverse(
            x,
            strength=
                self.strength,
        )


# ============================================================
# SELF TEST
# ============================================================

def _self_test():

    print(
        "=" * 72
    )

    print(
        "GRADIENT REVERSAL LAYER SELF TEST"
    )

    print(
        "=" * 72
    )

    # ========================================================
    # INPUT
    # ========================================================

    x = torch.tensor(
        [
            1.0,
            2.0,
            3.0,
        ],
        requires_grad=True,
    )

    strength = 0.5

    # ========================================================
    # FORWARD
    # ========================================================

    y = grad_reverse(
        x,
        strength=
            strength,
    )

    print()

    print(
        "Input:"
    )

    print(
        x.detach()
    )

    print()

    print(
        "GRL output:"
    )

    print(
        y.detach()
    )

    # Forward must be identity.

    forward_ok = torch.allclose(
        x.detach(),
        y.detach(),
    )

    # ========================================================
    # BACKWARD
    # ========================================================

    loss = y.sum()

    loss.backward()

    expected_gradient = torch.full_like(
        x,
        -strength,
    )

    gradient_ok = torch.allclose(
        x.grad,
        expected_gradient,
    )

    print()

    print(
        "Observed gradient:"
    )

    print(
        x.grad
    )

    print()

    print(
        "Expected gradient:"
    )

    print(
        expected_gradient
    )

    print()

    print(
        f"Forward identity : "
        f"{'PASSED' if forward_ok else 'FAILED'}"
    )

    print(
        f"Gradient reverse : "
        f"{'PASSED' if gradient_ok else 'FAILED'}"
    )

    if not forward_ok:

        raise RuntimeError(
            "GRL forward self-test failed."
        )

    if not gradient_ok:

        raise RuntimeError(
            "GRL backward self-test failed."
        )

    print()

    print(
        "GRL SELF TEST: PASSED"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    _self_test()