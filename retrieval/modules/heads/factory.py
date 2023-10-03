from .registry import head_entrypoint, is_head


def create_head(head_name, inp_dim, out_dim, **kwargs):
    """Create a head 
    """
    #
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    #
    if not is_head(head_name):
        raise RuntimeError('Unknown head (%s)' % head_name)

    #
    create_fn = head_entrypoint(head_name)

    #
    head = create_fn(inp_dim, out_dim, **kwargs)

    return head
