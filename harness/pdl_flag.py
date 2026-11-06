"""Programmatic dependent launch (PDL) switch for the exact-fast kernels. When ON, every kernel launched through the helpers below is launched
with launch_pdl=True and calls griddepcontrol.wait (tl.extra.cuda.gdc_wait) before touching any data, so consecutive kernels in the CUDA graph
overlap their launch/prologue; results are unchanged (the wait returns only after the prerequisite grid completed and flushed memory)."""
ON = False


def kw():
    return dict(PDL=True, launch_pdl=True) if ON else {}
