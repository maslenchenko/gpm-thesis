from .striped_mamba import StripedMamba, StripedMambaInputInterface, StripedMambaIsolate

models = {
    "stripedmamba": {"new_model": StripedMamba, "seq_len": 393216, "targets_length": 8192},
    "stripedmamba_input_interface": {
        "new_model": StripedMambaInputInterface,
        "seq_len": 393216,
        "targets_length": 8192,
    },
    "stripedmamba_isolate": {"new_model": StripedMambaIsolate, "seq_len": 393216, "targets_length": 1},
}
