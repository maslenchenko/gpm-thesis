import torch


def stochastic_revcomp_batch(seq_batch, out_batch, strand_pair, max_shift=0, generator=None):
    # seq_batch: (B, L, C), out_batch: (B, T, F)
    batch_size = seq_batch.shape[0]
    device = seq_batch.device
    if generator is None:
        generator = torch.Generator(device=device)

    revcomp_flags = torch.rand(batch_size, generator=generator, device=device) < 0.5
    if max_shift > 0:
        shifts = torch.randint(0, max_shift + 1, (batch_size,), generator=generator, device=device)
    else:
        shifts = torch.zeros(batch_size, dtype=torch.long, device=device)

    seq_out = seq_batch.clone()
    out_out = out_batch.clone()

    for i in range(batch_size):
        shift = int(shifts[i].item())
        if shift:
            seq_out[i] = torch.roll(seq_out[i], shifts=shift, dims=0)
        if revcomp_flags[i]:
            seq_out[i] = torch.flip(seq_out[i], dims=[0, 1])
            out_out[i] = torch.flip(out_out[i], dims=[0])[:, strand_pair]

    return seq_out, out_out, revcomp_flags, shifts


def shift_dna(seq, shift):
    # seq: (B, L, C) or (L, C)
    if shift == 0:
        return seq
    if seq.dim() == 2:
        seq = seq.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    if shift < 0:
        pad = torch.ones_like(seq[:, : -shift, :]) / 4.0
        out = torch.cat([seq[:, :shift, :], pad], dim=1)
    else:
        pad = torch.ones_like(seq[:, :shift, :]) / 4.0
        out = torch.cat([pad, seq[:, shift:, :]], dim=1)

    if squeeze_back:
        out = out.squeeze(0)
    return out


def ensemble_fwd_rev(predict_fn, strand_pair):
    def predict_wrapper(model, seq, *args, **kwargs):
        y = predict_fn(model, seq, *args, **kwargs)
        rev = torch.flip(seq, dims=[1, 2])
        y_rev = predict_fn(model, rev, *args, **kwargs)
        y_rev = torch.flip(y_rev, dims=[1])[:, :, strand_pair]
        return (y + y_rev) / 2

    return predict_wrapper


def ensemble_shift(predict_fn, max_shift):
    def predict_wrapper(model, seq, *args, **kwargs):
        y = 0
        for shift in range(max_shift + 1):
            y = y + predict_fn(model, shift_dna(seq, shift), *args, **kwargs)
        return y / (max_shift + 1)

    return predict_wrapper
