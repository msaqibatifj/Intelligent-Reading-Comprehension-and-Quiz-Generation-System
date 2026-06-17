"""
test_transformer_model.py — Unit tests for the from-scratch seq2seq Transformer.

All tests verify tensor shapes match expected dimensions
given the current batch size and vocabulary length.

Run:  python -m pytest tests/test_transformer_model.py -v
"""

import sys
import os

try:
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _SRC_DIR = os.path.join(_THIS_DIR, '..', 'src')
except NameError:
    _SRC_DIR = os.path.join(os.getcwd(), 'src')
sys.path.insert(0, os.path.normpath(_SRC_DIR))

import torch
import pytest

from transformer_model import (
    TransformerConfig,
    PositionalEncoding,
    MultiHeadAttention,
    FeedForward,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
    Seq2SeqTransformer,
)


# -- Fixtures ----------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    return TransformerConfig(vocab_size=30522, d_model=64, n_heads=4,
                             n_encoder_layers=2, n_decoder_layers=2,
                             d_ff=256, max_seq_len=128)


@pytest.fixture(scope="module")
def B():
    return 4


@pytest.fixture(scope="module")
def src_len():
    return 48


@pytest.fixture(scope="module")
def tgt_len():
    return 32


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -- Config tests ------------------------------------------------------------

def test_config_defaults():
    c = TransformerConfig()
    assert c.vocab_size == 30522
    assert c.d_model == 512
    assert c.n_heads == 8
    assert c.pad_token_id == 0


# -- PositionalEncoding tests ------------------------------------------------

def test_positional_encoding_shape(cfg, B, src_len, device):
    pe = PositionalEncoding(cfg).to(device)
    x = torch.randn(B, src_len, cfg.d_model, device=device)
    out = pe(x)
    assert out.shape == (B, src_len, cfg.d_model)


def test_positional_encoding_pe_buffer(cfg):
    pe = PositionalEncoding(cfg)
    assert pe.pe.shape == (1, cfg.max_seq_len, cfg.d_model)


# -- MultiHeadAttention tests ------------------------------------------------

def test_mha_shape(cfg, B, src_len, device):
    mha = MultiHeadAttention(cfg).to(device)
    x = torch.randn(B, src_len, cfg.d_model, device=device)
    out = mha(x, x, x)
    assert out.shape == (B, src_len, cfg.d_model)


def test_mha_masked_shape(cfg, B, src_len, device):
    mha = MultiHeadAttention(cfg).to(device)
    x = torch.randn(B, src_len, cfg.d_model, device=device)
    mask = torch.ones(B, 1, 1, src_len, dtype=torch.bool, device=device)
    out = mha(x, x, x, mask=mask)
    assert out.shape == (B, src_len, cfg.d_model)


def test_mha_four_different_lengths(cfg, B, device):
    T_q, T_k = 16, 32
    mha = MultiHeadAttention(cfg).to(device)
    q = torch.randn(B, T_q, cfg.d_model, device=device)
    k = torch.randn(B, T_k, cfg.d_model, device=device)
    v = torch.randn(B, T_k, cfg.d_model, device=device)
    out = mha(q, k, v)
    assert out.shape == (B, T_q, cfg.d_model)


# -- FeedForward tests -------------------------------------------------------

def test_feed_forward_shape(cfg, B, src_len, device):
    ff = FeedForward(cfg).to(device)
    x = torch.randn(B, src_len, cfg.d_model, device=device)
    out = ff(x)
    assert out.shape == (B, src_len, cfg.d_model)


# -- Encoder Layer tests -----------------------------------------------------

def test_encoder_layer_shape(cfg, B, src_len, device):
    layer = TransformerEncoderLayer(cfg).to(device)
    x = torch.randn(B, src_len, cfg.d_model, device=device)
    out = layer(x)
    assert out.shape == (B, src_len, cfg.d_model)


def test_encoder_layer_masked_shape(cfg, B, src_len, device):
    layer = TransformerEncoderLayer(cfg).to(device)
    x = torch.randn(B, src_len, cfg.d_model, device=device)
    mask = torch.ones(B, 1, 1, src_len, dtype=torch.bool, device=device)
    out = layer(x, src_mask=mask)
    assert out.shape == (B, src_len, cfg.d_model)


# -- Decoder Layer tests -----------------------------------------------------

def test_decoder_layer_shape(cfg, B, src_len, tgt_len, device):
    layer = TransformerDecoderLayer(cfg).to(device)
    x = torch.randn(B, tgt_len, cfg.d_model, device=device)
    memory = torch.randn(B, src_len, cfg.d_model, device=device)
    out = layer(x, memory)
    assert out.shape == (B, tgt_len, cfg.d_model)


def test_decoder_layer_masked_shape(cfg, B, src_len, tgt_len, device):
    layer = TransformerDecoderLayer(cfg).to(device)
    x = torch.randn(B, tgt_len, cfg.d_model, device=device)
    memory = torch.randn(B, src_len, cfg.d_model, device=device)
    tgt_mask = torch.ones(B, 1, tgt_len, tgt_len, dtype=torch.bool, device=device)
    mem_mask = torch.ones(B, 1, 1, src_len, dtype=torch.bool, device=device)
    out = layer(x, memory, tgt_mask=tgt_mask, memory_mask=mem_mask)
    assert out.shape == (B, tgt_len, cfg.d_model)


# -- Encoder tests -----------------------------------------------------------

def test_encoder_shape(cfg, B, src_len, device):
    enc = TransformerEncoder(cfg).to(device)
    x = torch.randn(B, src_len, cfg.d_model, device=device)
    out = enc(x)
    assert out.shape == (B, src_len, cfg.d_model)


# -- Decoder tests -----------------------------------------------------------

def test_decoder_shape(cfg, B, src_len, tgt_len, device):
    dec = TransformerDecoder(cfg).to(device)
    x = torch.randn(B, tgt_len, cfg.d_model, device=device)
    memory = torch.randn(B, src_len, cfg.d_model, device=device)
    out = dec(x, memory)
    assert out.shape == (B, tgt_len, cfg.d_model)


# -- Full Seq2SeqTransformer tests ------------------------------------------

def test_seq2seq_forward_shape(cfg, B, src_len, tgt_len, device):
    model = Seq2SeqTransformer(cfg).to(device)
    src = torch.randint(0, cfg.vocab_size, (B, src_len), device=device)
    tgt = torch.randint(0, cfg.vocab_size, (B, tgt_len), device=device)
    logits = model(src, tgt)
    assert logits.shape == (B, tgt_len, cfg.vocab_size)


def test_seq2seq_forward_with_padding_mask(cfg, B, src_len, tgt_len, device):
    model = Seq2SeqTransformer(cfg).to(device)
    src = torch.randint(0, cfg.vocab_size, (B, src_len), device=device)
    tgt = torch.randint(0, cfg.vocab_size, (B, tgt_len), device=device)
    padding_mask = torch.ones(B, src_len, dtype=torch.long, device=device)
    padding_mask[:, src_len // 2:] = 0
    logits = model(src, tgt, src_padding_mask=padding_mask)
    assert logits.shape == (B, tgt_len, cfg.vocab_size)


def test_seq2seq_generate_shape(cfg, B, src_len, device):
    model = Seq2SeqTransformer(cfg).to(device)
    src = torch.randint(0, cfg.vocab_size, (B, src_len), device=device)
    out = model.generate(src, max_len=32)
    assert out.shape == (B, 32)
    assert out.dtype == torch.long


def test_seq2seq_generate_with_mask(cfg, B, src_len, device):
    model = Seq2SeqTransformer(cfg).to(device)
    src = torch.randint(0, cfg.vocab_size, (B, src_len), device=device)
    padding_mask = torch.ones(B, src_len, dtype=torch.long, device=device)
    padding_mask[:, src_len // 3:] = 0
    out = model.generate(src, max_len=24, src_padding_mask=padding_mask)
    assert out.shape == (B, 24)


def test_seq2seq_generate_different_max_lens(cfg, B, src_len, device):
    model = Seq2SeqTransformer(cfg).to(device)
    src = torch.randint(0, cfg.vocab_size, (B, src_len), device=device)
    for max_len in [8, 16, 32]:
        out = model.generate(src, max_len=max_len)
        assert out.shape == (B, max_len), f"Expected (B, {max_len}), got {out.shape}"


def test_seq2seq_logits_are_probabilistic(cfg, B, src_len, tgt_len, device):
    """Verify output logits are finite (no NaN / Inf) and span a range > 0."""
    model = Seq2SeqTransformer(cfg).to(device)
    src = torch.randint(0, cfg.vocab_size, (B, src_len), device=device)
    tgt = torch.randint(0, cfg.vocab_size, (B, tgt_len), device=device)
    logits = model(src, tgt)
    assert torch.isfinite(logits).all()
    assert logits.max() - logits.min() > 0.0


def test_seq2seq_loss_decreases(cfg, B, src_len, tgt_len, device):
    """After a single gradient step, loss should not increase (sanity check)."""
    model = Seq2SeqTransformer(cfg).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    src = torch.randint(0, cfg.vocab_size, (B, src_len), device=device)
    tgt = torch.randint(0, cfg.vocab_size, (B, tgt_len), device=device)

    logits = model(src, tgt[:, :-1])
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.vocab_size),
        tgt[:, 1:].reshape(-1),
        ignore_index=cfg.pad_token_id,
    )
    loss.backward()
    opt.step()

    logits2 = model(src, tgt[:, :-1])
    loss2 = torch.nn.functional.cross_entropy(
        logits2.reshape(-1, cfg.vocab_size),
        tgt[:, 1:].reshape(-1),
        ignore_index=cfg.pad_token_id,
    )
    assert loss2.item() <= loss.item() + 0.5, f"Loss increased: {loss.item():.4f} → {loss2.item():.4f}"


# -- Mask helpers ------------------------------------------------------------

def test_causal_mask_shape(cfg, device):
    mask = Seq2SeqTransformer._causal_mask(16, device)
    assert mask.shape == (1, 1, 16, 16)
    assert mask.dtype == torch.bool or mask.dtype == torch.float


def test_causal_mask_is_lower_triangular(cfg, device):
    mask = Seq2SeqTransformer._causal_mask(8, device)
    lower = torch.tril(torch.ones(1, 1, 8, 8, device=device))
    assert (mask == lower).all(), "Causal mask must be lower triangular"


def test_padding_mask_shape(cfg, B, src_len, device):
    pm = torch.ones(B, src_len, dtype=torch.long, device=device)
    pm[:, 0] = 0
    attn_mask = Seq2SeqTransformer._padding_mask(pm)
    expected_shape = (B, 1, 1, src_len)
    assert attn_mask.shape == expected_shape, f"Expected {expected_shape}, got {attn_mask.shape}"


# -- Module parameters -------------------------------------------------------

def test_all_parameters_have_grad(cfg, B, src_len, tgt_len, device):
    model = Seq2SeqTransformer(cfg).to(device)
    src = torch.randint(0, cfg.vocab_size, (B, src_len), device=device)
    tgt = torch.randint(0, cfg.vocab_size, (B, tgt_len), device=device)
    logits = model(src, tgt[:, :-1])
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.vocab_size),
        tgt[:, 1:].reshape(-1),
        ignore_index=cfg.pad_token_id,
    )
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert param.grad.isfinite().all(), f"{name} has non-finite gradient"


def test_embedding_gradients_propagate(cfg, B, src_len, tgt_len, device):
    model = Seq2SeqTransformer(cfg).to(device)
    src = torch.randint(0, cfg.vocab_size, (B, src_len), device=device)
    tgt = torch.randint(0, cfg.vocab_size, (B, tgt_len), device=device)
    logits = model(src, tgt[:, :-1])
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.vocab_size),
        tgt[:, 1:].reshape(-1),
        ignore_index=cfg.pad_token_id,
    )
    loss.backward()
    assert model.src_embed.weight.grad is not None
    assert model.tgt_embed.weight.grad is not None


# -- Batch-size invariance ---------------------------------------------------

@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_variable_batch_sizes(cfg, src_len, tgt_len, device, batch_size):
    model = Seq2SeqTransformer(cfg).to(device)
    src = torch.randint(0, cfg.vocab_size, (batch_size, src_len), device=device)
    tgt = torch.randint(0, cfg.vocab_size, (batch_size, tgt_len), device=device)
    logits = model(src, tgt)
    assert logits.shape == (batch_size, tgt_len, cfg.vocab_size)


# -- Manual runner -----------------------------------------------------------

if __name__ == "__main__":
    import traceback
    tests = [
        test_config_defaults,
        test_positional_encoding_shape,
        test_positional_encoding_pe_buffer,
        test_mha_shape,
        test_mha_masked_shape,
        test_mha_four_different_lengths,
        test_feed_forward_shape,
        test_encoder_layer_shape,
        test_encoder_layer_masked_shape,
        test_decoder_layer_shape,
        test_decoder_layer_masked_shape,
        test_encoder_shape,
        test_decoder_shape,
        test_seq2seq_forward_shape,
        test_seq2seq_forward_with_padding_mask,
        test_seq2seq_generate_shape,
        test_seq2seq_generate_with_mask,
        test_seq2seq_generate_different_max_lens,
        test_seq2seq_logits_are_probabilistic,
        test_seq2seq_loss_decreases,
        test_causal_mask_shape,
        test_causal_mask_is_lower_triangular,
        test_padding_mask_shape,
        test_all_parameters_have_grad,
        test_embedding_gradients_propagate,
        test_variable_batch_sizes,
    ]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'-' * 50}")
    print(f"  {passed} passed, {failed} failed")
