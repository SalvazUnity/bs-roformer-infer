# brain — retired multi-backend BS-RoFormer campaign

> **Superseded 2026-09-14.** Apple MLX backends and Torch MPS devices were
> removed before release. This directory preserves the campaign's evidence and
> rejected design history; it is not the current package contract. See D14 in
> [`decisions.md`](decisions.md), plus the root `README.md` and `CLAUDE.md`, for
> the landed Torch-only contract.

Working memory for one campaign: give `bs-roformer-infer` an **MLX backend** and
an **MPS device**, so a caller switches hardware paths by changing one argument
and loses nothing.

This directory is maintainer-facing scratch canon. It is not the package
contract — `README.md` (users) and `CLAUDE.md` (maintainers) stay authoritative.
When a decision here ships, restate it there and note it here as landed.

## Navigate

| File | What it answers |
|---|---|
| [`evidence.md`](evidence.md) | What was actually measured on real hardware, and what remains unmeasured |
| [`decisions.md`](decisions.md) | What is settled, why, and what is still open |
| [`architecture.md`](architecture.md) | Superseded `backend` × `device` design, retained as rejected-path history |
| [`plan.md`](plan.md) | Phased implementation with its acceptance gates |

## The one-line problem

`device` and `backend` are two different axes and the package currently only has
a partial version of the first:

- **`device`** — which chip Torch computes on. Today: `cpu`, `cuda`, `cuda:N`.
  Missing: `mps`, even though every load-bearing operator was measured to run on
  it natively.
- **`backend`** — which framework computes at all. Today: only Torch. MLX runs
  the same checkpoint ~2.2x faster than Torch on MPS, at float32 noise parity.

Both are additive. Neither may remove or degrade an existing path.
