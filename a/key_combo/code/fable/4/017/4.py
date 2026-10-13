from __future__ import annotations

import random
import sys
from typing import Sequence

import numpy as np

try:
    from arc_utils import Color
except Exception:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import Color


def _border_cycle(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    if h < 2 or w < 2:
        raise ValueError("grid must have a rectangular border")

    out: list[tuple[int, int]] = []
    for c in range(w):
        out.append((0, c))
    for r in range(1, h):
        out.append((r, w - 1))
    for c in range(w - 2, -1, -1):
        out.append((h - 1, c))
    for r in range(h - 2, 0, -1):
        out.append((r, 0))

    expected = 2 * h + 2 * w - 4
    if len(out) != expected:
        raise RuntimeError("constructed border length mismatch")
    return out


def _extract_border(grid: np.ndarray) -> list[int]:
    border = _border_cycle(grid.shape)
    return [int(grid[r, c]) for r, c in border]


def _build_border(
    shape: tuple[int, int],
    sequence: Sequence[int],
    *,
    fill_non_border: int,
) -> np.ndarray:
    h, w = map(int, shape)
    border = _border_cycle((h, w))
    if len(sequence) != 2 * h + 2 * w - 4:
        raise ValueError("sequence length must match border length")

    out = np.full((h, w), int(fill_non_border), dtype=int)
    for (r, c), color in zip(border, sequence):
        out[r, c] = int(color)
    return out


def _reconstruct_border(
    length: int,
    k: int,
    shift: int,
    motif: Sequence[int],
) -> list[int]:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(motif) != k:
        raise ValueError("motif length mismatch")

    period = 2 * k
    out = [0] * length
    for idx in range(length):
        p = (idx + shift) % period
        x = p % k
        if p < k:
            out[idx] = int(motif[x])
        else:
            out[idx] = int(motif[k - 1 - x])
    return out


def _candidate_solutions(border_seq: Sequence[int]) -> list[tuple[int, int, tuple[int, ...], int]]:
    known = [(idx, int(v)) for idx, v in enumerate(border_seq) if int(v) != int(Color.GRAY)]
    if not known:
        raise ValueError("no visible border pixels; motif phase cannot be inferred")

    L = len(border_seq)
    solutions: list[tuple[int, int, tuple[int, ...], int]] = []
    for k in range(2, L // 2 + 1):
        period = 2 * k
        if L % period != 0:
            continue
        for shift in range(period):
            motif: list[int | None] = [None] * k
            valid = True
            for idx, color in known:
                p = (idx + shift) % period
                x = p % k
                if p < k:
                    pos = x
                else:
                    pos = k - 1 - x

                if motif[pos] is None:
                    motif[pos] = color
                elif motif[pos] != color:
                    valid = False
                    break

            if valid:
                assigned = sum(v is not None for v in motif)
                solutions.append((k, shift, tuple(int(v) for v in motif), assigned))

    if not solutions:
        raise ValueError("no valid period/phase hypothesis")
    return solutions


def _infer_border(border_seq: Sequence[int]) -> tuple[list[int], int, int, tuple[int, ...]]:
    solutions = _candidate_solutions(border_seq)

    full = [(k, shift, motif, assigned) for k, shift, motif, assigned in solutions if assigned == k]
    if not full:
        raise ValueError("insufficient visible pixels to recover full motif")

    L = len(border_seq)
    reconstructed: dict[tuple[int, ...], list[tuple[int, int, tuple[int, ...]]]] = {}
    for k, shift, motif, _ in full:
        seq = tuple(_reconstruct_border(L, k, shift, motif))
        reconstructed.setdefault(seq, []).append((k, shift, motif))

    if len(reconstructed) == 1:
        best_seq, _ = list(reconstructed.items())[0]
    else:
        # Deterministic tie-break: choose lexicographically smallest restored border,
        # then smallest k and shift.
        best_seq = sorted(reconstructed.keys())[0]

    best_k, best_shift, best_motif = min(reconstructed[best_seq], key=lambda t: (t[0], t[1]))
    return list(best_seq), best_k, best_shift, best_motif


def _unique_full_reconstructions(border_seq: Sequence[int]) -> set[tuple[int, ...]]:
    solutions = _candidate_solutions(border_seq)
    complete = [(k, shift, motif) for k, shift, motif, assigned in solutions if assigned == k]
    L = len(border_seq)
    return {
        tuple(_reconstruct_border(L, k, shift, motif))
        for k, shift, motif in complete
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    从顺时针展开的边界色带上恢复“图样段/倒序段”交替的环形墙纸，并修补被灰色断口覆盖的位置。

    Input structure:
    - 画布内部全部为黑色背景。
    - 只有外框边界会出现颜色图案。
    - 按边界顺时针遍历后形成一维循环序列，图样由一段 motif 与其倒序段交替组成，周期为 2k。
    - 有一段连续格子被灰色覆盖，可能跨过边界拐角，且剩余可见部分可唯一确定 motif 与相位。

    Rule:
    - 按顺时针读出边界色带，列出灰色位置形成缺口。
    - 在所有满足 L%(2k)=0 的 period=2k 候选中，尝试不同相位 shift，建立 motif 与反向约束方程。
    - 选择能完全恢复 motif 的唯一可行解，并按该 period/phase 重建全部边界。
    - 用恢复后的颜色替换缺口处，保持内部像素不变。

    Key insight:
    - 修复必须在环形边界的一维坐标上进行，核心规律是 motif 与 reverse(motif) 的交替，而不是二维邻域扫描。
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")

    border_seq = _extract_border(g)
    repaired_border, k, shift, _motif = _infer_border(border_seq)

    out = g.copy()
    border_coords = _border_cycle(g.shape)
    for (r, c), color in zip(border_coords, repaired_border):
        out[r, c] = int(color)
    return out


# ------------------------------
# Generator helpers.


def _non_palindromic_motif(rng: random.Random, k: int, palette: tuple[int, ...]) -> tuple[int, ...]:
    if k < 2:
        raise ValueError("motif length must be at least 2")

    for _ in range(200):
        motif = [rng.choice(palette) for _ in range(k)]
        if motif != motif[::-1]:
            return tuple(motif)
    raise RuntimeError("failed to generate non-palindromic motif")


def _feasible_motif_lengths(perimeter: int) -> list[int]:
    out = []
    for k in range(2, perimeter // 2 + 1):
        period = 2 * k
        if perimeter % period == 0 and perimeter // period >= 2:
            out.append(k)
    return out


def _rotate(seq: Sequence[int], shift: int) -> list[int]:
    n = len(seq)
    shift %= n
    return list(seq[shift:]) + list(seq[:shift])


def _make_damaged_border(
    base_border: Sequence[int],
    *,
    start: int,
    length: int,
) -> list[int]:
    L = len(base_border)
    damaged = [int(v) for v in base_border]
    for offset in range(length):
        damaged[(start + offset) % L] = int(Color.GRAY)
    return damaged


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(4, 24)
    width = rng.randint(4, 24)
    perimeter = 2 * height + 2 * width - 4
    if perimeter < 8:
        return None

    feasible = _feasible_motif_lengths(perimeter)
    if not feasible:
        return None

    k = rng.choice(feasible)
    period = 2 * k
    repeat_count = perimeter // period

    palette = [c for c in Color.NOT_BLACK if c != Color.GRAY]
    if len(palette) < 2:
        return None

    motif = _non_palindromic_motif(rng, k, tuple(int(c) for c in palette))
    period_block = motif + tuple(reversed(motif))
    full_periodic = list(period_block * repeat_count)

    shift = rng.randint(0, period - 1)
    base_border = _rotate(full_periodic, shift)

    max_damage = max(1, period // k - 1)  # shorter than one motif index total appearance count
    damage_len = rng.randint(1, min(max_damage, perimeter - 1))
    damage_start = rng.randint(0, perimeter - 1)
    damaged = _make_damaged_border(base_border, start=damage_start, length=damage_len)

    unique = _unique_full_reconstructions(damaged)
    if len(unique) != 1:
        return None
    _ = next(iter(unique))

    repaired = _infer_border(damaged)[0]
    if any(v == Color.GRAY for v in repaired):
        return None

    # Cross-check the reconstruction exactly matches the designed wallpaper.
    if tuple(repaired) != tuple(base_border):
        return None

    return _build_border((height, width), damaged, fill_non_border=Color.BLACK)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(5000):
        candidate = _gen_attempt(rng)
        if candidate is not None:
            return candidate

    raise RuntimeError("failed to generate valid 4.py input after many attempts")
