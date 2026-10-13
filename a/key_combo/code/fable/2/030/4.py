from __future__ import annotations

import random

import numpy as np


PALETTE = tuple(range(10))
QUADRANT_ORDER = ("NW", "NE", "SE", "SW")
ALIGN_K = (0, 1, 2, 3)
COUNTER_ALIGN_K = (0, 3, 2, 1)
SIZE_CHOICES = (6, 8, 10, 12, 14)
GEN_ATTEMPTS = 400
PATTERN_ATTEMPTS = 120


def _component_sizes(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            size = 0
            while stack:
                rr, cc = stack.pop()
                size += 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            sizes.append(size)
    sizes.sort(reverse=True)
    return sizes


def _transform_distances(bits: np.ndarray) -> tuple[int, int]:
    d_same: list[int] = []
    d_comp: list[int] = []
    for k in range(1, 4):
        rotated = np.rot90(bits, k)
        d_same.append(int(np.count_nonzero(bits != rotated)))
        d_comp.append(int(np.count_nonzero(bits != np.logical_not(rotated))))
    for k in range(4):
        flipped = np.fliplr(np.rot90(bits, k))
        d_same.append(int(np.count_nonzero(bits != flipped)))
        d_comp.append(int(np.count_nonzero(bits != np.logical_not(flipped))))
    return min(d_same), min(d_comp)


def _transition_count(bits: np.ndarray) -> int:
    return int(np.count_nonzero(bits[:, 1:] != bits[:, :-1]) + np.count_nonzero(bits[1:, :] != bits[:-1, :]))


def _valid_base_pattern(bits: np.ndarray) -> bool:
    n = int(bits.shape[0])
    if bits.shape != (n, n) or n % 2:
        return False
    if int(bits.sum()) != bits.size // 2:
        return False

    mixed_rows = sum(0 < int(row.sum()) < n for row in bits)
    mixed_cols = sum(0 < int(col.sum()) < n for col in bits.T)
    if mixed_rows < max(3, n // 2) or mixed_cols < max(3, n // 2):
        return False

    transitions = _transition_count(bits)
    if transitions < bits.size // 3 or transitions > int(1.7 * bits.size):
        return False

    same_dist, comp_dist = _transform_distances(bits)
    need = max(8, bits.size // 4)
    if same_dist < need or comp_dist < need:
        return False

    sizes_one = _component_sizes(bits)
    sizes_zero = _component_sizes(~bits)
    if not sizes_one or not sizes_zero:
        return False
    if sizes_one[0] < max(n, bits.size // 6):
        return False
    if sizes_zero[0] < max(n, bits.size // 6):
        return False
    if len(sizes_one) > max(8, n) or len(sizes_zero) > max(8, n):
        return False

    return True


def _sample_base_pattern(n: int) -> np.ndarray:
    for _ in range(PATTERN_ATTEMPTS):
        score = np.zeros((n, n), dtype=float)

        block = random.randint(2, max(2, min(4, n // 2)))
        coarse_side = (n + block - 1) // block
        coarse = np.random.normal(size=(coarse_side, coarse_side))
        score += np.kron(coarse, np.ones((block, block), dtype=float))[:n, :n]

        block2 = max(1, block - 1)
        coarse_side2 = (n + block2 - 1) // block2
        coarse2 = np.random.normal(scale=0.55, size=(coarse_side2, coarse_side2))
        score += 0.4 * np.kron(coarse2, np.ones((block2, block2), dtype=float))[:n, :n]

        for _ in range(random.randint(4, 7)):
            r0 = random.randrange(n - 1)
            r1 = random.randrange(r0 + 1, n + 1)
            c0 = random.randrange(n - 1)
            c1 = random.randrange(c0 + 1, n + 1)
            score[r0:r1, c0:c1] += random.uniform(-1.5, 1.5)

        score += np.linspace(random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8), n)[:, None]
        score += np.linspace(random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8), n)[None, :]
        score += np.random.normal(scale=0.06, size=(n, n))

        flat = score.reshape(-1)
        order = np.argsort(flat, kind="mergesort")
        bits = np.zeros(flat.size, dtype=bool)
        bits[order[-(flat.size // 2) :]] = True
        bits = bits.reshape(n, n)
        if _valid_base_pattern(bits):
            return bits

    raise RuntimeError("failed to sample a non-degenerate base pattern")


def _aligned_to_original_coord(q_idx: int, r: int, c: int, n: int) -> tuple[int, int]:
    if q_idx == 0:
        return r, c
    if q_idx == 1:
        return c, n - 1 - r
    if q_idx == 2:
        return n - 1 - r, n - 1 - c
    if q_idx == 3:
        return n - 1 - c, r
    raise ValueError("invalid quadrant index")


def _quadrant_offsets(n: int) -> tuple[tuple[int, int], ...]:
    return ((0, 0), (0, n + 1), (n + 1, n + 1), (n + 1, 0))


def _split_quadrants(grid: np.ndarray) -> tuple[int, int, int, list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h == w and h % 2 == 1, "expected an odd square grid"

    sep = h // 2
    sep_color = int(g[sep, 0])
    assert np.all(g[sep, :] == sep_color), "expected a solid separator row"
    assert np.all(g[:, sep] == sep_color), "expected a solid separator column"

    quads = [
        g[:sep, :sep].copy(),
        g[:sep, sep + 1 :].copy(),
        g[sep + 1 :, sep + 1 :].copy(),
        g[sep + 1 :, :sep].copy(),
    ]
    n = sep
    assert all(q.shape == (n, n) for q in quads), "expected four equal square quadrants"
    return sep_color, sep, n, quads


def _align_quadrants(quads: list[np.ndarray], align_k: tuple[int, int, int, int]) -> list[np.ndarray]:
    return [np.rot90(quads[i], align_k[i]) for i in range(4)]


def _local_palettes(quads: list[np.ndarray]) -> list[tuple[int, int]]:
    palettes: list[tuple[int, int]] = []
    for quad in quads:
        colors = tuple(int(v) for v in np.unique(quad))
        assert len(colors) == 2, "each quadrant must use exactly two colors"
        palettes.append(colors)
    return palettes


def _candidate_repairs(grid: np.ndarray, align_k: tuple[int, int, int, int] = ALIGN_K) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    _sep_color, sep, n, quads = _split_quadrants(g)
    aligned = _align_quadrants(quads, align_k)
    palettes = _local_palettes(aligned)

    outputs: list[np.ndarray] = []
    seen: set[tuple[bytes, tuple[int, int]]] = set()
    for mask in range(16):
        bit_layers = []
        one_choices = []
        for i in range(4):
            choice = (mask >> i) & 1
            one_choices.append(choice)
            bit_layers.append(aligned[i] == palettes[i][choice])

        votes = np.stack(bit_layers, axis=0).sum(axis=0)
        mismatch = (votes != 0) & (votes != 4)
        if int(mismatch.sum()) != 1:
            continue

        r, c = map(int, np.argwhere(mismatch)[0])
        majority = 1 if int(votes[r, c]) == 3 else 0
        bad_quads = [i for i in range(4) if int(bit_layers[i][r, c]) != majority]
        if len(bad_quads) != 1:
            continue

        q_idx = bad_quads[0]
        choice = one_choices[q_idx]
        repair_color = int(palettes[q_idx][choice if majority else 1 - choice])
        qr, qc = _aligned_to_original_coord(q_idx, r, c, n)
        off_r, off_c = _quadrant_offsets(n)[q_idx]
        gr = off_r + qr
        gc = off_c + qc
        if int(g[gr, gc]) == repair_color:
            continue

        out = g.copy()
        out[gr, gc] = repair_color
        key = (out.tobytes(), tuple(map(int, out.shape)))
        if key not in seen:
            seen.add(key)
            outputs.append(out)

    return outputs


def _build_grid(separator_color: int, quads: list[np.ndarray]) -> np.ndarray:
    n = int(quads[0].shape[0])
    grid = np.full((2 * n + 1, 2 * n + 1), int(separator_color), dtype=int)
    for idx, quad in enumerate(quads):
        r0, c0 = _quadrant_offsets(n)[idx]
        grid[r0 : r0 + n, c0 : c0 + n] = quad
    return grid


def _render_quadrant(bits: np.ndarray, bit0_color: int, bit1_color: int) -> np.ndarray:
    return np.where(bits, int(bit1_color), int(bit0_color)).astype(int)


def _make_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = random.choice(SIZE_CHOICES)
    base = _sample_base_pattern(n)

    colors = list(PALETTE)
    random.shuffle(colors)
    separator_color = colors[0]
    body_colors = colors[1:9]
    pairs = [tuple(body_colors[2 * i : 2 * i + 2]) for i in range(4)]

    rotated_bits = [
        base,
        np.rot90(base, -1),
        np.rot90(base, 2),
        np.rot90(base, 1),
    ]

    clean_quads: list[np.ndarray] = []
    mappings: list[tuple[int, int]] = []
    for idx, pair in enumerate(pairs):
        a, b = pair
        if random.random() < 0.5:
            bit0_color, bit1_color = a, b
        else:
            bit0_color, bit1_color = b, a
        mappings.append((bit0_color, bit1_color))
        clean_quads.append(_render_quadrant(rotated_bits[idx], bit0_color, bit1_color))

    bad_q = random.randrange(4)
    ar = random.randrange(n)
    ac = random.randrange(n)
    true_bit = int(base[ar, ac])
    qr, qc = _aligned_to_original_coord(bad_q, ar, ac, n)

    noisy_quads = [quad.copy() for quad in clean_quads]
    bit0_color, bit1_color = mappings[bad_q]
    noisy_quads[bad_q][qr, qc] = int(bit0_color if true_bit else bit1_color)

    inp = _build_grid(separator_color, noisy_quads)
    out = _build_grid(separator_color, clean_quads)
    return inp, out, base


def _validate_case(inp: np.ndarray, out: np.ndarray, base: np.ndarray) -> bool:
    if np.array_equal(inp, out):
        return False
    if int(np.count_nonzero(inp != out)) != 1:
        return False

    if not _valid_base_pattern(base):
        return False

    sep_color, sep, n, quads = _split_quadrants(inp)
    if int(out[sep, 0]) != sep_color:
        return False

    palettes = []
    all_colors = {sep_color}
    bad_count_quads = 0
    for quad in quads:
        colors = tuple(int(v) for v in np.unique(quad))
        if len(colors) != 2:
            return False
        if set(colors) & all_colors:
            return False
        palettes.append(colors)
        all_colors.update(colors)
        counts = sorted(int(np.count_nonzero(quad == color)) for color in colors)
        if counts[0] != counts[1]:
            if counts[1] - counts[0] != 2:
                return False
            bad_count_quads += 1
    if bad_count_quads != 1:
        return False

    candidates = _candidate_repairs(inp, ALIGN_K)
    if len(candidates) != 1 or not np.array_equal(candidates[0], out):
        return False

    if _candidate_repairs(inp, (0, 0, 0, 0)):
        return False
    if _candidate_repairs(inp, COUNTER_ALIGN_K):
        return False

    same_dist, comp_dist = _transform_distances(base)
    need = max(8, base.size // 4)
    return same_dist >= need and comp_dist >= need


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the single bad pixel in a four-quadrant windmill by rotation alignment and cross-copy voting.

    Input structure:
    - One color forms a solid center row and center column, splitting the odd square grid into four equal square quadrants.
    - Each quadrant is filled with exactly two private colors, and the quadrants are the same binary pattern in windmill order:
      NW is the base view, NE is rotated 90 degrees clockwise, SE is rotated 180 degrees, and SW is rotated 270 degrees clockwise.
    - The underlying binary pattern has no rotational or mirror self-symmetry, and exactly one pixel in exactly one quadrant is flipped to the other local color.

    Rule:
    - Rotate the four quadrants back to the NW orientation.
    - For each quadrant, decide which of its two colors is structure bit 0 or bit 1 by finding the binary assignment where all aligned cells agree except one 3-vs-1 vote.
    - The unique disagreeing aligned cell identifies the defect; repaint that cell in the bad quadrant with the local color that matches the majority structure bit.
    - Keep the separator row, separator column, and every other pixel unchanged.

    Key insight:
    - Raw colors cannot be compared directly because every quadrant uses a disjoint palette; only after undoing the windmill rotations and stripping each quadrant down to a binary mask does the lone error appear as a single voting loss.
    """
    g = np.asarray(grid, dtype=int)
    candidates = _candidate_repairs(g, ALIGN_K)
    assert len(candidates) == 1, "expected a unique semantic repair"
    return candidates[0]


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp, out, base = _make_case()
        if _validate_case(inp, out, base) and np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate a valid windmill vote-repair puzzle")
