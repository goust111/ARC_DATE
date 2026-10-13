from __future__ import annotations

from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4

MAX_GENERATION_ATTEMPTS = 800
SHAPES = (11, 13, 15)


def _encode_layers(red_mask: np.ndarray, blue_mask: np.ndarray) -> np.ndarray:
    out = np.zeros(red_mask.shape, dtype=int)
    out[blue_mask] = BLUE
    out[red_mask] = RED
    out[red_mask & blue_mask] = YELLOW
    return out


def _decode_layers(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert np.all(np.isin(g, [BLACK, BLUE, RED, YELLOW])), "unexpected colors present"
    red_layer = (g == RED) | (g == YELLOW)
    blue_layer = (g == BLUE) | (g == YELLOW)
    return red_layer, blue_layer


def _vertical_clean(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=bool) & np.fliplr(np.asarray(mask, dtype=bool))


def _horizontal_clean(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=bool) & np.flipud(np.asarray(mask, dtype=bool))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode yellow as dual membership, denoise the red and blue layers with orthogonal center-axis symmetry, then recolor by the cleaned overlap.

    Input structure:
    - A black background grid contains only blue, red, and yellow foreground pixels.
    - Red and yellow together form the red layer, which is nearly symmetric about the grid's vertical center axis.
    - Blue and yellow together form the blue layer, which is nearly symmetric about the grid's horizontal center axis.
    - Yellow does not denote an independent object type; it marks cells that belong to both layers at once.

    Rule:
    - Build the red layer `R` from all red or yellow cells, and the blue layer `B` from all blue or yellow cells.
    - Clean `R` by keeping only cells whose left-right mirror across the grid's vertical center axis is also in `R`.
    - Clean `B` by keeping only cells whose up-down mirror across the grid's horizontal center axis is also in `B`.
    - Re-encode the cleaned layers: cells in both cleaned layers become yellow, red-only cells stay red, blue-only cells stay blue, and all other cells become black.

    Key insight:
    - The two colors must be interpreted as overlapping symmetry-constrained layers: yellow is a shared-layer code, so denoising happens independently per layer but the final color depends on their cleaned overlap.
    """
    red_layer, blue_layer = _decode_layers(grid)
    clean_red = _vertical_clean(red_layer)
    clean_blue = _horizontal_clean(blue_layer)
    return _encode_layers(clean_red, clean_blue)


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _weighted_choice(rng: np.random.Generator, items: list[tuple[int, int]], weights: list[float]) -> tuple[int, int]:
    probs = np.asarray(weights, dtype=float)
    probs /= probs.sum()
    idx = int(rng.choice(len(items), p=probs))
    return items[idx]


def _grow_connected_mask(
    rng: np.random.Generator,
    shape: tuple[int, int],
    target: int,
    *,
    axis_side: str,
) -> np.ndarray | None:
    h, w = map(int, shape)
    for _ in range(120):
        mask = np.zeros((h, w), dtype=bool)
        if axis_side == "right":
            start_r = int(rng.integers(max(0, h // 2 - 2), min(h, h // 2 + 3)))
            start_c = int(rng.integers(max(0, w - 3), w))
        else:
            start_r = int(rng.integers(max(0, h - 3), h))
            start_c = int(rng.integers(max(0, w // 2 - 2), min(w, w // 2 + 3)))
        mask[start_r, start_c] = True

        while int(mask.sum()) < target:
            frontier: list[tuple[int, int]] = []
            weights: list[float] = []
            for r, c in np.argwhere(mask):
                rr = int(r)
                cc = int(c)
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w) or mask[nr, nc]:
                        continue
                    cell = (nr, nc)
                    if cell in frontier:
                        continue
                    frontier.append(cell)
                    axis_bonus = (nc + 1) if axis_side == "right" else (nr + 1)
                    center_bonus = 1.0 / (1.0 + abs(nr - h / 2.0) + abs(nc - w / 2.0))
                    weights.append(float(axis_bonus) + center_bonus)
            if not frontier:
                break
            nr, nc = _weighted_choice(rng, frontier, weights)
            mask[nr, nc] = True

        if int(mask.sum()) != target:
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) < 2 or len(cols) < 2:
            continue
        if axis_side == "right" and not np.any(mask[:, :-1]):
            continue
        if axis_side == "bottom" and not np.any(mask[:-1, :]):
            continue
        return mask
    return None


def _mirror_vertical_from_half(half_mask: np.ndarray, full_width: int) -> np.ndarray:
    h, half_w = map(int, half_mask.shape)
    out = np.zeros((h, full_width), dtype=bool)
    for r, c in np.argwhere(half_mask):
        rr = int(r)
        cc = int(c)
        out[rr, cc] = True
        out[rr, full_width - 1 - cc] = True
    return out


def _mirror_horizontal_from_half(half_mask: np.ndarray, full_height: int) -> np.ndarray:
    half_h, w = map(int, half_mask.shape)
    out = np.zeros((full_height, w), dtype=bool)
    for r, c in np.argwhere(half_mask):
        rr = int(r)
        cc = int(c)
        out[rr, cc] = True
        out[full_height - 1 - rr, cc] = True
    return out


def _vertical_axis_gap(mask: np.ndarray) -> tuple[int, int]:
    h, w = mask.shape
    pts = np.argwhere(mask)
    true_k = w - 1
    scores: list[int] = []
    for k in range(2 * w - 1):
        score = 0
        for r, c in pts:
            cc = int(k) - int(c)
            if 0 <= cc < w and mask[int(r), cc]:
                score += 1
        scores.append(score)
    others = [score for k, score in enumerate(scores) if k != true_k]
    return int(scores[true_k]), int(scores[true_k] - max(others))


def _horizontal_axis_gap(mask: np.ndarray) -> tuple[int, int]:
    h, w = mask.shape
    pts = np.argwhere(mask)
    true_k = h - 1
    scores: list[int] = []
    for k in range(2 * h - 1):
        score = 0
        for r, c in pts:
            rr = int(k) - int(r)
            if 0 <= rr < h and mask[rr, int(c)]:
                score += 1
        scores.append(score)
    others = [score for k, score in enumerate(scores) if k != true_k]
    return int(scores[true_k]), int(scores[true_k] - max(others))


def _make_clean_red(rng: np.random.Generator, h: int, w: int) -> np.ndarray | None:
    half_w = w // 2 + 1
    target = int(rng.integers(4, 8))
    half = _grow_connected_mask(rng, (h, half_w), target, axis_side="right")
    if half is None:
        return None
    mask = _mirror_vertical_from_half(half, w)
    area = int(mask.sum())
    score, gap = _vertical_axis_gap(mask)
    if not (8 <= area <= 22 and score == area and gap >= 3):
        return None
    return mask


def _make_clean_blue(rng: np.random.Generator, h: int, w: int) -> np.ndarray | None:
    half_h = h // 2 + 1
    target = int(rng.integers(4, 8))
    half = _grow_connected_mask(rng, (half_h, w), target, axis_side="bottom")
    if half is None:
        return None
    mask = _mirror_horizontal_from_half(half, h)
    area = int(mask.sum())
    score, gap = _horizontal_axis_gap(mask)
    if not (8 <= area <= 22 and score == area and gap >= 3):
        return None
    return mask


def _adjacent_candidates(mask: np.ndarray) -> set[tuple[int, int]]:
    h, w = mask.shape
    out: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                out.add((nr, nc))
    return out


def _red_noise_candidates(
    red_clean: np.ndarray,
    blue_clean: np.ndarray,
) -> dict[str, list[tuple[int, int]]]:
    h, w = red_clean.shape
    center_c = w // 2
    cats = {"other": [], "empty": []}
    for r, c in sorted(_adjacent_candidates(red_clean)):
        if c == center_c:
            continue
        if red_clean[r, c]:
            continue
        mirror_c = w - 1 - c
        if red_clean[r, mirror_c]:
            continue
        if blue_clean[r, c] and not red_clean[r, c]:
            cats["other"].append((r, c))
        elif not blue_clean[r, c]:
            cats["empty"].append((r, c))
    return cats


def _blue_noise_candidates(
    blue_clean: np.ndarray,
    red_clean: np.ndarray,
    forbidden: set[tuple[int, int]],
) -> dict[str, list[tuple[int, int]]]:
    h, w = blue_clean.shape
    center_r = h // 2
    cats = {"other": [], "empty": []}
    for r, c in sorted(_adjacent_candidates(blue_clean)):
        if r == center_r or (r, c) in forbidden:
            continue
        if blue_clean[r, c]:
            continue
        mirror_r = h - 1 - r
        if blue_clean[mirror_r, c]:
            continue
        if red_clean[r, c] and not blue_clean[r, c]:
            cats["other"].append((r, c))
        elif not red_clean[r, c]:
            cats["empty"].append((r, c))
    return cats


def _pick_noise(
    rng: np.random.Generator,
    candidates: dict[str, list[tuple[int, int]]],
    total_count: int,
    *,
    layer: str,
    shape: tuple[int, int],
) -> set[tuple[int, int]] | None:
    h, w = map(int, shape)
    picked: list[tuple[int, int]] = []

    def mirror(cell: tuple[int, int]) -> tuple[int, int]:
        r, c = cell
        if layer == "red":
            return (r, w - 1 - c)
        return (h - 1 - r, c)

    for key in ("other", "empty"):
        pool = candidates[key][:]
        rng.shuffle(pool)
        choice = None
        for cell in pool:
            if cell in picked or mirror(cell) in picked:
                continue
            choice = cell
            break
        if choice is None:
            return None
        picked.append(choice)

    extras = [cell for key in ("other", "empty") for cell in candidates[key]]
    rng.shuffle(extras)
    for cell in extras:
        if len(picked) >= total_count:
            break
        if cell in picked or mirror(cell) in picked:
            continue
        picked.append(cell)

    if len(picked) != total_count:
        return None
    return set(picked)


def _validate_instance(
    inp: np.ndarray,
    red_clean: np.ndarray,
    blue_clean: np.ndarray,
    red_noise: set[tuple[int, int]],
    blue_noise: set[tuple[int, int]],
) -> bool:
    out = main(inp)
    expected = _encode_layers(red_clean, blue_clean)
    if not np.array_equal(out, expected):
        return False

    input_red, input_blue = _decode_layers(inp)
    red_score, red_gap = _vertical_axis_gap(input_red)
    blue_score, blue_gap = _horizontal_axis_gap(input_blue)
    if red_score <= int(red_clean.sum()) - 2 or blue_score <= int(blue_clean.sum()) - 2:
        return False
    if red_gap < 2 or blue_gap < 2:
        return False

    overlap = red_clean & blue_clean
    if int(overlap.sum()) < 2:
        return False
    if int(np.count_nonzero(red_clean & ~blue_clean)) < 4:
        return False
    if int(np.count_nonzero(blue_clean & ~red_clean)) < 4:
        return False

    if len(red_noise) < 2 or len(blue_noise) < 2:
        return False
    if not np.any((inp == YELLOW) & (out == BLUE)):
        return False
    if not np.any((inp == YELLOW) & (out == RED)):
        return False
    if not np.any((inp == RED) & (out == BLACK)):
        return False
    if not np.any((inp == BLUE) & (out == BLACK)):
        return False
    if not np.any(out == YELLOW):
        return False
    if np.array_equal(inp, out):
        return False
    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.choice(SHAPES))
    w = int(rng.choice(SHAPES))

    red_clean = _make_clean_red(rng, h, w)
    blue_clean = _make_clean_blue(rng, h, w)
    if red_clean is None or blue_clean is None:
        return None

    overlap = red_clean & blue_clean
    if not (2 <= int(overlap.sum()) <= 6):
        return None
    if int(np.count_nonzero(red_clean & ~blue_clean)) < 4:
        return None
    if int(np.count_nonzero(blue_clean & ~red_clean)) < 4:
        return None
    if int((red_clean | blue_clean).sum()) > (h * w) // 2:
        return None

    red_candidates = _red_noise_candidates(red_clean, blue_clean)
    red_noise = _pick_noise(
        rng,
        red_candidates,
        total_count=int(rng.integers(2, 4)),
        layer="red",
        shape=(h, w),
    )
    if red_noise is None:
        return None

    blue_candidates = _blue_noise_candidates(blue_clean, red_clean, red_noise)
    blue_noise = _pick_noise(
        rng,
        blue_candidates,
        total_count=int(rng.integers(2, 4)),
        layer="blue",
        shape=(h, w),
    )
    if blue_noise is None:
        return None

    if red_noise & blue_noise:
        return None

    noisy_red = red_clean.copy()
    noisy_blue = blue_clean.copy()
    for r, c in red_noise:
        noisy_red[r, c] = True
    for r, c in blue_noise:
        noisy_blue[r, c] = True

    inp = _encode_layers(noisy_red, noisy_blue)
    if not _validate_instance(inp, red_clean, blue_clean, red_noise, blue_noise):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-layer symmetry puzzle")
