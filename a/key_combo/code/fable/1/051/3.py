from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, scale_sprite


ORIENTATION_COUNT = 8
ALL_COLORS = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 600


def _orient(grid: np.ndarray, op: int) -> np.ndarray:
    op = int(op)
    if op == 0:
        return grid.copy()
    if op == 1:
        return np.rot90(grid, 1)
    if op == 2:
        return np.rot90(grid, 2)
    if op == 3:
        return np.rot90(grid, 3)
    if op == 4:
        return np.fliplr(grid)
    if op == 5:
        return np.flipud(grid)
    if op == 6:
        return grid.T.copy()
    if op == 7:
        return np.rot90(grid, 2).T.copy()
    raise ValueError(f"invalid orientation op: {op}")


def _map_coord(shape: tuple[int, int], op: int, r: int, c: int) -> tuple[int, int]:
    h, w = map(int, shape)
    r = int(r)
    c = int(c)
    op = int(op)
    if op == 0:
        return r, c
    if op == 1:
        return w - 1 - c, r
    if op == 2:
        return h - 1 - r, w - 1 - c
    if op == 3:
        return c, h - 1 - r
    if op == 4:
        return r, w - 1 - c
    if op == 5:
        return h - 1 - r, c
    if op == 6:
        return c, r
    if op == 7:
        return w - 1 - c, h - 1 - r
    raise ValueError(f"invalid orientation op: {op}")


def _compress_uniform(panel: np.ndarray, out_shape: tuple[int, int]) -> tuple[np.ndarray, int] | None:
    h, w = map(int, panel.shape)
    oh, ow = map(int, out_shape)
    if oh <= 0 or ow <= 0:
        return None
    if h % oh != 0 or w % ow != 0:
        return None
    k_row = h // oh
    k_col = w // ow
    if k_row != k_col or k_row <= 0:
        return None
    k = int(k_row)
    blocks = panel.reshape(oh, k, ow, k)
    reduced = blocks[:, 0, :, 0].copy()
    if not bool(np.all(blocks == reduced[:, None, :, None])):
        return None
    return reduced, k


def _candidate_alignments(panel: np.ndarray, ref_shape: tuple[int, int]) -> list[dict]:
    ref_h, ref_w = map(int, ref_shape)
    pre_shapes = {(ref_h, ref_w), (ref_w, ref_h)}
    seen: dict[tuple[tuple[int, int], bytes], dict] = {}

    for pre_shape in pre_shapes:
        compressed = _compress_uniform(panel, pre_shape)
        if compressed is None:
            continue
        small, scale = compressed
        for align_op in range(ORIENTATION_COUNT):
            aligned = _orient(small, align_op)
            if aligned.shape != ref_shape:
                continue
            key = (tuple(map(int, aligned.shape)), aligned.tobytes())
            if key not in seen:
                seen[key] = {
                    "grid": aligned,
                    "scale": int(scale),
                    "compressed_shape": tuple(map(int, small.shape)),
                    "align_op": int(align_op),
                }

    return list(seen.values())


def _extract_panels(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = components_mask(g != bg, connectivity=4)
    assert len(comps) == 3, "expected exactly three testimony regions"

    panels = []
    for coords in comps:
        r0, c0, r1, c1 = bbox_coords(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert coords.shape[0] == patch.size, "each testimony region must be a solid rectangle"
        assert not np.any(patch == bg), "testimony regions must not contain separator color"
        panels.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "patch": patch,
                "shape": tuple(map(int, patch.shape)),
                "area": int(patch.size),
            }
        )

    panels.sort(key=lambda item: (int(item["area"]), int(item["bbox"][0]), int(item["bbox"][1])))
    assert len({int(panel["area"]) for panel in panels}) == 3, "scales must be distinct"
    return bg, panels


def _disagreement_score(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> int:
    return int(np.sum(a != b) + np.sum(a != c) + np.sum(b != c))


def _recover_aligned_versions(grid: np.ndarray) -> dict:
    _bg, panels = _extract_panels(grid)
    ref = panels[0]["patch"]
    ref_shape = tuple(map(int, ref.shape))
    candidates_1 = _candidate_alignments(panels[1]["patch"], ref_shape)
    candidates_2 = _candidate_alignments(panels[2]["patch"], ref_shape)
    assert candidates_1 and candidates_2, "each enlarged testimony must compress to the smallest size"

    best_score: int | None = None
    second_score: int | None = None
    best_combo: tuple[dict, dict] | None = None

    for cand_1 in candidates_1:
        for cand_2 in candidates_2:
            score = _disagreement_score(ref, cand_1["grid"], cand_2["grid"])
            if best_score is None or score < best_score:
                second_score = best_score
                best_score = score
                best_combo = (cand_1, cand_2)
            elif score != best_score and (second_score is None or score < second_score):
                second_score = score

    assert best_combo is not None
    assert second_score is None or best_score < second_score, "alignment must be unique"
    aligned = [ref.copy(), best_combo[0]["grid"].copy(), best_combo[1]["grid"].copy()]
    return {
        "panels": panels,
        "aligned": aligned,
        "best_score": int(best_score),
        "second_score": None if second_score is None else int(second_score),
        "candidate_counts": (len(candidates_1), len(candidates_2)),
        "chosen": best_combo,
    }


def _majority_vote_three(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ambiguous = (a != b) & (a != c) & (b != c)
    assert not bool(np.any(ambiguous)), "three testimonies must admit a per-cell majority"
    return np.where(a == b, a, np.where(a == c, a, b)).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """Register the three scaled testimonies, majority-vote their cells, and return the repaired smallest picture.

    Input structure:
    - A dominant separator color splits the canvas into three disconnected solid rectangular testimony panels.
    - The three panels are the same mother picture at scales 1, k2, and k3, with k2 and k3 distinct integers greater than 1.
    - Each panel is fully tiled by integer scale blocks and may be independently rotated or mirrored by one of the eight orthogonal orientations.

    Rule:
    - Extract the three rectangular panels and take the smallest one as the target size and reference orientation.
    - For each larger panel, infer its integer scale from its side lengths, compress each uniform scale block back to one cell, and try all orthogonal orientations.
    - Choose the unique alignment that makes the three reduced panels agree as much as possible, then take the per-cell majority color across the three aligned panels.

    Key insight:
    - The testimonies must be normalized along two independent axes, scale and orientation, before their sparse block errors can be separated from true pattern differences by three-way majority vote.
    """
    recovered = _recover_aligned_versions(np.asarray(grid, dtype=int))
    a, b, c = recovered["aligned"]
    return _majority_vote_three(a, b, c)


def _orientation_margin(grid: np.ndarray) -> int:
    base = np.asarray(grid, dtype=int)
    margins = []
    for op in range(1, ORIENTATION_COUNT):
        other = _orient(base, op)
        if other.shape == base.shape:
            margins.append(int(np.sum(base != other)))
    return min(margins) if margins else int(base.size)


def _sample_mother(palette: list[int]) -> np.ndarray | None:
    h = random.randint(3, 5)
    w = random.randint(4, 6)
    if h == w:
        w += 1
    if h * w < 12:
        return None
    color_count = random.randint(3, min(5, len(palette)))
    colors = random.sample(palette, color_count)
    mother = np.random.choice(colors, size=(h, w)).astype(int)
    if len(np.unique(mother)) < 3:
        return None
    if _orientation_margin(mother) < max(3, mother.size // 4):
        return None
    if any(np.all(row == row[0]) for row in mother):
        return None
    if any(np.all(col == col[0]) for col in mother.T):
        return None
    return mother


def _choose_scales() -> list[int]:
    enlarged = sorted(random.sample((2, 3, 4), 2))
    return [1, *enlarged]


def _sample_orientations() -> list[int]:
    ops = [0, random.randrange(ORIENTATION_COUNT), random.randrange(ORIENTATION_COUNT)]
    if ops[1] == 0 and ops[2] == 0:
        ops[random.choice((1, 2))] = random.choice((1, 2, 3, 4, 5, 6, 7))
    return ops


def _sample_error_sets(shape: tuple[int, int]) -> list[list[tuple[int, int]]]:
    h, w = map(int, shape)
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    extra = random.randint(0, min(3, len(cells) - 3))
    counts = [1, 1, 1]
    for _ in range(extra):
        counts[random.randrange(3)] += 1
    chosen = cells[: sum(counts)]
    out: list[list[tuple[int, int]]] = []
    start = 0
    for count in counts:
        out.append(chosen[start : start + count])
        start += count
    return out


def _apply_errors(
    mother: np.ndarray,
    palette: list[int],
    orientation: int,
    scale: int,
    error_positions: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    oriented = _orient(mother, orientation)
    scaled = scale_sprite(oriented, scale)
    aligned_small = mother.copy()

    for r, c in error_positions:
        true_color = int(mother[r, c])
        wrong_choices = [color for color in palette if int(color) != true_color]
        wrong_color = int(random.choice(wrong_choices))
        aligned_small[r, c] = wrong_color
        rr, cc = _map_coord(mother.shape, orientation, r, c)
        scaled[rr * scale : (rr + 1) * scale, cc * scale : (cc + 1) * scale] = wrong_color

    return scaled.astype(int), aligned_small.astype(int)


def _layout_panels(patches: list[np.ndarray], bg: int) -> np.ndarray | None:
    bar = random.randint(1, 2)
    axis = random.choice(("horizontal", "vertical"))
    heights = [int(p.shape[0]) for p in patches]
    widths = [int(p.shape[1]) for p in patches]

    if axis == "horizontal":
        total_w = sum(widths) + 2 * bar
        if total_w > 30:
            return None
        max_h = max(heights)
        slack = random.randint(0, min(4, 30 - max_h))
        canvas_h = max_h + slack
        grid = np.full((canvas_h, total_w), int(bg), dtype=int)
        c0 = 0
        for patch in patches:
            r0 = random.randint(0, canvas_h - patch.shape[0])
            grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch
            c0 += patch.shape[1]
            if c0 < total_w:
                c0 += bar
        return grid

    total_h = sum(heights) + 2 * bar
    if total_h > 30:
        return None
    max_w = max(widths)
    slack = random.randint(0, min(4, 30 - max_w))
    canvas_w = max_w + slack
    grid = np.full((total_h, canvas_w), int(bg), dtype=int)
    r0 = 0
    for patch in patches:
        c0 = random.randint(0, canvas_w - patch.shape[1])
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch
        r0 += patch.shape[0]
        if r0 < total_h:
            r0 += bar
    return grid


def _bg_is_unique_majority(grid: np.ndarray, bg: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)[::-1]
    if len(order) < 2:
        return int(vals[order[0]]) == int(bg)
    return int(vals[order[0]]) == int(bg) and int(counts[order[0]]) > int(counts[order[1]])


def _instance_is_semantic(
    grid: np.ndarray,
    expected: np.ndarray,
    aligned_truths: list[np.ndarray],
    scales: list[int],
    orientations: list[int],
) -> bool:
    if not np.array_equal(main(grid), expected):
        return False

    recovered = _recover_aligned_versions(grid)
    if recovered["best_score"] <= 0:
        return False
    if recovered["second_score"] is not None and recovered["best_score"] >= recovered["second_score"]:
        return False

    for got, want in zip(recovered["aligned"], aligned_truths):
        if not np.array_equal(got, want):
            return False

    if not any(op != orientations[0] for op in orientations[1:]):
        return False
    if not all(scale > 0 for scale in scales):
        return False

    for witness in aligned_truths:
        if np.array_equal(witness, expected):
            return False

    for i in range(3):
        for j in range(i + 1, 3):
            if not np.any(aligned_truths[i] != aligned_truths[j]):
                return False

    return True


def _make_instance() -> tuple[np.ndarray, np.ndarray] | None:
    bg = random.choice(ALL_COLORS)
    palette = [color for color in ALL_COLORS if color != bg]
    mother = _sample_mother(palette)
    if mother is None:
        return None

    scales = _choose_scales()
    orientations = _sample_orientations()
    error_sets = _sample_error_sets(mother.shape)

    patches = []
    aligned_truths = []
    for scale, orientation, errors in zip(scales, orientations, error_sets):
        patch, aligned_small = _apply_errors(mother, palette, orientation, scale, errors)
        patches.append(patch)
        aligned_truths.append(aligned_small)

    grid = _layout_panels(patches, bg)
    if grid is None:
        return None
    if max(grid.shape) > 30:
        return None
    if not _bg_is_unique_majority(grid, bg):
        return None

    expected = _majority_vote_three(*aligned_truths)
    if not np.array_equal(expected, mother):
        return None
    if not _instance_is_semantic(grid, expected, aligned_truths, scales, orientations):
        return None
    return grid.astype(int), expected.astype(int)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        instance = _make_instance()
        if instance is not None:
            return instance[0]
    raise RuntimeError("failed to generate a valid three-testimony ARC input")
