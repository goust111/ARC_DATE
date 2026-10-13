from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _ring_coords_from_bbox(b: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = b
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        pts.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        pts.append((r, c0))
    return pts


def _component_mask(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    b = _bbox(coords)
    r0, c0, r1, c1 = b
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return b, mask


def _is_rectangular_ring(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    expected = np.zeros_like(mask, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(mask, expected))


def _read_legend(grid: np.ndarray, legend_coords: np.ndarray) -> list[int]:
    order = np.lexsort((legend_coords[:, 1], legend_coords[:, 0]))
    sorted_coords = legend_coords[order]
    return [int(grid[r, c]) for r, c in sorted_coords]


def _dominant_color(colors_in_order: list[int], legend: list[int]) -> int:
    counts = Counter(colors_in_order)
    max_count = max(counts.values())
    candidates = [color for color, count in counts.items() if count == max_count]
    if len(candidates) == 1:
        return int(candidates[0])

    legend_rank = {color: idx for idx, color in enumerate(legend)}
    first_seen: dict[int, int] = {}
    for idx, color in enumerate(colors_in_order):
        first_seen.setdefault(int(color), idx)

    def key(color: int) -> tuple[int, int]:
        if color in legend_rank:
            return (0, int(legend_rank[color]))
        return (1, int(first_seen[color]))

    return int(min(candidates, key=key))


def _legend_operation(dominant: int, legend: list[int]) -> str:
    if dominant not in legend:
        return "keep"
    idx = int(legend.index(dominant))
    if idx == 1:
        return "delete"
    if idx == 2:
        return "fill"
    return "keep"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean a nested set of concentric rings according to a corner legend and each ring's dominant color.

    Input structure:
    - The background is Black.
    - One cluster contains 3 or 4 concentric one-pixel rectangular rings drawn in Grey, with one Black gap between neighboring rings.
    - Colored pixels may replace some Grey border pixels on any ring, and a separate corner legend is a vertical list of 2 to 4 colored pixels.

    Rule:
    - Read the corner legend from top to bottom to get an ordered color list.
    - For each ring, count its non-Grey colored pixels and pick the dominant color; ties are broken by earlier appearance in the legend.
    - A dominant color matching the legend's first entry keeps the ring, the second deletes the whole ring to Black, and the third repaints the whole ring with that dominant color.
    - Dominant colors outside the legend, or in later legend slots, leave the ring unchanged, and the legend area itself is cleared to Black.

    Key insight:
    - The legend encodes operations by order rather than by direct recolor mapping, so the ring must first be identified by layer and summarized by its dominant embedded color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    ring_infos: list[dict[str, object]] = []
    ring_mask = np.zeros_like(g, dtype=bool)
    for coords in _components(g != BLACK):
        b, mask = _component_mask(coords)
        if not _is_rectangular_ring(mask):
            continue
        order = _ring_coords_from_bbox(b)
        ring_infos.append({"bbox": b, "coords": coords, "order": order})
        ring_mask[coords[:, 0], coords[:, 1]] = True

    assert 3 <= len(ring_infos) <= 4

    legend_coords = np.argwhere((g != BLACK) & (~ring_mask))
    assert 2 <= len(legend_coords) <= 4
    legend = _read_legend(g, legend_coords)
    assert len(set(legend)) == len(legend)

    out = g.copy()
    out[legend_coords[:, 0], legend_coords[:, 1]] = BLACK

    for info in sorted(ring_infos, key=lambda item: item["bbox"][0]):  # type: ignore[index]
        order = info["order"]  # type: ignore[assignment]
        coords = info["coords"]  # type: ignore[assignment]
        colors_in_order = [
            int(g[r, c])
            for r, c in order
            if int(g[r, c]) not in (BLACK, GREY)
        ]
        if not colors_in_order:
            continue
        dominant = _dominant_color(colors_in_order, legend)
        op = _legend_operation(dominant, legend)
        if op == "delete":
            out[coords[:, 0], coords[:, 1]] = BLACK
        elif op == "fill":
            out[coords[:, 0], coords[:, 1]] = int(dominant)

    return out


def _choose_layout(
    n_rings: int, legend_len: int, rng: random.Random
) -> tuple[int, int, int, int, int, int, str]:
    min_dim = 3 + 4 * (n_rings - 1)
    for _ in range(200):
        corner = rng.choice(("tl", "tr", "bl", "br"))
        top_min = legend_len + 1 if corner in ("tl", "tr") else 1
        bottom_min = legend_len + 1 if corner in ("bl", "br") else 1
        left_min = 2 if corner in ("tl", "bl") else 1
        right_min = 2 if corner in ("tr", "br") else 1

        top = rng.randint(top_min, min(top_min + 4, 8))
        bottom = rng.randint(bottom_min, min(bottom_min + 4, 8))
        left = rng.randint(left_min, min(left_min + 4, 8))
        right = rng.randint(right_min, min(right_min + 4, 8))

        max_h = 30 - top - bottom
        max_w = 30 - left - right
        if max_h < min_dim or max_w < min_dim:
            continue

        outer_h = rng.randint(min_dim, min(max_h, 23))
        outer_w = rng.randint(min_dim, min(max_w, 23))
        return outer_h + top + bottom, outer_w + left + right, top, left, outer_h, outer_w, corner
    raise RuntimeError("failed to choose a valid layout")


def _branch_choices(legend_len: int) -> list[str]:
    branches = ["empty", "keep", "delete", "outside"]
    if legend_len >= 3:
        branches.append("fill")
    if legend_len >= 4:
        branches.append("noop4")
    return branches


def _branch_dominant(branch: str, legend: list[int], outside_colors: list[int], rng: random.Random) -> int | None:
    if branch == "empty":
        return None
    if branch == "keep":
        return int(legend[0])
    if branch == "delete":
        return int(legend[1])
    if branch == "fill":
        return int(legend[2])
    if branch == "noop4":
        return int(legend[3])
    return int(rng.choice(outside_colors))


def _counts_for_branch(
    branch: str, dominant: int | None, legend: list[int], outside_colors: list[int], rng: random.Random
) -> list[tuple[int, int]]:
    if branch == "empty" or dominant is None:
        return []

    tie_options: list[int] = []
    if branch == "keep" and len(legend) >= 2:
        tie_options = legend[1:]
    elif branch == "delete" and len(legend) >= 3:
        tie_options = legend[2:]
    elif branch == "fill" and len(legend) >= 4:
        tie_options = legend[3:]

    mode_choices = ["plain", "mixed"]
    if tie_options:
        mode_choices.append("tie")
    mode = rng.choice(mode_choices)

    if mode == "plain":
        return [(int(dominant), rng.randint(2, 4))]

    if mode == "tie":
        partner = int(rng.choice(tie_options))
        count = rng.randint(2, 3)
        return [(int(dominant), count), (partner, count)]

    other_pool = [color for color in legend + outside_colors if color != dominant]
    rng.shuffle(other_pool)
    counts = [(int(dominant), rng.randint(3, 4))]
    extra_count = rng.randint(1, min(2, len(other_pool)))
    used = 0
    for color in other_pool:
        if used >= extra_count:
            break
        counts.append((int(color), 1))
        used += 1
    return counts


def _paint_ring(
    grid: np.ndarray, coords: list[tuple[int, int]], color_counts: list[tuple[int, int]], rng: random.Random
) -> None:
    if not color_counts:
        return
    total = sum(count for _, count in color_counts)
    assert total <= len(coords)
    positions = rng.sample(range(len(coords)), total)
    paint_colors: list[int] = []
    for color, count in color_counts:
        paint_colors.extend([int(color)] * int(count))
    rng.shuffle(paint_colors)
    for idx, color in zip(positions, paint_colors):
        r, c = coords[idx]
        grid[r, c] = int(color)


def _legend_positions(h: int, w: int, legend_len: int, corner: str) -> list[tuple[int, int]]:
    if corner == "tl":
        return [(r, 0) for r in range(legend_len)]
    if corner == "tr":
        return [(r, w - 1) for r in range(legend_len)]
    if corner == "bl":
        return [(h - legend_len + r, 0) for r in range(legend_len)]
    return [(h - legend_len + r, w - 1) for r in range(legend_len)]


def _expected_output(
    inp: np.ndarray,
    ring_meta: list[dict[str, object]],
    legend_positions: list[tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    for r, c in legend_positions:
        out[r, c] = BLACK
    for meta in ring_meta:
        op = meta["op"]
        coords = meta["coords"]
        if op == "delete":
            for r, c in coords:
                out[r, c] = BLACK
        elif op == "fill":
            fill_color = int(meta["dominant"])
            for r, c in coords:
                out[r, c] = fill_color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rings = rng.choice((3, 4))
    legend_len = rng.choice((2, 3, 4))
    legend = rng.sample(PALETTE_COLORS, legend_len)
    outside_colors = [color for color in PALETTE_COLORS if color not in legend]
    if not outside_colors:
        return None

    h, w, r0, c0, outer_h, outer_w, corner = _choose_layout(n_rings, legend_len, rng)
    grid = np.full((h, w), BLACK, dtype=int)

    ring_meta: list[dict[str, object]] = []
    available_branches = _branch_choices(legend_len)
    chosen_branches = [rng.choice(available_branches) for _ in range(n_rings)]
    if all(branch not in ("delete", "fill") for branch in chosen_branches):
        forced = "fill" if legend_len >= 3 and rng.random() < 0.5 else "delete"
        chosen_branches[rng.randrange(n_rings)] = forced

    for layer, branch in enumerate(chosen_branches):
        bbox = (
            r0 + 2 * layer,
            c0 + 2 * layer,
            r0 + outer_h - 1 - 2 * layer,
            c0 + outer_w - 1 - 2 * layer,
        )
        coords = _ring_coords_from_bbox(bbox)
        for r, c in coords:
            grid[r, c] = GREY

        dominant = _branch_dominant(branch, legend, outside_colors, rng)
        color_counts = _counts_for_branch(branch, dominant, legend, outside_colors, rng)
        _paint_ring(grid, coords, color_counts, rng)

        ring_meta.append(
            {
                "coords": coords,
                "dominant": dominant,
                "op": "keep" if dominant is None else _legend_operation(int(dominant), legend),
            }
        )

    legend_positions = _legend_positions(h, w, legend_len, corner)
    for (r, c), color in zip(legend_positions, legend):
        grid[r, c] = int(color)

    expected = _expected_output(grid, ring_meta, legend_positions)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
