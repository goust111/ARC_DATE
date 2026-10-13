from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists() and str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components


BLACK = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 500
_SEEN_OUTPUT_SIGS: set[str] = set()


def _infer_lattice_from_component(grid: np.ndarray, component: dict) -> dict | None:
    frame_color = int(component["color"])
    bbox = tuple(map(int, component["bbox"]))
    r0, c0, r1, c1 = bbox
    sub = crop_bbox(grid, bbox)
    frame_mask = sub == frame_color

    row_lines = [r for r in range(sub.shape[0]) if bool(np.all(frame_mask[r, :]))]
    col_lines = [c for c in range(sub.shape[1]) if bool(np.all(frame_mask[:, c]))]
    if len(row_lines) < 3 or len(col_lines) < 3:
        return None
    if row_lines[0] != 0 or row_lines[-1] != sub.shape[0] - 1:
        return None
    if col_lines[0] != 0 or col_lines[-1] != sub.shape[1] - 1:
        return None

    row_steps = [b - a for a, b in zip(row_lines, row_lines[1:])]
    col_steps = [b - a for a, b in zip(col_lines, col_lines[1:])]
    if len(set(row_steps)) != 1 or len(set(col_steps)) != 1:
        return None
    if row_steps[0] <= 1 or col_steps[0] <= 1:
        return None

    template_mask = np.zeros_like(frame_mask, dtype=bool)
    template_mask[row_lines, :] = True
    template_mask[:, col_lines] = True
    if not np.array_equal(frame_mask, template_mask):
        return None

    chamber_h = row_steps[0] - 1
    chamber_w = col_steps[0] - 1
    chambers: list[tuple[int, int, int, int]] = []
    for i in range(len(row_lines) - 1):
        rr0 = r0 + row_lines[i] + 1
        rr1 = r0 + row_lines[i + 1] - 1
        if rr1 - rr0 + 1 != chamber_h:
            return None
        for j in range(len(col_lines) - 1):
            cc0 = c0 + col_lines[j] + 1
            cc1 = c0 + col_lines[j + 1] - 1
            if cc1 - cc0 + 1 != chamber_w:
                return None
            chambers.append((rr0, cc0, rr1, cc1))

    return {
        "frame_color": frame_color,
        "bbox": bbox,
        "template_mask": template_mask,
        "row_lines": row_lines,
        "col_lines": col_lines,
        "chambers": chambers,
        "count": len(chambers),
    }


def _find_lattice_candidates(grid: np.ndarray) -> list[dict]:
    components = find_components(
        grid,
        bg=BLACK,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    candidates: list[dict] = []
    for component in components:
        if int(component["area"]) < 8:
            continue
        info = _infer_lattice_from_component(grid, component)
        if info is not None:
            candidates.append(info)
    return candidates


def _lattice_template_full(shape: tuple[int, int], info: dict) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = info["bbox"]
    out[r0 : r1 + 1, c0 : c1 + 1] = info["template_mask"]
    return out


def _fill_output(
    shape: tuple[int, int],
    frame_color: int,
    template_full: np.ndarray,
    chambers: list[tuple[int, int, int, int]],
    counts: dict[int, int],
    *,
    chamber_order: list[int] | None = None,
    use_wrong_mod: bool = False,
) -> np.ndarray:
    ordered = chambers if chamber_order is None else [chambers[i] for i in chamber_order]
    k = len(chambers)
    out = np.zeros(shape, dtype=int)
    out[template_full] = frame_color
    for color, n in counts.items():
        idx = (n % k) if use_wrong_mod else ((n - 1) % k)
        r0, c0, r1, c1 = ordered[idx]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the scatter pixels of each color, use those counts as modular chamber addresses, and paint the matching lattice cavities.

    Input structure:
    - A single periodic monochrome lattice frame encloses several equal-size black rectangular cavities in row-major order.
    - Outside that template sit isolated scatter pixels of several colors, including some pixels that share the frame color but are not on template positions.
    - The surrounding background and all unfilled cavities are black.

    Rule:
    - Recover the periodic frame template and list its enclosed cavities in reading order from top-left to bottom-right.
    - Treat every non-black pixel that is not on a template position as a scatter counter pixel, including same-color frame impostors.
    - For each scatter color with count n, fill cavity number ((n-1) mod K)+1 entirely with that color, where K is the number of cavities.
    - Erase all scatter pixels while keeping the lattice frame unchanged.

    Key insight:
    - Pixel placement is a decoy: the only information carried by the scattered dots is each color's count, and the frame color's fake dots can only be separated from the true skeleton by reconstructing the periodic lattice template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    candidates = _find_lattice_candidates(g)
    assert len(candidates) == 1, "expected one uniquely decodable lattice frame"
    info = candidates[0]
    template_full = _lattice_template_full(g.shape, info)

    scatter_mask = (g != BLACK) & ~template_full
    counts = Counter(map(int, g[scatter_mask]))
    assert counts, "expected at least one scatter counter color"

    targets = [((n - 1) % info["count"]) for n in counts.values()]
    assert len(set(targets)) == len(targets), "scatter counts must address distinct cavities"

    return _fill_output(
        g.shape,
        info["frame_color"],
        template_full,
        info["chambers"],
        {int(color): int(n) for color, n in counts.items()},
    )


def _draw_lattice(
    grid: np.ndarray,
    top: int,
    left: int,
    chamber_rows: int,
    chamber_cols: int,
    chamber_h: int,
    chamber_w: int,
    frame_color: int,
) -> tuple[int, int, int, int]:
    height = chamber_rows * chamber_h + chamber_rows + 1
    width = chamber_cols * chamber_w + chamber_cols + 1
    r1 = top + height - 1
    c1 = left + width - 1

    for i in range(chamber_rows + 1):
        rr = top + i * (chamber_h + 1)
        grid[rr, left : c1 + 1] = frame_color
    for j in range(chamber_cols + 1):
        cc = left + j * (chamber_w + 1)
        grid[top : r1 + 1, cc] = frame_color
    return top, left, r1, c1


def _chamber_boxes(
    top: int,
    left: int,
    chamber_rows: int,
    chamber_cols: int,
    chamber_h: int,
    chamber_w: int,
) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for i in range(chamber_rows):
        for j in range(chamber_cols):
            r0 = top + 1 + i * (chamber_h + 1)
            c0 = left + 1 + j * (chamber_w + 1)
            out.append((r0, c0, r0 + chamber_h - 1, c0 + chamber_w - 1))
    return out


def _scatter_positions(
    shape: tuple[int, int],
    lattice_bbox: tuple[int, int, int, int],
    needed: int,
) -> list[tuple[int, int]] | None:
    h, w = shape
    r0, c0, r1, c1 = lattice_bbox
    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if not (r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1)
    ]
    random.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) >= 2 for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == needed:
                return chosen
    return None


def _count_map(
    chamber_count: int,
    colors: list[int],
    frame_color: int,
) -> dict[int, int] | None:
    if chamber_count < len(colors):
        return None

    targets = {chamber_count}
    while len(targets) < len(colors):
        targets.add(random.randint(1, chamber_count))
    targets_list = list(targets)
    random.shuffle(targets_list)

    counts_by_target: dict[int, int] = {}
    boosted = False
    for target in targets_list:
        if target == chamber_count:
            wraps = random.randint(1, 2)
            counts_by_target[target] = target + wraps * chamber_count
            boosted = True
        else:
            wraps = 1 if random.random() < 0.4 else 0
            counts_by_target[target] = target + wraps * chamber_count
    if not boosted:
        return None

    color_order = colors[:]
    random.shuffle(color_order)
    if frame_color not in color_order:
        return None

    result = {int(color): int(counts_by_target[target]) for color, target in zip(color_order, targets_list)}
    if frame_color not in result:
        return None
    if len(set(result.values())) != len(result):
        return None
    residues = {((n - 1) % chamber_count) for n in result.values()}
    if len(residues) != len(result):
        return None
    if max(result.values()) <= chamber_count:
        return None
    if result[frame_color] <= 0:
        return None
    return result


def _alternative_orders(
    chambers: list[tuple[int, int, int, int]],
) -> tuple[list[int], list[int]]:
    row_major = list(range(len(chambers)))
    reverse_order = list(reversed(row_major))
    column_major = sorted(row_major, key=lambda idx: (chambers[idx][1], chambers[idx][0]))
    return reverse_order, column_major


def _scene_is_valid(grid: np.ndarray, spec: dict) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False

    expected = _fill_output(
        grid.shape,
        spec["frame_color"],
        spec["template_full"],
        spec["chambers"],
        spec["counts"],
    )
    if not np.array_equal(actual, expected):
        return False

    if len(_find_lattice_candidates(grid)) != 1:
        return False

    reverse_order, column_major = _alternative_orders(spec["chambers"])
    alt_reverse = _fill_output(
        grid.shape,
        spec["frame_color"],
        spec["template_full"],
        spec["chambers"],
        spec["counts"],
        chamber_order=reverse_order,
    )
    alt_column = _fill_output(
        grid.shape,
        spec["frame_color"],
        spec["template_full"],
        spec["chambers"],
        spec["counts"],
        chamber_order=column_major,
    )
    if np.array_equal(expected, alt_reverse) or np.array_equal(expected, alt_column):
        return False

    alt_wrong_mod = _fill_output(
        grid.shape,
        spec["frame_color"],
        spec["template_full"],
        spec["chambers"],
        spec["counts"],
        use_wrong_mod=True,
    )
    if np.array_equal(expected, alt_wrong_mod):
        return False

    no_frame_counts = {color: count for color, count in spec["counts"].items() if color != spec["frame_color"]}
    alt_no_frame = _fill_output(
        grid.shape,
        spec["frame_color"],
        spec["template_full"],
        spec["chambers"],
        no_frame_counts,
    )
    if np.array_equal(expected, alt_no_frame):
        return False

    return True


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    chamber_rows = random.randint(2, 4)
    chamber_cols = random.randint(2, 4)
    chamber_h = random.randint(2, 4)
    chamber_w = random.randint(2, 4)
    lattice_h = chamber_rows * chamber_h + chamber_rows + 1
    lattice_w = chamber_cols * chamber_w + chamber_cols + 1

    grid_h = random.randint(max(lattice_h + 3, 18), 30)
    grid_w = random.randint(max(lattice_w + 3, 18), 30)
    top = random.randint(1, grid_h - lattice_h - 1)
    left = random.randint(1, grid_w - lattice_w - 1)

    frame_color = random.choice(PALETTE)
    colors = [frame_color]
    extra_count = 3 if chamber_rows * chamber_cols < 6 else random.randint(3, 4)
    extras = random.sample([c for c in PALETTE if c != frame_color], k=extra_count - 1)
    colors.extend(extras)

    counts = _count_map(chamber_rows * chamber_cols, colors, frame_color)
    if counts is None:
        return None

    total_scatter = sum(counts.values())
    grid = np.zeros((grid_h, grid_w), dtype=int)
    bbox = _draw_lattice(grid, top, left, chamber_rows, chamber_cols, chamber_h, chamber_w, frame_color)
    chambers = _chamber_boxes(top, left, chamber_rows, chamber_cols, chamber_h, chamber_w)
    positions = _scatter_positions(grid.shape, bbox, total_scatter)
    if positions is None:
        return None

    order = positions[:]
    random.shuffle(order)
    cursor = 0
    for color, count in counts.items():
        for _ in range(count):
            r, c = order[cursor]
            grid[r, c] = int(color)
            cursor += 1

    template_full = np.zeros(grid.shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    template_full[r0 : r1 + 1, c0 : c1 + 1] = (
        (np.indices((r1 - r0 + 1, c1 - c0 + 1))[0] % (chamber_h + 1) == 0)
        | (np.indices((r1 - r0 + 1, c1 - c0 + 1))[1] % (chamber_w + 1) == 0)
    )

    spec = {
        "frame_color": frame_color,
        "counts": counts,
        "chambers": chambers,
        "template_full": template_full,
    }
    if not _scene_is_valid(grid, spec):
        return None

    output = _fill_output(grid.shape, frame_color, template_full, chambers, counts)
    return grid, output


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _generate_attempt()
        if result is None:
            continue
        grid, output = result
        out_sig = hashlib.md5(output.tobytes() + str(output.shape).encode()).hexdigest()
        if out_sig in _SEEN_OUTPUT_SIGS:
            continue
        _SEEN_OUTPUT_SIGS.add(out_sig)
        return grid
    raise RuntimeError("failed to generate a valid lattice-counter ARC input")
