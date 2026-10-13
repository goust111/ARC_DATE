from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
MARKER = 4
TEMPLATE_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _erode4(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask.copy()
    pad = np.pad(mask, 1, constant_values=False)
    return (
        mask
        & pad[:-2, 1:-1]
        & pad[2:, 1:-1]
        & pad[1:-1, :-2]
        & pad[1:-1, 2:]
    )


def _erode_steps(mask: np.ndarray, steps: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(int(steps)):
        out = _erode4(out)
        if not np.any(out):
            break
    return out


def _first_empty_step(mask: np.ndarray) -> int:
    current = mask.copy()
    step = 0
    while np.any(current):
        current = _erode4(current)
        step += 1
        if step > 10:
            break
    return step


def _neighbor_count(mask: np.ndarray, r: int, c: int) -> int:
    total = 0
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
            total += 1
    return total


def _endpoint_count(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    return sum(_neighbor_count(mask, int(r), int(c)) == 1 for r, c in coords)


def _naive_bbox_shrink(mask: np.ndarray, steps: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    if 2 * steps >= h or 2 * steps >= w:
        return out
    out[steps : h - steps, steps : w - steps] = mask[steps : h - steps, steps : w - steps]
    return out


def _multiple_of(diff: tuple[int, int], base: tuple[int, int]) -> int | None:
    dr, dc = diff
    br, bc = base
    if br == 0 and bc == 0:
        return None
    if br == 0:
        if dr != 0 or bc == 0 or dc % bc != 0:
            return None
        k = dc // bc
    elif bc == 0:
        if dc != 0 or dr % br != 0:
            return None
        k = dr // br
    else:
        if dr * bc != dc * br:
            return None
        if dr % br != 0 or dc % bc != 0:
            return None
        k = dr // br
        if k != dc // bc:
            return None
    return int(k) if k > 0 else None


def _template_mask_from_grid(grid: np.ndarray) -> tuple[int, np.ndarray, tuple[int, int]]:
    bg = _get_bg(grid)
    fg_colors = [int(v) for v in np.unique(grid) if int(v) != bg]
    assert len(fg_colors) == 2, "expected exactly two foreground colors"

    candidates: list[tuple[int, int, np.ndarray]] = []
    for color in fg_colors:
        for coords in _components(grid == color):
            candidates.append((len(coords), color, coords))

    max_area = max(area for area, _color, _coords in candidates)
    template_candidates = [(color, coords) for area, color, coords in candidates if area == max_area]
    assert len(template_candidates) == 1 and max_area > 1, "expected one large template component"
    template_color, template_coords = template_candidates[0]
    r0, c0, r1, c1 = _bbox_from_coords(template_coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for rr, cc in template_coords:
        mask[int(rr) - r0, int(cc) - c0] = True
    return int(template_color), mask, (int(r0), int(c0))


def _marker_positions(grid: np.ndarray, template_color: int) -> np.ndarray:
    bg = _get_bg(grid)
    colors = [int(v) for v in np.unique(grid) if int(v) != bg]
    marker_color = [color for color in colors if color != template_color][0]
    comps = _components(grid == marker_color)
    assert len(comps) >= 2 and all(len(comp) == 1 for comp in comps), "markers must be singleton pixels"
    return np.array([comp[0] for comp in comps], dtype=int)


def _render_output(
    canvas_shape: tuple[int, int],
    template_mask: np.ndarray,
    origin: tuple[int, int],
    vector: tuple[int, int],
    count: int,
    color: int,
) -> np.ndarray:
    out = np.full(canvas_shape, BG, dtype=int)
    coords0 = np.argwhere(template_mask)
    base_r, base_c = origin
    for rr, cc in coords0:
        out[base_r + int(rr), base_c + int(cc)] = color

    current = template_mask.copy()
    for k in range(1, count + 1):
        current = _erode4(current)
        if not np.any(current):
            continue
        shift_r = base_r + k * vector[0]
        shift_c = base_c + k * vector[1]
        for rr, cc in np.argwhere(current):
            out[shift_r + int(rr), shift_c + int(cc)] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp translated template copies whose layers are peeled away by 4-neighbor erosion.

    Input structure:
    - A black background contains one connected monochrome template.
    - Separate yellow singleton markers lie on one equally spaced straight lattice line.
    - The template bbox upper-left corner is the lattice 0 site: the nearest marker is one lattice step away, and later markers continue the same step.

    Rule:
    - Keep the original template unchanged at lattice site 0.
    - Number the markers by distance from the template bbox upper-left corner as k=1,2,...
    - At the kth marker, place a translated template copy whose bbox upper-left corner lands on that marker, then erode that copy k times; each erosion removes every pixel that is 4-neighbor adjacent to background.
    - Remove the yellow markers; if a copy becomes empty after its k erosions, leave nothing there.

    Key insight:
    - The distant copies are not scaled versions: each one is the same template after repeated boundary stripping, so notches deepen and thin arms disappear first.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    template_color, template_mask, origin = _template_mask_from_grid(g)
    markers = _marker_positions(g, template_color)

    diffs = [(int(r - origin[0]), int(c - origin[1])) for r, c in markers]
    vector = min(diffs, key=lambda rc: rc[0] * rc[0] + rc[1] * rc[1])
    assert vector != (0, 0), "invalid lattice vector"

    indexed: list[tuple[int, tuple[int, int]]] = []
    for diff in diffs:
        k = _multiple_of(diff, vector)
        assert k is not None, "markers must lie on one equally spaced line"
        indexed.append((k, diff))

    ordered = sorted(indexed)
    ks = [k for k, _diff in ordered]
    assert ks == list(range(1, len(ks) + 1)), "markers must occupy consecutive lattice positions"

    return _render_output(g.shape, template_mask, origin, vector, len(ks), template_color)


def _make_notched_template() -> np.ndarray:
    h = random.randint(5, 7)
    w = random.randint(5, 7)
    mask = np.ones((h, w), dtype=bool)
    if random.random() < 0.5:
        depth = random.randint(2, w - 2)
        length = random.randint(1, h - 2)
        r0 = random.randint(1, h - length - 1)
        mask[r0 : r0 + length, w - depth : w] = False
    else:
        depth = random.randint(2, h - 2)
        length = random.randint(1, w - 2)
        c0 = random.randint(1, w - length - 1)
        mask[h - depth : h, c0 : c0 + length] = False
    return mask


def _make_arm_template() -> np.ndarray:
    h = random.randint(5, 7)
    w = random.randint(5, 7)
    mask = np.zeros((h, w), dtype=bool)
    if random.random() < 0.5:
        core_w = random.randint(3, w - 1)
        arm_r = random.randint(1, h - 2)
        mask[:, :core_w] = True
        mask[arm_r, core_w:w] = True
    else:
        core_h = random.randint(3, h - 1)
        arm_c = random.randint(1, w - 2)
        mask[:core_h, :] = True
        mask[core_h:h, arm_c] = True
    return mask


def _make_hybrid_template() -> np.ndarray:
    h = random.randint(5, 7)
    w = random.randint(5, 7)
    mask = np.zeros((h, w), dtype=bool)
    core_w = random.randint(4, w - 1)
    arm_r = random.randint(1, h - 2)
    mask[:, :core_w] = True
    mask[arm_r, core_w:w] = True
    depth = random.randint(1, h - 2)
    length = random.randint(1, core_w - 2)
    c0 = random.randint(1, core_w - length - 1)
    mask[h - depth : h, c0 : c0 + length] = False
    return mask


def _template_is_valid(mask: np.ndarray) -> bool:
    if not mask[0, 0] or not mask[-1, :].any() or not mask[:, -1].any():
        return False
    if len(_components(mask)) != 1:
        return False
    if np.count_nonzero(mask) < 10:
        return False

    er1 = _erode4(mask)
    if not np.any(er1):
        return False
    empty_step = _first_empty_step(mask)
    if empty_step not in (2, 3):
        return False

    has_inner_bg = bool(np.any(~mask[1:-1, 1:-1]))
    has_endpoint = _endpoint_count(mask) >= 1
    if not (has_inner_bg or has_endpoint):
        return False

    if np.array_equal(er1, _naive_bbox_shrink(mask, 1)) and empty_step == 2:
        return False
    if empty_step == 3 and np.array_equal(_erode_steps(mask, 2), _naive_bbox_shrink(mask, 2)):
        return False
    return True


def _make_template() -> np.ndarray | None:
    factories = [_make_notched_template, _make_arm_template, _make_hybrid_template]
    for _ in range(100):
        mask = random.choice(factories)()
        if _template_is_valid(mask):
            return mask
    return None


def _candidate_vectors(h: int, w: int, count: int) -> list[tuple[int, int]]:
    vectors: list[tuple[int, int]] = []
    for gap in (1, 2):
        vectors.extend(
            [
                (0, w + gap),
                (0, -(w + gap)),
                (h + gap, 0),
                (-(h + gap), 0),
            ]
        )
    for gap_r in (0, 1):
        for gap_c in (0, 1):
            vr = h + gap_r
            vc = w + gap_c
            vectors.extend([(vr, vc), (vr, -vc), (-vr, vc), (-vr, -vc)])

    valid: list[tuple[int, int]] = []
    for vr, vc in vectors:
        min_r = min(0, *(k * vr for k in range(1, count + 1)))
        max_r = max(h - 1, *(k * vr + h - 1 for k in range(1, count + 1)))
        min_c = min(0, *(k * vc for k in range(1, count + 1)))
        max_c = max(w - 1, *(k * vc + w - 1 for k in range(1, count + 1)))
        if max_r - min_r + 1 <= 30 and max_c - min_c + 1 <= 30:
            valid.append((vr, vc))
    return valid


def _semantic_checks(
    grid: np.ndarray,
    template_mask: np.ndarray,
    origin: tuple[int, int],
    vector: tuple[int, int],
    marker_count: int,
    template_color: int,
) -> bool:
    expected = _render_output(grid.shape, template_mask, origin, vector, marker_count, template_color)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    er1 = _erode4(template_mask)
    if np.array_equal(er1, template_mask):
        return False

    areas = [int(np.count_nonzero(_erode_steps(template_mask, k))) for k in range(marker_count + 1)]
    if areas[1] <= 0:
        return False
    if any(a < b for a, b in zip(areas, areas[1:])):
        return False
    if marker_count >= 2 and areas[1] == areas[2]:
        return False

    empty_step = _first_empty_step(template_mask)
    if marker_count == empty_step:
        mr = origin[0] + marker_count * vector[0]
        mc = origin[1] + marker_count * vector[1]
        if actual[mr, mc] != BG:
            return False

    diffs = [(int(r - origin[0]), int(c - origin[1])) for r, c in np.argwhere(grid == MARKER)]
    recovered = sorted(_multiple_of(diff, vector) for diff in diffs)
    if recovered != list(range(1, marker_count + 1)):
        return False
    return True


def _generate_once() -> np.ndarray | None:
    template_mask = _make_template()
    if template_mask is None:
        return None

    empty_step = _first_empty_step(template_mask)
    marker_count = random.choice([2, min(3, empty_step)])
    h, w = template_mask.shape
    vectors = _candidate_vectors(h, w, marker_count)
    if not vectors:
        return None
    vector = random.choice(vectors)

    vr, vc = vector
    min_r = min(0, *(k * vr for k in range(1, marker_count + 1)))
    max_r = max(h - 1, *(k * vr + h - 1 for k in range(1, marker_count + 1)))
    min_c = min(0, *(k * vc for k in range(1, marker_count + 1)))
    max_c = max(w - 1, *(k * vc + w - 1 for k in range(1, marker_count + 1)))
    span_h = max_r - min_r + 1
    span_w = max_c - min_c + 1

    free_h = 30 - span_h
    free_w = 30 - span_w
    if free_h < 2 or free_w < 2:
        return None

    top_pad = random.randint(1, min(4, free_h - 1))
    bottom_pad = random.randint(1, free_h - top_pad)
    left_pad = random.randint(1, min(4, free_w - 1))
    right_pad = random.randint(1, free_w - left_pad)
    grid_h = span_h + top_pad + bottom_pad
    grid_w = span_w + left_pad + right_pad
    origin = (-min_r + top_pad, -min_c + left_pad)

    color = random.choice(TEMPLATE_COLORS)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for rr, cc in np.argwhere(template_mask):
        grid[origin[0] + int(rr), origin[1] + int(cc)] = color
    for k in range(1, marker_count + 1):
        mr = origin[0] + k * vr
        mc = origin[1] + k * vc
        grid[mr, mc] = MARKER

    if not _semantic_checks(grid, template_mask, origin, vector, marker_count, color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
