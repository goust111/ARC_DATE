from __future__ import annotations

import random

import numpy as np


BLACK = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)

GRID_SIDE_RANGE = (20, 30)
PAIR_COUNT_RANGE = (2, 4)
PAIR_GAP_RANGE = (1, 3)
GEN_RETRIES = 400
LOCAL_SHAPE_RETRIES = 120
PLACEMENT_RETRIES = 120
MARGIN = 1

NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8, copy=False).tobytes()


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    stack = [tuple(map(int, coords[0]))]
    seen[stack[0]] = True
    count = 0
    while stack:
        r, c = stack.pop()
        count += 1
        for dr, dc in NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return count == int(mask.sum())


def _find_components(grid: np.ndarray, bg: int = BLACK) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in NEIGHBORS8:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(pts, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )

    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def _project_rows(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for r in range(mask.shape[0]):
        cols = np.flatnonzero(mask[r])
        if cols.size:
            out[r, int(cols[0]) : int(cols[-1]) + 1] = True
    return out


def _project_cols(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for c in range(mask.shape[1]):
        rows = np.flatnonzero(mask[:, c])
        if rows.size:
            out[int(rows[0]) : int(rows[-1]) + 1, c] = True
    return out


def _relative_side(source_bbox: tuple[int, int, int, int], other_bbox: tuple[int, int, int, int]) -> str | None:
    sr0, sc0, sr1, sc1 = map(int, source_bbox)
    or0, oc0, or1, oc1 = map(int, other_bbox)
    row_overlap = _interval_overlap(sr0, sr1, or0, or1)
    col_overlap = _interval_overlap(sc0, sc1, oc0, oc1)
    if or1 < sr0 and col_overlap > 0:
        return "above"
    if or0 > sr1 and col_overlap > 0:
        return "below"
    if oc1 < sc0 and row_overlap > 0:
        return "left"
    if oc0 > sc1 and row_overlap > 0:
        return "right"
    return None


def _candidate_edges(components: list[dict]) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for i, source in enumerate(components):
        row_proj = _project_rows(source["mask"])
        col_proj = _project_cols(source["mask"])
        for j, other in enumerate(components):
            if i == j:
                continue
            side = _relative_side(source["bbox"], other["bbox"])
            if side in ("above", "below"):
                if row_proj.shape == other["mask"].shape and np.array_equal(row_proj, other["mask"]):
                    edges.append((i, j))
            elif side in ("left", "right"):
                if col_proj.shape == other["mask"].shape and np.array_equal(col_proj, other["mask"]):
                    edges.append((i, j))
    return sorted(set(edges))


def _pair_solutions(components: list[dict], limit: int = 2) -> list[list[tuple[int, int]]]:
    n = len(components)
    if n % 2 == 1:
        return []
    edges = _candidate_edges(components)
    incident: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for edge in edges:
        a, b = edge
        incident[a].append(edge)
        incident[b].append(edge)

    solutions: list[list[tuple[int, int]]] = []

    def backtrack(remaining: set[int], chosen: list[tuple[int, int]]) -> bool:
        if not remaining:
            solutions.append(sorted(chosen))
            return len(solutions) >= limit

        pick = min(
            remaining,
            key=lambda idx: sum(1 for edge in incident[idx] if edge[0] in remaining and edge[1] in remaining),
        )
        candidates = [edge for edge in incident[pick] if edge[0] in remaining and edge[1] in remaining]
        if not candidates:
            return False

        for edge in candidates:
            a, b = edge
            chosen.append(edge)
            stop = backtrack(remaining - {a, b}, chosen)
            chosen.pop()
            if stop:
                return True
        return False

    backtrack(set(range(n)), [])
    return solutions


def _solve_pairs(components: list[dict]) -> list[tuple[int, int]]:
    solutions = _pair_solutions(components, limit=2)
    assert len(solutions) == 1, "expected a unique projection pairing"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each shadow silhouette with the source shape that projects to it, then recolor the source by the shadow.

    Input structure:
    - The background is black.
    - Several monochrome 8-connected components appear in separated groups.
    - In each group, one component is an irregular source shape and the other is a smoother shadow silhouette on a cardinal side of it.

    Rule:
    - For every ordered component pair, use their relative placement to decide which projection to test.
    - A component above or below another must equal the other component's row-wise span projection: in each occupied row, fill from the leftmost to the rightmost source pixel.
    - A component left or right of another must equal the other component's column-wise span projection: in each occupied column, fill from the topmost to the bottommost source pixel.
    - Choose the unique one-to-one matching of all components under those projection tests, recolor each source shape to its matched shadow color, and erase the shadow components to black.

    Key insight:
    - The smaller-looking silhouettes are not arbitrary helpers; they are exact directional projections whose orientation is implied by whether they sit above/below or left/right of the source shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    components = _find_components(g, bg=BLACK)
    assert components, "expected at least one foreground component"

    pairs = _solve_pairs(components)
    out = g.copy()
    for source_idx, shadow_idx in pairs:
        shadow = components[shadow_idx]
        out[shadow["coords"][:, 0], shadow["coords"][:, 1]] = BLACK
    for source_idx, shadow_idx in pairs:
        source = components[source_idx]
        shadow = components[shadow_idx]
        out[source["coords"][:, 0], source["coords"][:, 1]] = int(shadow["color"])
    return out


def _make_vertical_masks() -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(LOCAL_SHAPE_RETRIES):
        height = random.randint(4, 7)
        current = random.randint(2, 5)
        ends: list[int] = []
        for _row in range(height):
            current = max(2, min(6, current + random.choice((-1, 0, 1))))
            ends.append(current)
        if len(set(ends)) == 1 and random.random() < 0.75:
            continue

        width = max(ends) + 1
        bridge_count = random.choice((1, 1, 2))
        bridge_rows = set(random.sample(range(1, height - 1), k=min(bridge_count, height - 2)))

        shadow = np.zeros((height, width), dtype=bool)
        for r, end in enumerate(ends):
            shadow[r, : end + 1] = True

        source = np.zeros_like(shadow)
        source[:, 0] = True
        for r, end in enumerate(ends):
            source[r, end] = True
        for r in bridge_rows:
            source[r, : ends[r] + 1] = True

        if int(source.sum()) >= int(shadow.sum()):
            continue
        if not _connected(source):
            continue
        if np.array_equal(source, shadow):
            continue
        if not np.array_equal(_project_rows(source), shadow):
            continue
        if np.array_equal(_project_cols(source), shadow):
            continue
        return source, shadow
    return None


def _make_horizontal_masks() -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(LOCAL_SHAPE_RETRIES):
        width = random.randint(4, 7)
        current = random.randint(2, 5)
        ends: list[int] = []
        for _col in range(width):
            current = max(2, min(6, current + random.choice((-1, 0, 1))))
            ends.append(current)
        if len(set(ends)) == 1 and random.random() < 0.75:
            continue

        height = max(ends) + 1
        bridge_count = random.choice((1, 1, 2))
        bridge_cols = set(random.sample(range(1, width - 1), k=min(bridge_count, width - 2)))

        shadow = np.zeros((height, width), dtype=bool)
        for c, end in enumerate(ends):
            shadow[: end + 1, c] = True

        source = np.zeros_like(shadow)
        source[0, :] = True
        for c, end in enumerate(ends):
            source[end, c] = True
        for c in bridge_cols:
            source[: ends[c] + 1, c] = True

        if int(source.sum()) >= int(shadow.sum()):
            continue
        if not _connected(source):
            continue
        if np.array_equal(source, shadow):
            continue
        if not np.array_equal(_project_cols(source), shadow):
            continue
        if np.array_equal(_project_rows(source), shadow):
            continue
        return source, shadow
    return None


def _make_pair_spec(kind: str, source_color: int, shadow_color: int) -> dict | None:
    if kind == "vertical":
        masks = _make_vertical_masks()
        side = random.choice(("above", "below"))
    else:
        masks = _make_horizontal_masks()
        side = random.choice(("left", "right"))
    if masks is None:
        return None
    source_mask, shadow_mask = masks
    shape_h, shape_w = source_mask.shape
    gap = random.randint(*PAIR_GAP_RANGE)

    if kind == "vertical":
        patch_h = shape_h * 2 + gap
        patch_w = shape_w
        inp = np.zeros((patch_h, patch_w), dtype=int)
        out = np.zeros((patch_h, patch_w), dtype=int)
        if side == "above":
            shadow_top, source_top = 0, shape_h + gap
        else:
            source_top, shadow_top = 0, shape_h + gap
        inp[source_top : source_top + shape_h, :] = np.where(source_mask, source_color, 0)
        inp[shadow_top : shadow_top + shape_h, :] = np.where(shadow_mask, shadow_color, 0)
        out[source_top : source_top + shape_h, :] = np.where(source_mask, shadow_color, 0)
    else:
        patch_h = shape_h
        patch_w = shape_w * 2 + gap
        inp = np.zeros((patch_h, patch_w), dtype=int)
        out = np.zeros((patch_h, patch_w), dtype=int)
        if side == "left":
            shadow_left, source_left = 0, shape_w + gap
        else:
            source_left, shadow_left = 0, shape_w + gap
        inp[:, source_left : source_left + shape_w] = np.where(source_mask, source_color, 0)
        inp[:, shadow_left : shadow_left + shape_w] = np.where(shadow_mask, shadow_color, 0)
        out[:, source_left : source_left + shape_w] = np.where(source_mask, shadow_color, 0)

    return {
        "kind": kind,
        "side": side,
        "input_patch": inp,
        "output_patch": out,
        "shadow_key": _mask_key(shadow_mask),
    }


def _patch_cells(patch: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    rows, cols = np.nonzero(patch)
    return [(int(top + r), int(left + c)) for r, c in zip(rows, cols, strict=False)]


def _can_place(cells: list[tuple[int, int]], blocked: set[tuple[int, int]], height: int, width: int) -> bool:
    return all(0 <= r < height and 0 <= c < width and (r, c) not in blocked for r, c in cells)


def _reserve(blocked: set[tuple[int, int]], cells: list[tuple[int, int]], height: int, width: int) -> None:
    for r, c in cells:
        for nr in range(r - MARGIN, r + MARGIN + 1):
            for nc in range(c - MARGIN, c + MARGIN + 1):
                if 0 <= nr < height and 0 <= nc < width:
                    blocked.add((nr, nc))


def _place_patch(target: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    view = target[top : top + ph, left : left + pw]
    mask = patch != BLACK
    view[mask] = patch[mask]
    target[top : top + ph, left : left + pw] = view


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    max_pairs = 4 if min(height, width) >= 24 else 3
    pair_count = random.randint(PAIR_COUNT_RANGE[0], min(PAIR_COUNT_RANGE[1], max_pairs))
    vertical_count = random.randint(1, pair_count - 1)
    kinds = ["vertical"] * vertical_count + ["horizontal"] * (pair_count - vertical_count)
    random.shuffle(kinds)

    colors = random.sample(PALETTE, 2 * pair_count)
    specs: list[dict] = []
    shadow_keys: set[tuple[tuple[int, int], bytes]] = set()
    for idx, kind in enumerate(kinds):
        spec = _make_pair_spec(kind, colors[2 * idx], colors[2 * idx + 1])
        if spec is None or spec["shadow_key"] in shadow_keys:
            return None
        shadow_keys.add(spec["shadow_key"])
        specs.append(spec)

    inp = np.zeros((height, width), dtype=int)
    expected = np.zeros_like(inp)
    blocked: set[tuple[int, int]] = set()

    for spec in specs:
        patch = spec["input_patch"]
        ph, pw = patch.shape
        if ph > height - 2 or pw > width - 2:
            return None

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(1, height - ph - 1)
            left = random.randint(1, width - pw - 1)
            cells = _patch_cells(patch, top, left)
            if not _can_place(cells, blocked, height, width):
                continue
            _place_patch(inp, patch, top, left)
            _place_patch(expected, spec["output_patch"], top, left)
            _reserve(blocked, cells, height, width)
            placed = True
            break
        if not placed:
            return None

    components = _find_components(inp, bg=BLACK)
    if len(components) != 2 * pair_count:
        return None

    solutions = _pair_solutions(components, limit=2)
    if len(solutions) != 1:
        return None

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shadow-projection input")
