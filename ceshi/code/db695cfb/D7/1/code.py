# Variant of db695cfb
# Dimension: D7 — 交互触发对象 (obstacle)
# Change: 品红像素——位于主对角线路径上的品红像素触发垂直对角延伸... → 色块触发（非单像素）

import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the D7 variant
# ----------------------------------
# a) Concrete behavior:
#    - Blue endpoint pairs still define the main 45-degree diagonals exactly as in the seed task.
#    - The trigger is no longer an isolated magenta pixel. It is now a whole 4-connected magenta
#      block that intersects the future blue diagonal.
#    - Two trigger block families are used:
#      1. domino blocks (area 2, either 1x2 or 2x1) -> perpendicular rays extend exactly 2 cells.
#      2. 2x2 square blocks (area 4) -> perpendicular rays extend exactly 4 cells.
#    - A block activates only if the completed blue diagonal passes through one or more cells of
#      that block. Rays are emitted from those touched cells, not from off-diagonal cells of the block.
#    Example:
#      original: a single magenta pixel on the diagonal produces a full boundary-to-boundary
#      perpendicular line.
#      variant: a 2-cell domino on the diagonal produces only short 2-cell stubs, while a 2x2
#      square produces longer 4-cell stubs. The object size now changes the output.
#
# b) Complexity calibration:
#    - Only two trigger categories are used, so the mapping stays learnable from a few examples.
#    - The categories are visually distinct: 2 cells vs 2x2 square, and their output lengths
#      are also distinct: 2 vs 4.
#    - Off-diagonal magenta blocks of the same shapes are allowed as distractors, so the solver
#      must use both object shape and diagonal interaction.
#
# c) Implicit-assumption audit of the seed solver:
#    - The original draw loop assumed every non-blue cell on the diagonal is an independent trigger.
#      That is no longer valid: different cells can belong to one block whose size determines the
#      ray length. The new solver groups non-blue cells into 4-connected same-color components first.
#    - The original boundary-extending ray loop assumed a single fixed ray length policy.
#      That also changes: ray length is now component-dependent.
#    - The original line-color detection via the longest diagonal pair still works because trigger
#      blocks are small (domino or 2x2), while blue endpoint pairs remain at distance >= 3.
#
# d) Placement legality:
#    - Trigger blocks must stay disjoint and may not touch each other orthogonally, otherwise they
#      would merge into one larger magenta component and break the intended classification.
#    - Rays may only paint background cells; generator placement rejects any block whose rays would
#      collide with existing input pixels or previously reserved output cells.
#    - Square blocks are placed so that exactly two of their cells lie on the blue diagonal; domino
#      blocks are placed so that exactly one cell lies on the blue diagonal.
#
# e) Visual observability:
#    - The new rule is visible in the output because the magenta growth is no longer always
#      boundary-reaching. Small blocks produce short stubs, large blocks produce longer ones.
#    - The block shape itself is visible in the input, so the length rule can be inferred without
#      hidden numeric metadata.
#
# f) Visual cleanliness:
#    - Segment count and trigger count are kept modest.
#    - Full intended stub lengths must fit in-bounds, avoiding ambiguous truncation near borders.
#    - Magenta components are kept separated so each trigger object is easy to parse.
#
# g) Color reasoning:
#    - No hidden ordering or color-id logic is used. Color roles remain the same as the seed task:
#      blue builds the diagonal, magenta provides interaction objects, background is the majority color.
#
# h) New solver assumptions guaranteed by the generator:
#    - Any activating magenta component is either a domino or a 2x2 square.
#    - Different magenta blocks are 4-disconnected.
#    - A square trigger touches the diagonal in two cells; a domino trigger touches it in one cell.


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _classify_trigger_component(cells: set[tuple[int, int]]) -> int:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    h = max(rows) - min(rows) + 1
    w = max(cols) - min(cols) + 1
    area = len(cells)
    if area == 2 and ((h == 1 and w == 2) or (h == 2 and w == 1)):
        return 2
    if area == 4 and h == 2 and w == 2:
        return 4
    return 0


def _collect_trigger_components(
    grid: np.ndarray, bg: int, line_color: int
) -> tuple[list[dict[str, object]], dict[tuple[int, int], int]]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    components: list[dict[str, object]] = []
    cell_to_component: dict[tuple[int, int], int] = {}

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (bg, line_color) or (r, c) in seen:
                continue

            stack = [(r, c)]
            seen.add((r, c))
            cells: set[tuple[int, int]] = set()
            while stack:
                cr, cc = stack.pop()
                cells.add((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(grid[nr, nc]) == color:
                        seen.add((nr, nc))
                        stack.append((nr, nc))

            ray_len = _classify_trigger_component(cells)
            idx = len(components)
            components.append({"cells": cells, "color": color, "length": ray_len})
            if ray_len > 0:
                for cell in cells:
                    cell_to_component[cell] = idx

    return components, cell_to_component


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete blue diagonals, then emit length-limited perpendicular stubs from magenta trigger blocks.

    Input structure:
    - The background is the most common color. Blue pixels still appear as endpoint pairs on 45-degree diagonals.
    - Magenta objects are now small 4-connected blocks, and only blocks intersected by a completed blue diagonal can activate.

    Rule:
    - Connect each blue endpoint pair along its diagonal.
    - For each activated magenta block, emit perpendicular diagonal rays from the block cells touched by the blue diagonal. Domino blocks extend 2 cells per direction; 2x2 square blocks extend 4 cells per direction.

    Key insight:
    - The interaction is block-based rather than pixel-based: identify the whole magenta object first, then use its size/shape class to decide the ray length.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    bg = get_bg(grid)

    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    best = None  # (dist, color)
    for col, pts in by_color.items():
        if len(pts) < 2:
            continue
        for i in range(len(pts)):
            r0, c0 = pts[i]
            for j in range(i + 1, len(pts)):
                r1, c1 = pts[j]
                dr = r1 - r0
                dc = c1 - c0
                if dr == 0 or abs(dr) != abs(dc):
                    continue
                dist = abs(dr)
                cand = (dist, col)
                if best is None or cand > best:
                    best = cand

    assert best is not None, "at least one diagonal-aligned same-color pair must exist"

    _, line_color = best
    pts = by_color[line_color]
    components, cell_to_component = _collect_trigger_components(grid, bg, line_color)

    triggers: dict[int, dict[str, object]] = {}

    groups_a: dict[int, list[tuple[int, int]]] = {}
    groups_b: dict[int, list[tuple[int, int]]] = {}
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = _diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = [(dr, -dc), (-dr, dc)]
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
                continue
            if v == line_color:
                continue
            comp_idx = cell_to_component.get((r, c))
            if comp_idx is None:
                continue
            comp = components[comp_idx]
            entry = triggers.setdefault(
                comp_idx,
                {
                    "cells": set(),
                    "dirs": set(),
                    "length": int(comp["length"]),
                    "color": int(comp["color"]),
                },
            )
            entry["cells"].add((r, c))
            entry["dirs"].update(perp_dirs)

    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for info in triggers.values():
        ray_len = int(info["length"])
        color = int(info["color"])
        for r0, c0 in info["cells"]:
            for ddr, ddc in info["dirs"]:
                for step in range(1, ray_len + 1):
                    r = r0 + step * ddr
                    c = c0 + step * ddc
                    if not (0 <= r < h and 0 <= c < w):
                        break
                    if int(out[r, c]) == bg:
                        out[r, c] = color

    return out


# --------------- generator ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (10, 25)
_W_RANGE = (10, 21)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SEGMENT_COUNT = (1, 3)
_SEG_LEN_RANGE = (4, 16)
_TRIGGER_BLOCKS_PER_SEG = (0, 1)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 1)
_OFF_DIAG_BLOCKS = (0, 2)

# === CFG: retry limits ===
_SEGMENT_PLACE_RETRIES = 120
_BLOCK_PLACE_RETRIES = 80
_DISTRACTOR_RETRIES = 80
_GEN_RETRIES = 400


def _compute_fixed_ray_cells(
    touched: list[tuple[int, int]], perp_dirs: list[tuple[int, int]], ray_len: int, H: int, W: int
) -> set[tuple[int, int]] | None:
    cells: set[tuple[int, int]] = set()
    for tr, tc in touched:
        for ddr, ddc in perp_dirs:
            for step in range(1, ray_len + 1):
                r = tr + step * ddr
                c = tc + step * ddc
                if not (0 <= r < H and 0 <= c < W):
                    return None
                cells.add((r, c))
    return cells


def _touches_existing_magenta(
    new_cells: set[tuple[int, int]], existing_magenta: set[tuple[int, int]]
) -> bool:
    for r, c in new_cells:
        if (r, c) in existing_magenta:
            return True
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in existing_magenta:
                return True
    return False


def _build_expected(inp: np.ndarray, bg: int, segments: list[dict[str, object]]) -> np.ndarray:
    H, W = inp.shape
    expected = inp.copy()
    for seg in segments:
        for r, c in seg["path"]:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for block in seg["blocks"]:
            for tr, tc in block["touched"]:
                for ddr, ddc in seg["perp_dirs"]:
                    for step in range(1, block["length"] + 1):
                        r = tr + step * ddr
                        c = tc + step * ddc
                        if 0 <= r < H and 0 <= c < W and expected[r, c] == bg:
                            expected[r, c] = _TRIGGER_COLOR
    return expected


def _try_place_segment(
    slope_type: str, seg_len: int, H: int, W: int, occupied: set[tuple[int, int]], drawn: set[tuple[int, int]]
) -> tuple[list[tuple[int, int]], int] | None:
    for _ in range(_SEGMENT_PLACE_RETRIES):
        if slope_type == "backslash":
            if H - 1 - seg_len < 0 or W - 1 - seg_len < 0:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(0, W - 1 - seg_len)
            dc = 1
        else:
            if H - 1 - seg_len < 0 or seg_len > W - 1:
                return None
            r0 = random.randint(0, H - 1 - seg_len)
            c0 = random.randint(seg_len, W - 1)
            dc = -1

        path = [(r0 + k, c0 + k * dc) for k in range(seg_len + 1)]
        if any(rc in occupied or rc in drawn for rc in path):
            continue
        return path, dc
    return None


def _enumerate_trigger_options(path: list[tuple[int, int]], dc: int, H: int, W: int) -> list[dict[str, object]]:
    options: list[dict[str, object]] = []
    path_set = set(path)
    n = len(path)

    for k in range(1, n - 1):
        r, c = path[k]
        for extra in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            er, ec = extra
            if not (0 <= er < H and 0 <= ec < W):
                continue
            if extra in path_set:
                continue
            options.append(
                {
                    "kind": "domino",
                    "cells": {(r, c), extra},
                    "touched": [(r, c)],
                    "length": 2,
                }
            )

    for k in range(1, n - 2):
        r, c = path[k]
        if dc == 1:
            cells = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
            touched = [(r, c), (r + 1, c + 1)]
        else:
            cells = {(r, c - 1), (r, c), (r + 1, c - 1), (r + 1, c)}
            touched = [(r, c), (r + 1, c - 1)]
        if all(0 <= rr < H and 0 <= cc < W for rr, cc in cells):
            options.append({"kind": "square", "cells": cells, "touched": touched, "length": 4})

    return options


def _try_place_offdiag_block(
    H: int,
    W: int,
    path_cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    drawn: set[tuple[int, int]],
    magenta_cells: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    for _ in range(_DISTRACTOR_RETRIES):
        kind = random.choice(["domino", "square"])
        if kind == "square":
            r = random.randint(0, H - 2)
            c = random.randint(0, W - 2)
            cells = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
        else:
            r = random.randint(0, H - 1)
            c = random.randint(0, W - 1)
            if random.choice([True, False]):
                if c + 1 >= W:
                    continue
                cells = {(r, c), (r, c + 1)}
            else:
                if r + 1 >= H:
                    continue
                cells = {(r, c), (r + 1, c)}

        if cells & path_cells or cells & occupied or cells & drawn:
            continue
        if _touches_existing_magenta(cells, magenta_cells):
            continue
        return cells
    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    n_segments = random.randint(*_SEGMENT_COUNT)
    max_seg_len = min(H - 1, W - 1, _SEG_LEN_RANGE[1])
    if max_seg_len < _SEG_LEN_RANGE[0]:
        return None

    occupied: set[tuple[int, int]] = set()
    drawn: set[tuple[int, int]] = set()
    magenta_cells: set[tuple[int, int]] = set()
    path_cells: set[tuple[int, int]] = set()
    segments: list[dict[str, object]] = []
    total_trigger_blocks = 0

    for _ in range(n_segments):
        slope_type = random.choice(["backslash", "slash"])
        seg_len = random.randint(_SEG_LEN_RANGE[0], min(max_seg_len, _SEG_LEN_RANGE[1]))
        result = _try_place_segment(slope_type, seg_len, H, W, occupied, drawn)
        if result is None:
            continue
        path, dc = result
        perp_dirs = [(1, -dc), (-1, dc)]

        blocks: list[dict[str, object]] = []
        local_magenta: set[tuple[int, int]] = set()
        local_rays: set[tuple[int, int]] = set()
        desired_blocks = random.randint(*_TRIGGER_BLOCKS_PER_SEG)
        options = _enumerate_trigger_options(path, dc, H, W)
        random.shuffle(options)

        for opt in options:
            if len(blocks) >= desired_blocks:
                break
            cells = set(opt["cells"])
            touched = list(opt["touched"])
            ray_cells = _compute_fixed_ray_cells(touched, perp_dirs, int(opt["length"]), H, W)
            if ray_cells is None:
                continue
            if cells & occupied or cells & drawn or cells & local_magenta or cells & local_rays:
                continue
            if ray_cells & occupied or ray_cells & drawn or ray_cells & local_magenta or ray_cells & local_rays:
                continue
            if ray_cells & cells:
                continue
            if ray_cells & (set(path) - set(touched)):
                continue
            if _touches_existing_magenta(cells, magenta_cells | local_magenta):
                continue

            blocks.append(
                {
                    "kind": str(opt["kind"]),
                    "cells": sorted(cells),
                    "touched": touched,
                    "length": int(opt["length"]),
                }
            )
            local_magenta.update(cells)
            local_rays.update(ray_cells)

        blue_fill = set(path[1:-1]) - local_magenta
        if blue_fill & local_rays:
            continue

        segments.append(
            {
                "endpoints": (path[0], path[-1]),
                "path": path,
                "perp_dirs": perp_dirs,
                "blocks": blocks,
            }
        )
        occupied.update([path[0], path[-1]])
        for block in blocks:
            occupied.update(block["cells"])
            magenta_cells.update(block["cells"])
        drawn.update(blue_fill)
        drawn.update(local_rays)
        path_cells.update(path)
        total_trigger_blocks += len(blocks)

    if not segments or total_trigger_blocks == 0:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for seg in segments:
        ep0, ep1 = seg["endpoints"]
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for block in seg["blocks"]:
            for cell in block["cells"]:
                inp[cell] = _TRIGGER_COLOR

    all_line_positions = list(zip(*np.where(inp == _LINE_COLOR)))
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r = random.randint(0, H - 1)
            c = random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in drawn or (r, c) in path_cells:
                continue
            if any(abs(r - lr) == abs(c - lc) and r != lr for lr, lc in all_line_positions):
                continue
            inp[r, c] = _LINE_COLOR
            occupied.add((r, c))
            all_line_positions.append((r, c))
            break

    for _ in range(random.randint(*_OFF_DIAG_BLOCKS)):
        cells = _try_place_offdiag_block(H, W, path_cells, occupied, drawn, magenta_cells)
        if cells is None:
            continue
        for cell in cells:
            inp[cell] = _TRIGGER_COLOR
            occupied.add(cell)
            magenta_cells.add(cell)

    try:
        result = main(inp)
    except Exception:
        return None

    expected = _build_expected(inp, bg, segments)
    if np.array_equal(result, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
