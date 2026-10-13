from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
INPUT_COLOR = 5
COLOR_CYCLE = (2, 8, 3)
LABELS = (
    frozenset((0,)),
    frozenset((1,)),
    frozenset((2,)),
    frozenset((0, 1)),
    frozenset((0, 2)),
    frozenset((1, 2)),
)
MAX_GEN_ATTEMPTS = 400


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(coords)
    return comps


def _extract_block(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    h, w = patch.shape

    if h == 3 and w in (7, 9):
        if not np.all(patch[:, 1:-1] == INPUT_COLOR):
            raise AssertionError("horizontal block core must be a solid 3x(5|7) rectangle")
        left = frozenset(int(r) for r in np.where(patch[:, 0] == INPUT_COLOR)[0].tolist())
        right = frozenset(int(r) for r in np.where(patch[:, -1] == INPUT_COLOR)[0].tolist())
        if len(left) not in (1, 2) or len(right) not in (1, 2):
            raise AssertionError("horizontal block interfaces must have size 1 or 2")
        if np.any((patch[:, 0] != BG) & (patch[:, 0] != INPUT_COLOR)) or np.any((patch[:, -1] != BG) & (patch[:, -1] != INPUT_COLOR)):
            raise AssertionError("unexpected colors on interface columns")
        return {
            "coords": coords,
            "bbox": (r0, c0, r1, c1),
            "orient": "h",
            "core_bbox": (r0, c0 + 1, r1, c1 - 1),
            "lead": left,
            "trail": right,
            "length": w - 2,
        }

    if w == 3 and h in (7, 9):
        if not np.all(patch[1:-1, :] == INPUT_COLOR):
            raise AssertionError("vertical block core must be a solid (5|7)x3 rectangle")
        top = frozenset(int(c) for c in np.where(patch[0, :] == INPUT_COLOR)[0].tolist())
        bottom = frozenset(int(c) for c in np.where(patch[-1, :] == INPUT_COLOR)[0].tolist())
        if len(top) not in (1, 2) or len(bottom) not in (1, 2):
            raise AssertionError("vertical block interfaces must have size 1 or 2")
        if np.any((patch[0, :] != BG) & (patch[0, :] != INPUT_COLOR)) or np.any((patch[-1, :] != BG) & (patch[-1, :] != INPUT_COLOR)):
            raise AssertionError("unexpected colors on interface rows")
        return {
            "coords": coords,
            "bbox": (r0, c0, r1, c1),
            "orient": "v",
            "core_bbox": (r0 + 1, c0, r1 - 1, c1),
            "lead": top,
            "trail": bottom,
            "length": h - 2,
        }

    raise AssertionError("each component must be a 3x(5|7) or (5|7)x3 rectangle with one-cell interface stubs")


def _extract_blocks(grid: np.ndarray) -> list[dict]:
    comps = _find_components(np.asarray(grid, dtype=int), INPUT_COLOR)
    blocks = [_extract_block(grid, comp) for comp in comps]
    if not 4 <= len(blocks) <= 6:
        raise AssertionError("expected 4 to 6 blocks")
    orientations = {block["orient"] for block in blocks}
    if len(orientations) != 1:
        raise AssertionError("all blocks must share the same chain orientation")
    return blocks


def _chain_order(blocks: list[dict]) -> list[int]:
    n = len(blocks)
    succs: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if blocks[i]["trail"] == blocks[j]["lead"]:
                succs[i].append(j)

    found: list[list[int]] = []

    def dfs(path: list[int], used: set[int]) -> None:
        if len(found) > 1:
            return
        if len(path) == n:
            found.append(path.copy())
            return
        for nxt in succs[path[-1]]:
            if nxt in used:
                continue
            used.add(nxt)
            path.append(nxt)
            dfs(path, used)
            path.pop()
            used.remove(nxt)

    for start in range(n):
        dfs([start], {start})
        if len(found) > 1:
            break

    if len(found) != 1:
        raise AssertionError("expected exactly one valid chain ordering")
    return found[0]


def _paint_transformed(out: np.ndarray, block: dict, chain_index: int) -> None:
    color = COLOR_CYCLE[(chain_index - 1) % len(COLOR_CYCLE)]
    for r, c in block["coords"]:
        out[r, c] = color

    r0, c0, r1, c1 = block["core_bbox"]
    start_offset = 1 if chain_index % 2 == 1 else 2
    if block["orient"] == "h":
        center_r = r0 + 1
        for offset in range(start_offset, block["length"], 2):
            out[center_r, c0 + offset] = BG
    else:
        center_c = c0 + 1
        for offset in range(start_offset, block["length"], 2):
            out[r0 + offset, center_c] = BG


def _expected_from_blocks(shape: tuple[int, int], blocks: list[dict], order: list[int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for chain_index, block_idx in enumerate(order, start=1):
        _paint_transformed(out, blocks[block_idx], chain_index)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor interface-marked 3-wide rectangles by chain order, then punch phase-shifted holes through their center lines.

    Input structure:
    - The background is black.
    - There are 4 to 6 disconnected grey objects, each formed by a solid 3x(5 or 7) rectangle plus one-cell interface stubs on its two short sides.
    - All objects share one chaining orientation: either all are horizontal with left/right interfaces, or all are vertical with top/bottom interfaces.

    Rule:
    - Read each object's interface-offset set on its leading and trailing short sides.
    - Connect two objects exactly when the predecessor's trailing interface offsets equal the successor's leading interface offsets, then recover the unique end-to-end chain.
    - Recolor chain positions 1,2,3,... by repeating red(2), teal(8), green(3) over every grey pixel of each object, including interface stubs.
    - On the solid rectangle core only, punch black holes along the middle row or column: odd chain positions remove the 2nd, 4th, 6th... core pixels from the top/left end; even chain positions remove the 3rd, 5th, 7th... core pixels.

    Key insight:
    - The same recovered chain index drives two independent periodic effects: color uses index mod 3, while hole phase uses index mod 2.
    """
    g = np.asarray(grid, dtype=int)
    blocks = _extract_blocks(g)
    order = _chain_order(blocks)
    return _expected_from_blocks(g.shape, blocks, order)


def _make_patch(orient: str, length: int, lead: frozenset[int], trail: frozenset[int]) -> np.ndarray:
    if orient == "h":
        patch = np.zeros((3, length + 2), dtype=int)
        patch[:, 1:-1] = INPUT_COLOR
        for offset in lead:
            patch[int(offset), 0] = INPUT_COLOR
        for offset in trail:
            patch[int(offset), -1] = INPUT_COLOR
        return patch

    patch = np.zeros((length + 2, 3), dtype=int)
    patch[1:-1, :] = INPUT_COLOR
    for offset in lead:
        patch[0, int(offset)] = INPUT_COLOR
    for offset in trail:
        patch[-1, int(offset)] = INPUT_COLOR
    return patch


def _placement_ok(placed: list[tuple[int, int, int, int]], bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + 2 < pr0 or pr1 + 2 < r0 or c1 + 2 < pc0 or pc1 + 2 < c0):
            return False
    return True


def _spatial_shortcut(order_positions: list[tuple[int, int]], orient: str) -> bool:
    if orient == "h":
        key = lambda idx: (order_positions[idx][1], order_positions[idx][0])
    else:
        key = lambda idx: (order_positions[idx][0], order_positions[idx][1])
    sorted_ids = sorted(range(len(order_positions)), key=key)
    return sorted_ids == list(range(len(order_positions))) or sorted_ids == list(range(len(order_positions) - 1, -1, -1))


def _generate_spec() -> tuple[str, list[dict]]:
    n = random.choice((4, 5, 6))
    orient = random.choice(("h", "v"))
    vertex_count = (n + 2) // 2 + 1
    labels = random.sample(list(LABELS), vertex_count)
    vertices = [min(i // 2, vertex_count - 1) for i in range(n + 1)]

    lengths = [5, 7]
    while len(lengths) < n:
        lengths.append(random.choice((5, 7)))
    random.shuffle(lengths)

    specs: list[dict] = []
    for i in range(n):
        lead = labels[vertices[i]]
        trail = labels[vertices[i + 1]]
        specs.append(
            {
                "orient": orient,
                "length": lengths[i],
                "lead": lead,
                "trail": trail,
            }
        )
    return orient, specs


def _place_specs(orient: str, specs: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    n = len(specs)
    min_h = 18 if orient == "h" else max(18, n * 3 + 4)
    min_w = max(20, n * 4 + 2) if orient == "h" else 18
    h = random.randint(min_h, min(30, min_h + 6))
    w = random.randint(min_w, min(30, min_w + 8))
    if max(h, w) > 30:
        return None

    grid = np.zeros((h, w), dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []
    order_positions = [(-1, -1)] * n

    place_order = list(range(n))
    random.shuffle(place_order)
    for idx in place_order:
        spec = specs[idx]
        patch = _make_patch(orient, spec["length"], spec["lead"], spec["trail"])
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r in range(h - ph + 1):
            for c in range(w - pw + 1):
                bbox = (r, c, r + ph - 1, c + pw - 1)
                if _placement_ok(placed_boxes, bbox):
                    candidates.append((r, c))
        if not candidates:
            return None
        r, c = random.choice(candidates)
        grid[r : r + ph, c : c + pw] = np.maximum(grid[r : r + ph, c : c + pw], patch)
        placed_boxes.append((r, c, r + ph - 1, c + pw - 1))
        order_positions[idx] = (r + ph // 2, c + pw // 2)
        spec["bbox"] = (r, c, r + ph - 1, c + pw - 1)

    if _spatial_shortcut(order_positions, orient):
        return None

    return grid, specs


def _expected_from_specs(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for chain_index, spec in enumerate(specs, start=1):
        patch = _make_patch(spec["orient"], spec["length"], spec["lead"], spec["trail"])
        r0, c0, r1, c1 = spec["bbox"]
        color = COLOR_CYCLE[(chain_index - 1) % len(COLOR_CYCLE)]
        patch_colored = np.where(patch == INPUT_COLOR, color, BG)
        out[r0 : r1 + 1, c0 : c1 + 1] = np.maximum(out[r0 : r1 + 1, c0 : c1 + 1], patch_colored)
        if spec["orient"] == "h":
            center_r = r0 + 1
            start_offset = 1 if chain_index % 2 == 1 else 2
            for offset in range(start_offset, spec["length"], 2):
                out[center_r, c0 + 1 + offset] = BG
        else:
            center_c = c0 + 1
            start_offset = 1 if chain_index % 2 == 1 else 2
            for offset in range(start_offset, spec["length"], 2):
                out[r0 + 1 + offset, center_c] = BG
    return out


def _validate_generated(grid: np.ndarray, specs: list[dict]) -> bool:
    blocks = _extract_blocks(grid)
    order = _chain_order(blocks)
    expected = _expected_from_specs(grid.shape, specs)
    if len(order) != len(specs):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    for block in blocks:
        r0, c0, r1, c1 = block["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if block["orient"] == "h":
            if not np.all(patch[:, 1:-1] == INPUT_COLOR):
                return False
        else:
            if not np.all(patch[1:-1, :] == INPUT_COLOR):
                return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        orient, specs = _generate_spec()
        placed = _place_specs(orient, [dict(spec) for spec in specs])
        if placed is None:
            continue
        grid, placed_specs = placed
        if _validate_generated(grid, placed_specs):
            return grid
    raise RuntimeError("failed to generate a valid input")
