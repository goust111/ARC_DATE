from __future__ import annotations

import numpy as np


BG = 0
BRIDGE = 5
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_ATTEMPTS = 500
PAIR_ATTEMPTS = 250
ISOLATE_ATTEMPTS = 250
HALO = 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _label_objects(grid: np.ndarray) -> tuple[list[dict], np.ndarray, dict[int, int]]:
    h, w = grid.shape
    owner = np.full((h, w), -1, dtype=int)
    components: list[dict] = []
    counts: dict[int, int] = {}
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, BRIDGE):
            continue
        for coords in _component_coords(grid == color):
            idx = len(components)
            owner[coords[:, 0], coords[:, 1]] = idx
            rows = coords[:, 0]
            cols = coords[:, 1]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (
                        int(rows.min()),
                        int(cols.min()),
                        int(rows.max()),
                        int(cols.max()),
                    ),
                }
            )
            counts[color] = counts.get(color, 0) + 1
    return components, owner, counts


def _line_signature(coords: np.ndarray) -> tuple[str, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if int(rows.min()) == int(rows.max()):
        row = int(rows[0])
        sorted_cols = np.sort(cols)
        assert np.all(np.diff(sorted_cols) == 1), "horizontal bridge must be contiguous"
        return "horizontal", row, int(sorted_cols[0]), int(sorted_cols[-1])
    if int(cols.min()) == int(cols.max()):
        col = int(cols[0])
        sorted_rows = np.sort(rows)
        assert np.all(np.diff(sorted_rows) == 1), "vertical bridge must be contiguous"
        return "vertical", col, int(sorted_rows[0]), int(sorted_rows[-1])
    raise AssertionError("gray bridge must be a straight line")


def _band_spans(width: int) -> list[tuple[int, int]]:
    if width % 2 == 1:
        half = width // 2
        return [(half, half)]
    half = width // 2
    return [(half, half - 1), (half - 1, half)]


def _band_rect(
    orientation: str,
    fixed: int,
    start: int,
    end: int,
    lower: int,
    upper: int,
) -> tuple[int, int, int, int]:
    if orientation == "horizontal":
        return fixed - lower, start, fixed + upper, end
    return start, fixed - lower, end, fixed + upper


def _clip_rect(rect: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    h, w = shape
    r0, c0, r1, c1 = rect
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(h - 1, r1)
    cc1 = min(w - 1, c1)
    if rr0 > rr1 or cc0 > cc1:
        return None
    return rr0, cc0, rr1, cc1


def _band_candidates(
    grid: np.ndarray,
    orientation: str,
    fixed: int,
    start: int,
    end: int,
    width: int,
    color: int,
) -> list[dict]:
    h, w = grid.shape
    candidates: list[dict] = []
    for index, (lower, upper) in enumerate(_band_spans(width)):
        rect = _band_rect(orientation, fixed, start, end, lower, upper)
        r0, c0, r1, c1 = rect
        total_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        clipped = _clip_rect(rect, grid.shape)
        in_bounds_area = 0
        penalty = 0
        if clipped is None:
            penalty = 1000 * total_area
        else:
            rr0, cc0, rr1, cc1 = clipped
            in_bounds_area = (rr1 - rr0 + 1) * (cc1 - cc0 + 1)
            penalty = 1000 * (total_area - in_bounds_area)
            sub = grid[rr0 : rr1 + 1, cc0 : cc1 + 1]
            foreign = (sub != BG) & (sub != BRIDGE) & (sub != color)
            penalty += 100 * int(foreign.sum())

        contact = 0
        if orientation == "horizontal":
            left = start - 1
            right = end + 1
            for row in range(r0, r1 + 1):
                if 0 <= row < h and 0 <= left < w and int(grid[row, left]) == color:
                    contact += 1
                if 0 <= row < h and 0 <= right < w and int(grid[row, right]) == color:
                    contact += 1
        else:
            top = start - 1
            bottom = end + 1
            for col in range(c0, c1 + 1):
                if 0 <= top < h and 0 <= col < w and int(grid[top, col]) == color:
                    contact += 1
                if 0 <= bottom < h and 0 <= col < w and int(grid[bottom, col]) == color:
                    contact += 1

        candidates.append(
            {
                "rect": rect,
                "penalty": penalty,
                "contact": contact,
                "index": index,
                "in_bounds_area": in_bounds_area,
            }
        )

    candidates.sort(key=lambda item: (item["penalty"], -item["contact"], item["index"]))
    return candidates


def _paint_rect(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    clipped = _clip_rect(rect, grid.shape)
    if clipped is None:
        return
    r0, c0, r1, c1 = clipped
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each grey same-color bridge with a thicker band whose width is that color's total object count.

    Input structure:
    - The background is black, and every non-grey foreground object is a small solid rectangle.
    - Some pairs of same-color rectangles are joined by a one-pixel-thick grey horizontal or vertical bridge.
    - Each bridge is a single straight 4-connected line segment, while each color appears as 2-4 separate objects overall.

    Rule:
    - Find every grey bridge and determine which two colored objects it 4-touches at its endpoints.
    - Count how many separate objects of that color exist anywhere in the grid; call that count W.
    - Replace the bridge by a W-pixel-thick rectangular band of that same color, expanding perpendicular to the bridge direction.
    - Cover the original grey bridge pixels with the band and leave the colored objects themselves unchanged.

    Key insight:
    - The grey line identifies which same-color objects should be paired, but the replacement thickness comes from a different clue: the global number of objects of that color.
    """
    g = np.asarray(grid, dtype=int)
    components, owner, counts = _label_objects(g)
    out = g.copy()

    for coords in _component_coords(g == BRIDGE):
        orientation, fixed, start, end = _line_signature(coords)
        neighbor_ids: set[int] = set()
        for r, c in coords:
            for dr, dc in NEIGHBORS4:
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                    comp_id = int(owner[nr, nc])
                    if comp_id >= 0:
                        neighbor_ids.add(comp_id)

        assert len(neighbor_ids) == 2, "each bridge must touch exactly two objects"
        colors = {int(components[idx]["color"]) for idx in neighbor_ids}
        assert len(colors) == 1, "a bridge must connect same-color objects"
        color = next(iter(colors))
        width = int(counts[color])

        candidates = _band_candidates(g, orientation, fixed, start, end, width, color)
        _paint_rect(out, candidates[0]["rect"], color)

    return out


def _rect_clear(grid: np.ndarray, reserved: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    clipped = _clip_rect(rect, grid.shape)
    if clipped is None or clipped != rect:
        return False
    r0, c0, r1, c1 = rect
    return np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BG) and not np.any(
        reserved[r0 : r1 + 1, c0 : c1 + 1]
    )


def _mark_reserved(reserved: np.ndarray, rect: tuple[int, int, int, int], pad: int = HALO) -> None:
    h, w = reserved.shape
    r0, c0, r1, c1 = rect
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _rand_dims(rng: np.random.Generator, h_range: tuple[int, int], w_range: tuple[int, int]) -> tuple[int, int]:
    return (
        int(rng.integers(h_range[0], h_range[1] + 1)),
        int(rng.integers(w_range[0], w_range[1] + 1)),
    )


def _place_pair(
    rng: np.random.Generator,
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    width: int,
    orientation: str,
) -> bool:
    h, w = grid.shape
    for _ in range(PAIR_ATTEMPTS):
        obj_h1, obj_w1 = _rand_dims(rng, (2, 4), (2, 4))
        obj_h2, obj_w2 = _rand_dims(rng, (2, 4), (2, 4))
        bridge_len = int(rng.integers(2, 7))

        if orientation == "horizontal":
            bridge_row = int(rng.integers(0, h))
            top1 = bridge_row - int(rng.integers(0, obj_h1))
            top2 = bridge_row - int(rng.integers(0, obj_h2))
            if not (0 <= top1 and top1 + obj_h1 <= h and 0 <= top2 and top2 + obj_h2 <= h):
                continue
            total_w = obj_w1 + bridge_len + obj_w2
            if total_w > w:
                continue
            left1 = int(rng.integers(0, w - total_w + 1))
            left2 = left1 + obj_w1 + bridge_len
            rect1 = (top1, left1, top1 + obj_h1 - 1, left1 + obj_w1 - 1)
            rect2 = (top2, left2, top2 + obj_h2 - 1, left2 + obj_w2 - 1)
            bridge_rect = (bridge_row, left1 + obj_w1, bridge_row, left2 - 1)
            fixed = bridge_row
            start = left1 + obj_w1
            end = left2 - 1
        else:
            bridge_col = int(rng.integers(0, w))
            left1 = bridge_col - int(rng.integers(0, obj_w1))
            left2 = bridge_col - int(rng.integers(0, obj_w2))
            if not (0 <= left1 and left1 + obj_w1 <= w and 0 <= left2 and left2 + obj_w2 <= w):
                continue
            total_h = obj_h1 + bridge_len + obj_h2
            if total_h > h:
                continue
            top1 = int(rng.integers(0, h - total_h + 1))
            top2 = top1 + obj_h1 + bridge_len
            rect1 = (top1, left1, top1 + obj_h1 - 1, left1 + obj_w1 - 1)
            rect2 = (top2, left2, top2 + obj_h2 - 1, left2 + obj_w2 - 1)
            bridge_rect = (top1 + obj_h1, bridge_col, top2 - 1, bridge_col)
            fixed = bridge_col
            start = top1 + obj_h1
            end = top2 - 1

        if not (_rect_clear(grid, reserved, rect1) and _rect_clear(grid, reserved, rect2) and _rect_clear(grid, reserved, bridge_rect)):
            continue

        trial = grid.copy()
        _paint_rect(trial, rect1, color)
        _paint_rect(trial, rect2, color)
        _paint_rect(trial, bridge_rect, BRIDGE)
        candidates = _band_candidates(trial, orientation, fixed, start, end, width, color)
        best = candidates[0]
        if best["penalty"] != 0:
            continue
        if len(candidates) > 1:
            second = candidates[1]
            if best["penalty"] == second["penalty"] and best["contact"] == second["contact"]:
                continue
        band_rect = best["rect"]
        if _clip_rect(band_rect, grid.shape) != band_rect:
            continue
        if np.any(reserved[band_rect[0] : band_rect[2] + 1, band_rect[1] : band_rect[3] + 1]):
            continue

        _paint_rect(grid, rect1, color)
        _paint_rect(grid, rect2, color)
        _paint_rect(grid, bridge_rect, BRIDGE)
        for rect in (rect1, rect2, bridge_rect, band_rect):
            _mark_reserved(reserved, rect)
        return True
    return False


def _place_isolated(
    rng: np.random.Generator,
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
) -> bool:
    h, w = grid.shape
    for _ in range(ISOLATE_ATTEMPTS):
        obj_h, obj_w = _rand_dims(rng, (1, 3), (1, 4))
        top = int(rng.integers(0, h - obj_h + 1))
        left = int(rng.integers(0, w - obj_w + 1))
        rect = (top, left, top + obj_h - 1, left + obj_w - 1)
        if not _rect_clear(grid, reserved, rect):
            continue
        _paint_rect(grid, rect, color)
        _mark_reserved(reserved, rect)
        return True
    return False


def _sample_group_counts(rng: np.random.Generator, group_count: int) -> list[int]:
    base = [2, 3, 4]
    rng.shuffle(base)
    counts = base[: min(group_count, 3)]
    while len(counts) < group_count:
        counts.append(int(rng.integers(2, 5)))
    rng.shuffle(counts)
    return counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    group_count = int(rng.choice([2, 3, 4], p=[0.35, 0.45, 0.20]))
    counts = _sample_group_counts(rng, group_count)

    plans: list[dict] = []
    total_pairs = 0
    for count in counts:
        pair_count = 1 if count < 4 else int(rng.choice([1, 2]))
        isolate_count = count - 2 * pair_count
        total_pairs += pair_count
        plans.append({"count": count, "pair_count": pair_count, "isolate_count": isolate_count})

    colors = list(rng.choice(COLORS, size=group_count, replace=False))
    base_side = min(30, 12 + 3 * group_count + 2 * total_pairs)
    if base_side > 30:
        return None
    h = int(rng.integers(max(18, base_side - 3), 31))
    w = int(rng.integers(max(18, base_side - 3), 31))
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    orientations = ["horizontal", "vertical"] * (total_pairs // 2)
    if total_pairs % 2 == 1:
        orientations.append(str(rng.choice(["horizontal", "vertical"])))
    rng.shuffle(orientations)
    orientation_index = 0

    for plan, color in zip(plans, colors):
        for _ in range(plan["pair_count"]):
            orientation = orientations[orientation_index]
            orientation_index += 1
            if not _place_pair(rng, grid, reserved, int(color), int(plan["count"]), orientation):
                return None
        for _ in range(plan["isolate_count"]):
            if not _place_isolated(rng, grid, reserved, int(color)):
                return None

    components, _owner, observed_counts = _label_objects(grid)
    if len(components) != sum(counts):
        return None
    if sorted(observed_counts.values()) != sorted(counts):
        return None
    if len(_component_coords(grid == BRIDGE)) != total_pairs:
        return None

    out = main(grid)
    if out.shape != grid.shape or np.any(out == BRIDGE):
        return None
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
