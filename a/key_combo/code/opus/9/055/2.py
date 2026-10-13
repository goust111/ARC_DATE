import numpy as np


BG = 0
OBSTACLE = 5
RING_COLORS = (1, 2, 3, 4)
_RNG = np.random.default_rng()

MAX_GENERATION_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _ring_coords(r0: int, c0: int, size: int) -> np.ndarray:
    pts = []
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        pts.append((r, c1))
    return np.array(sorted(set(pts)), dtype=int)


def _extract_rings(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = [int(c) for c in np.unique(g) if int(c) not in (BG, OBSTACLE)]
    assert 3 <= len(colors) <= 4, "expected 3-4 colored rings"

    rings = []
    for color in colors:
        parts = _components(g == color)
        assert len(parts) == 1, "each ring color must form one component"
        coords = parts[0]
        r0, c0, r1, c1 = _bbox(coords)
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        assert side_h == side_w and side_h >= 3, "ring bbox must be a square"
        expected = _ring_coords(r0, c0, side_h)
        assert len(expected) == len(coords), "ring must be one pixel thick"
        assert set(map(tuple, coords)) == set(map(tuple, expected)), "component must match its square border"
        rings.append(
            {
                "color": color,
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "side": side_h,
            }
        )

    rings.sort(key=lambda item: item["side"], reverse=True)
    for i in range(len(rings) - 1):
        outer = rings[i]["bbox"]
        inner = rings[i + 1]["bbox"]
        assert outer[0] + 1 < inner[0]
        assert outer[1] + 1 < inner[1]
        assert inner[2] < outer[2] - 1
        assert inner[3] < outer[3] - 1
    return rings


def _extract_obstacles(grid: np.ndarray) -> tuple[np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    coords = np.argwhere(g == OBSTACLE)
    assert 3 <= len(coords) <= 5, "expected 3-5 obstacle pixels"
    rows = {int(r) for r, _ in coords}
    assert len(rows) == 1, "obstacles must lie on one row"
    row = next(iter(rows))
    cols = sorted(int(c) for _, c in coords)
    assert any(cols[i + 1] - cols[i] > 1 for i in range(len(cols) - 1)), "obstacles must be discontinuous"
    return coords, row


def _drop_one(
    coords: np.ndarray,
    color: int,
    max_drop: int,
    occupied: np.ndarray,
) -> tuple[np.ndarray, int, str]:
    h, _ = occupied.shape
    current = coords.copy()
    moved = 0
    cause = "max"

    for _ in range(max_drop):
        shifted = current.copy()
        shifted[:, 0] += 1
        if np.any(shifted[:, 0] >= h):
            cause = "boundary"
            break
        blockers = occupied[shifted[:, 0], shifted[:, 1]]
        if np.any(blockers != BG):
            cause = "obstacle" if np.any(blockers == OBSTACLE) else "ring"
            break
        current = shifted
        moved += 1

    occupied[current[:, 0], current[:, 1]] = color
    return current, moved, cause


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    rings = _extract_rings(g)
    out = np.full_like(g, BG)
    out[g == OBSTACLE] = OBSTACLE
    trace = []

    n = len(rings)
    for depth, ring in enumerate(rings, start=1):
        settled, moved, cause = _drop_one(
            ring["coords"],
            ring["color"],
            n - depth + 1,
            out,
        )
        trace.append(
            {
                "color": ring["color"],
                "depth": depth,
                "max_drop": n - depth + 1,
                "actual_drop": moved,
                "cause": cause,
            }
        )
        out[settled[:, 0], settled[:, 1]] = ring["color"]
    return out, trace


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop nested square rings downward by amounts determined by nesting depth.

    Input structure:
    - A black background contains 3-4 single-color square rings, each one pixel thick and strictly nested.
    - A discontinuous row of grey obstacle pixels sits below the rings on the same grid.
    - Each non-grey foreground color belongs to exactly one ring component.

    Rule:
    - Order the rings from outermost to innermost by their nesting depth.
    - If there are N rings, the depth-d ring may move down at most (N - d + 1) rows.
    - Process the rings from outermost to innermost; move each ring downward one row at a time until its next step would hit a grey obstacle, a previously settled ring, or the grid boundary.
    - Keep obstacle pixels and all colors unchanged apart from the rings' final positions.

    Key insight:
    - Nesting depth is the hidden motion parameter: outer rings have more downward travel budget, and their settled positions become new obstacles for deeper rings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    _extract_obstacles(g)
    out, _ = _simulate(g)
    return out


def _build_nested_rings(
    n_layers: int,
    gaps: list[int],
    inner_side: int,
    top: int,
    left: int,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    sizes = [inner_side]
    for gap in reversed(gaps):
        sizes.append(sizes[-1] + 2 * (gap + 1))
    sizes = list(reversed(sizes))

    placements = []
    r0 = top
    c0 = left
    for idx, side in enumerate(sizes):
        placements.append((r0, c0, side))
        if idx < len(gaps):
            step = gaps[idx] + 1
            r0 += step
            c0 += step
    return np.array(sizes, dtype=int), placements


def _sample_obstacles(
    width: int,
    outer_bbox: tuple[int, int, int, int],
    second_bbox: tuple[int, int, int, int] | None,
) -> list[int] | None:
    outer_left, outer_right = outer_bbox[1], outer_bbox[3]
    candidate_hits = list(range(outer_left, outer_right + 1))
    if second_bbox is not None:
        inner_left, inner_right = second_bbox[1], second_bbox[3]
        candidate_hits = [c for c in candidate_hits if c < inner_left or c > inner_right]
    if not candidate_hits:
        return None

    obstacle_count = int(_RNG.integers(3, 6))
    chosen: list[int] = []

    def add_col(pool: list[int]) -> bool:
        pool = [c for c in pool if all(abs(c - prev) > 1 for prev in chosen)]
        if not pool:
            return False
        chosen.append(int(_RNG.choice(pool)))
        return True

    if not add_col(candidate_hits):
        return None
    if _RNG.random() < 0.5:
        add_col(candidate_hits)

    outside = [c for c in range(width) if c < outer_left or c > outer_right]
    outer_only = [c for c in candidate_hits if c not in chosen]
    pools = [outside, outer_only, list(range(width))]
    while len(chosen) < obstacle_count:
        added = False
        for pool in pools:
            if add_col(pool):
                added = True
                break
        if not added:
            return None

    chosen.sort()
    if not any(chosen[i + 1] - chosen[i] > 1 for i in range(len(chosen) - 1)):
        return None
    return chosen


def _make_candidate() -> tuple[np.ndarray, str] | None:
    n_layers = 4 if _RNG.random() < 0.65 else 3
    mode = "outer_only"
    if n_layers == 4 and _RNG.random() < 0.55:
        mode = "chain_block"

    gaps = [int(_RNG.integers(1, 3)) for _ in range(n_layers - 1)]
    if mode == "chain_block":
        gaps[0] = 1

    inner_side = int(_RNG.integers(3, 5))
    sizes, placements = _build_nested_rings(n_layers, gaps, inner_side, top=0, left=0)
    outer_side = int(sizes[0])
    if outer_side > 19:
        return None

    outer_drop = 1 if mode == "chain_block" else int(_RNG.integers(1, n_layers))

    top = int(_RNG.integers(1, 4))
    left_margin = int(_RNG.integers(2, 5))
    right_margin = int(_RNG.integers(2, 5))
    left = left_margin
    outer_bottom = top + outer_side - 1
    obstacle_row = outer_bottom + outer_drop + 1
    extra_below = int(_RNG.integers(2, 5))
    height = obstacle_row + 1 + extra_below
    width = outer_side + left_margin + right_margin
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    _, placements = _build_nested_rings(n_layers, gaps, inner_side, top=top, left=left)
    for color, (r0, c0, side) in zip(RING_COLORS, placements):
        coords = _ring_coords(r0, c0, side)
        grid[coords[:, 0], coords[:, 1]] = color

    outer = placements[0]
    second = placements[1] if len(placements) > 1 else None
    outer_bbox = (outer[0], outer[1], outer[0] + outer[2] - 1, outer[1] + outer[2] - 1)
    second_bbox = None
    if second is not None:
        second_bbox = (second[0], second[1], second[0] + second[2] - 1, second[1] + second[2] - 1)

    obstacle_cols = _sample_obstacles(width, outer_bbox, second_bbox)
    if obstacle_cols is None:
        return None
    grid[obstacle_row, obstacle_cols] = OBSTACLE
    return grid, mode


def _validate_case(grid: np.ndarray, mode: str) -> bool:
    g = np.asarray(grid, dtype=int)
    rings = _extract_rings(g)
    obstacle_coords, obstacle_row = _extract_obstacles(g)
    outer_bottom = rings[0]["bbox"][2]
    blank_rows = obstacle_row - outer_bottom - 1
    if not (1 <= blank_rows <= 3):
        return False

    out, trace = _simulate(g)
    if np.array_equal(out, g):
        return False

    actual_drops = [step["actual_drop"] for step in trace]
    max_drops = [step["max_drop"] for step in trace]
    causes = [step["cause"] for step in trace]

    if trace[0]["cause"] != "obstacle":
        return False
    if trace[0]["actual_drop"] >= trace[0]["max_drop"]:
        return False
    if not any(drop > 0 for drop in actual_drops):
        return False
    if len(set(actual_drops)) < 2:
        return False
    if not any(a == b for a, b in zip(actual_drops[1:], max_drops[1:])):
        return False

    if mode == "chain_block":
        if len(trace) < 2 or trace[1]["cause"] != "ring":
            return False
    else:
        if any(step["cause"] == "ring" for step in trace[1:]) and len(rings) == 3:
            return False

    colors = set(np.unique(out)) - {BG}
    if len(colors) < len(rings) + 1:
        return False

    obstacle_cols = {int(c) for _, c in obstacle_coords}
    second_left = rings[1]["bbox"][1]
    second_right = rings[1]["bbox"][3]
    if not any(c < second_left or c > second_right for c in obstacle_cols):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        attempt = _make_candidate()
        if attempt is None:
            continue
        grid, mode = attempt
        if _validate_case(grid, mode):
            return grid
    raise RuntimeError("failed to generate a valid nested-depth gravity input")
