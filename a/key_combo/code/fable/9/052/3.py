import random
from collections import deque

import numpy as np


BG = 0
PALETTE = list(range(1, 10))
GEN_RETRIES = 400
MIN_ROW_GAP = 5


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_by_color(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(pts, dtype=int)
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords), "area": len(pts)})
    return comps


def _frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _emitter_direction(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = shape
    hits = []
    if r == 0:
        hits.append((1, 0))
    if r == h - 1:
        hits.append((-1, 0))
    if c == 0:
        hits.append((0, 1))
    if c == w - 1:
        hits.append((0, -1))
    assert len(hits) == 1, "emitters must lie on exactly one border side"
    return hits[0]


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    emitters = []
    frames = []
    solids = []
    frame_ids = np.full(g.shape, -1, dtype=int)
    solid_ids = np.full(g.shape, -1, dtype=int)

    for comp in _components_by_color(g):
        color = int(comp["color"])
        coords = comp["coords"]
        area = int(comp["area"])
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1

        if area == 1 and (r0 in (0, g.shape[0] - 1) or c0 in (0, g.shape[1] - 1)):
            dr, dc = _emitter_direction(g.shape, r0, c0)
            emitters.append({"pos": (r0, c0), "dir": (dr, dc), "color": color})
            continue

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        mask = sub == color
        if np.all(mask):
            solid_id = len(solids)
            solid_ids[coords[:, 0], coords[:, 1]] = solid_id
            solids.append({"bbox": (r0, c0, r1, c1), "color": color, "coords": coords})
            continue

        if h >= 3 and w >= 3 and np.array_equal(mask, _frame_mask(h, w)):
            frame_id = len(frames)
            frame_ids[coords[:, 0], coords[:, 1]] = frame_id
            frames.append({"bbox": (r0, c0, r1, c1), "color": color, "coords": coords})
            continue

        raise AssertionError("all non-emitter objects must be solid rectangles or hollow frames")

    assert emitters, "expected at least one emitter"
    assert frames or solids, "expected at least one interior rectangle"
    return {
        "grid": g,
        "emitters": sorted(emitters, key=lambda e: (e["pos"][0], e["pos"][1])),
        "frames": frames,
        "solids": solids,
        "frame_ids": frame_ids,
        "solid_ids": solid_ids,
    }


def _solve_variant(
    grid: np.ndarray,
    *,
    block_frames: bool = False,
    recolor_frames: bool = True,
    pass_solids: bool = False,
) -> tuple[np.ndarray, list[dict], dict]:
    parsed = _parse_grid(grid)
    g = parsed["grid"]
    out = g.copy()
    used = {}
    infos = []

    for emitter in parsed["emitters"]:
        r, c = emitter["pos"]
        dr, dc = emitter["dir"]
        r += dr
        c += dc
        current = int(emitter["color"])
        seen_frames = []
        seen_frame_ids = set()
        path = []
        hit_solid = False

        while _in_bounds(g.shape, r, c):
            frame_id = int(parsed["frame_ids"][r, c])
            solid_id = int(parsed["solid_ids"][r, c])

            if frame_id != -1:
                if block_frames:
                    break
                if recolor_frames:
                    current = int(parsed["frames"][frame_id]["color"])
                if frame_id not in seen_frame_ids:
                    seen_frame_ids.add(frame_id)
                    seen_frames.append(frame_id)
                r += dr
                c += dc
                continue

            if solid_id != -1:
                if pass_solids:
                    r += dr
                    c += dc
                    continue
                hit_solid = True
                break

            if int(g[r, c]) != BG:
                break

            prev = used.get((r, c))
            assert prev in (None, current), "ray colors must not conflict on the same cell"
            used[(r, c)] = current
            out[r, c] = current
            path.append((r, c))
            r += dr
            c += dc

        infos.append(
            {
                "emitter_pos": emitter["pos"],
                "emitter_color": int(emitter["color"]),
                "final_color": current,
                "frame_ids": tuple(seen_frames),
                "hit_solid": hit_solid,
                "path": tuple(path),
            }
        )

    return out, infos, parsed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend edge rays through hollow color-filters while solid blocks stop them.

    Input structure:
    - The background is black.
    - Single-pixel emitters sit on the grid border and point straight inward.
    - Interior foreground objects are monochromatic axis-aligned rectangles: either
      one-cell-thick hollow frames or solid filled blocks.

    Rule:
    - Each emitter launches a ray in the inward normal direction, initially painting
      background cells with the emitter's own color.
    - When the ray reaches a hollow frame wall, the wall stays unchanged, the ray
      passes through it, and its current paint color switches to that frame's color.
    - When the ray reaches a solid rectangle, it stops immediately before that cell;
      if it reaches the far border instead, it simply ends there.

    Key insight:
    - Rectangles with similar outlines play opposite roles: hollow ones are color
      filters that transmit the ray, while solid ones are blockers that truncate it.
    """
    out, _infos, _parsed = _solve_variant(np.asarray(grid, dtype=int))
    return out


def _sample_rows(height: int, count: int) -> list[int] | None:
    choices = list(range(3, height - 3))
    for _ in range(200):
        rows = sorted(random.sample(choices, count))
        if all(b - a >= MIN_ROW_GAP for a, b in zip(rows, rows[1:])):
            return rows
    return None


def _lane_tokens(lane_count: int) -> list[list[str]]:
    first = ["frame", "solid"]
    if random.random() < 0.5:
        first.insert(1, "frame")
    if lane_count == 2:
        last = ["frame"]
        if random.random() < 0.5:
            last.append("frame")
        return [first, last]
    middle = ["solid"]
    last = ["frame"]
    if random.random() < 0.5:
        last.append("frame")
    return [first, middle, last]


def _solid_offsets() -> tuple[int, int]:
    return random.choice([(0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1)])


def _build_lane_specs(rows: list[int], lane_count: int) -> list[dict]:
    specs = []
    for row, tokens in zip(rows, _lane_tokens(lane_count)):
        parts = []
        for token in tokens:
            if token == "frame":
                up = random.randint(1, 2)
                down = random.randint(1, 2)
                width = random.randint(4, 7)
            else:
                up, down = _solid_offsets()
                width = random.randint(3, 6)
            parts.append({"kind": token, "up": up, "down": down, "width": width})
        specs.append(
            {
                "row": row,
                "side": random.choice(["left", "right"]),
                "parts": parts,
                "pre_gap": random.randint(1, 4),
                "between": [random.randint(2, 4) for _ in range(len(parts) - 1)],
                "tail_gap": random.randint(2, 5),
            }
        )
    return specs


def _required_width(specs: list[dict]) -> int:
    need = 0
    for spec in specs:
        total = spec["pre_gap"] + spec["tail_gap"] + sum(p["width"] for p in spec["parts"]) + sum(spec["between"])
        need = max(need, total + 1)
    return need


def _lane_columns(width: int, spec: dict) -> list[tuple[int, int]]:
    col = spec["pre_gap"] + 1
    boxes = []
    for i, part in enumerate(spec["parts"]):
        left = col
        right = left + part["width"] - 1
        boxes.append((left, right))
        if i < len(spec["between"]):
            col = right + 1 + spec["between"][i]
    if spec["side"] == "left":
        return boxes
    return [(width - 1 - right, width - 1 - left) for left, right in boxes]


def _place_rectangles(grid: np.ndarray, specs: list[dict], colors: list[int]) -> list[dict]:
    objects = []
    color_iter = iter(colors)
    for spec in specs:
        cols = _lane_columns(grid.shape[1], spec)
        for part, (c0, c1) in zip(spec["parts"], cols):
            color = next(color_iter)
            r0 = spec["row"] - part["up"]
            r1 = spec["row"] + part["down"]
            if part["kind"] == "frame":
                grid[r0, c0 : c1 + 1] = color
                grid[r1, c0 : c1 + 1] = color
                grid[r0 : r1 + 1, c0] = color
                grid[r0 : r1 + 1, c1] = color
            else:
                grid[r0 : r1 + 1, c0 : c1 + 1] = color
            objects.append({"kind": part["kind"], "bbox": (r0, c0, r1, c1), "color": color})
    return objects


def _make_canonical() -> np.ndarray | None:
    lane_count = random.choice([2, 3])
    min_h = 15 if lane_count == 2 else 20
    height = random.randint(min_h, 24)
    rows = _sample_rows(height, lane_count)
    if rows is None:
        return None

    specs = _build_lane_specs(rows, lane_count)
    width_min = _required_width(specs)
    if width_min > 29:
        return None
    width = random.randint(width_min, min(29, width_min + 4))

    object_count = sum(len(spec["parts"]) for spec in specs)
    color_count = lane_count + object_count
    if color_count > len(PALETTE):
        return None
    colors = random.sample(PALETTE, color_count)
    emitter_colors = colors[:lane_count]
    object_colors = colors[lane_count:]

    grid = np.zeros((height, width), dtype=int)
    _place_rectangles(grid, specs, object_colors)
    for emitter_color, spec in zip(emitter_colors, specs):
        if spec["side"] == "left":
            grid[spec["row"], 0] = emitter_color
        else:
            grid[spec["row"], width - 1] = emitter_color
    return grid


def _verify_semantics(grid: np.ndarray) -> bool:
    correct, infos, parsed = _solve_variant(grid)
    if not np.array_equal(correct, main(grid)):
        return False

    painted = [cell for info in infos for cell in info["path"]]
    if len(painted) != len(set(painted)):
        return False
    if not any(info["frame_ids"] for info in infos):
        return False
    if not any(info["hit_solid"] for info in infos):
        return False
    if not any(info["frame_ids"] and not info["hit_solid"] for info in infos):
        return False
    if len(infos) == 3 and not any((not info["frame_ids"]) and info["hit_solid"] for info in infos):
        return False
    if not any(info["final_color"] != info["emitter_color"] for info in infos):
        return False
    if not any(len(info["frame_ids"]) >= 2 for info in infos):
        multi_gate_possible = sum(1 for info in infos if info["frame_ids"]) >= 2
        if not multi_gate_possible:
            return False

    wrong1, _infos1, _parsed1 = _solve_variant(grid, block_frames=True)
    wrong2, _infos2, _parsed2 = _solve_variant(grid, recolor_frames=False)
    wrong3, _infos3, _parsed3 = _solve_variant(grid, pass_solids=True)
    if np.array_equal(correct, wrong1):
        return False
    if np.array_equal(correct, wrong2):
        return False
    if np.array_equal(correct, wrong3):
        return False
    if np.array_equal(correct, grid):
        return False

    # Ensure each parsed rectangle is a genuine rectangle under the same coordinate system as the solver.
    for frame in parsed["frames"]:
        r0, c0, r1, c1 = frame["bbox"]
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.array_equal(sub == frame["color"], _frame_mask(r1 - r0 + 1, c1 - c0 + 1)):
            return False
    for solid in parsed["solids"]:
        r0, c0, r1, c1 = solid["bbox"]
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == solid["color"]):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _make_canonical()
        if grid is None:
            continue
        rotated = np.rot90(grid, random.randrange(4)).copy()
        if _verify_semantics(rotated):
            return rotated
    raise RuntimeError("failed to generate a valid filter-gate ARC input")
