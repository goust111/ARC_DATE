from __future__ import annotations

from collections import Counter, deque
from typing import Iterable

import numpy as np


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

ARROW_MASKS = {
    (-1, 0): np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    (1, 0): np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=bool),
    (0, -1): np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
    (0, 1): np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=bool),
}

GEN_ATTEMPTS = 400
GEN_SIZE_RANGE = (14, 24)
GEN_EXTRA_COUNT = (0, 3)
GEN_GAP_RANGE = (4, 8)
GEN_BORDER_BAND = 4
GEN_BG_COLORS = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_arrow_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _detect_arrows(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    arrows: list[dict] = []
    for comp in _find_arrow_components(g, bg):
        r0, c0, r1, c1 = comp["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if patch.shape != (3, 3):
            raise AssertionError("every foreground component must be a 3x3 arrow")
        if int(comp["coords"].shape[0]) != 5:
            raise AssertionError("every arrow must contain exactly 5 pixels")
        mask = patch == int(comp["color"])
        direction = None
        for cand_dir, cand_mask in ARROW_MASKS.items():
            if np.array_equal(mask, cand_mask):
                direction = cand_dir
                break
        if direction is None:
            raise AssertionError("unrecognized 3x3 arrow mask")
        center = (r0 + 1, c0 + 1)
        arrows.append(
            {
                "color": int(comp["color"]),
                "dir": direction,
                "bbox": (r0, c0, r1, c1),
                "center": center,
            }
        )
    arrows.sort(key=lambda a: (a["center"][0], a["center"][1], a["color"]))
    for idx, arrow in enumerate(arrows):
        arrow["id"] = idx
    return bg, arrows


def _build_bbox_owner(shape: tuple[int, int], arrows: list[dict]) -> np.ndarray:
    owner = np.full(shape, -1, dtype=int)
    for arrow in arrows:
        r0, c0, r1, c1 = arrow["bbox"]
        if np.any(owner[r0 : r1 + 1, c0 : c1 + 1] != -1):
            raise AssertionError("arrow bounding boxes must not overlap")
        owner[r0 : r1 + 1, c0 : c1 + 1] = int(arrow["id"])
    return owner


def _classify_hit(ray_dir: tuple[int, int], target_dir: tuple[int, int]) -> str:
    incoming = (-ray_dir[0], -ray_dir[1])
    if incoming == (-target_dir[0], -target_dir[1]):
        return "tail"
    if incoming == target_dir:
        return "front"
    return "side"


def _skip_past_bbox(bbox: tuple[int, int, int, int], ray_dir: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    dr, dc = ray_dir
    if dr == -1:
        return r0 - 1, c
    if dr == 1:
        return r1 + 1, c
    if dc == -1:
        return r, c0 - 1
    return r, c1 + 1


def _trace_ray(arrow: dict, arrows: list[dict], owner: np.ndarray) -> dict:
    h, w = owner.shape
    dr, dc = arrow["dir"]
    cr, cc = arrow["center"]
    r = cr + 2 * dr
    c = cc + 2 * dc
    path: list[tuple[int, int]] = []
    contacts: list[tuple[int, str]] = []

    while 0 <= r < h and 0 <= c < w:
        target = int(owner[r, c])
        if target < 0:
            path.append((r, c))
            r += dr
            c += dc
            continue
        if target == int(arrow["id"]):
            r, c = _skip_past_bbox(arrow["bbox"], (dr, dc), r, c)
            continue

        hit_kind = _classify_hit((dr, dc), arrows[target]["dir"])
        contacts.append((target, hit_kind))
        if hit_kind == "tail":
            r, c = _skip_past_bbox(arrows[target]["bbox"], (dr, dc), r, c)
            continue
        break

    side_target = contacts[-1][0] if contacts and contacts[-1][1] == "side" else None
    return {"path": path, "contacts": contacts, "side_target": side_target}


def _analyze_network(grid: np.ndarray) -> tuple[int, list[dict], list[dict], list[int]]:
    bg, arrows = _detect_arrows(grid)
    if not arrows:
        raise AssertionError("expected at least one arrow")
    owner = _build_bbox_owner(grid.shape, arrows)
    traces = [_trace_ray(arrow, arrows, owner) for arrow in arrows]

    incoming: set[int] = set()
    for trace in traces:
        for target, _kind in trace["contacts"]:
            incoming.add(int(target))

    roots = [idx for idx in range(len(arrows)) if idx not in incoming]
    if not roots:
        roots = list(range(len(arrows)))

    order: list[int] = []
    seen: set[int] = set()
    q: deque[int] = deque(sorted(roots))
    while q:
        idx = int(q.popleft())
        if idx in seen:
            continue
        seen.add(idx)
        order.append(idx)
        side_target = traces[idx]["side_target"]
        if side_target is not None and side_target not in seen:
            q.append(int(side_target))
    return bg, arrows, traces, order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate rays through a network of 3x3 arrows.

    Rays start one cell beyond an active arrow's tip. They paint background cells,
    pass straight through tail hits, stop on front hits, and side hits activate the
    struck arrow so its own ray continues the cascade. Arrow pixels themselves stay
    unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg, arrows, traces, order = _analyze_network(g)
    out = g.copy()

    for idx in order:
        color = int(arrows[idx]["color"])
        for r, c in traces[idx]["path"]:
            if int(g[r, c]) == bg and int(out[r, c]) == bg:
                out[r, c] = color

    return out


def _arrow_patch(direction: tuple[int, int], color: int, bg: int) -> np.ndarray:
    patch = np.full((3, 3), int(bg), dtype=int)
    patch[ARROW_MASKS[direction]] = int(color)
    return patch


def _rotate_point(pt: tuple[int, int], turns: int) -> tuple[int, int]:
    r, c = pt
    turns %= 4
    for _ in range(turns):
        r, c = c, -r
    return int(r), int(c)


def _rotate_dir(direction: tuple[int, int], turns: int) -> tuple[int, int]:
    return _rotate_point(direction, turns)


def _bbox_for_center(center: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = center
    return r - 1, c - 1, r + 1, c + 1


def _bboxes_with_gap(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int], gap: int = 1) -> bool:
    return not (
        b1[2] + gap < b2[0]
        or b2[2] + gap < b1[0]
        or b1[3] + gap < b2[1]
        or b2[3] + gap < b1[1]
    )


def _place_spec_ok(spec: dict, placed: list[dict], h: int, w: int) -> bool:
    r, c = spec["center"]
    if not (1 <= r <= h - 2 and 1 <= c <= w - 2):
        return False
    bbox = _bbox_for_center(spec["center"])
    for other in placed:
        if _bboxes_with_gap(bbox, _bbox_for_center(other["center"]), gap=1):
            return False
    return True


def _draw_specs(shape: tuple[int, int], bg: int, specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for spec in specs:
        r, c = spec["center"]
        grid[r - 1 : r + 2, c - 1 : c + 2] = _arrow_patch(spec["dir"], int(spec["color"]), int(bg))
    return grid


def _random_extra_spec(rng: np.random.Generator, h: int, w: int) -> dict:
    direction = DIRS[int(rng.integers(0, len(DIRS)))]
    if direction == (0, -1):
        center = (int(rng.integers(2, h - 2)), int(rng.integers(3, min(GEN_BORDER_BAND + 1, w - 2))))
    elif direction == (0, 1):
        lo = max(2, w - GEN_BORDER_BAND - 1)
        center = (int(rng.integers(2, h - 2)), int(rng.integers(lo, w - 2)))
    elif direction == (-1, 0):
        center = (int(rng.integers(3, min(GEN_BORDER_BAND + 1, h - 2))), int(rng.integers(2, w - 2)))
    else:
        lo = max(2, h - GEN_BORDER_BAND - 1)
        center = (int(rng.integers(lo, h - 2)), int(rng.integers(2, w - 2)))
    return {"center": center, "dir": direction}


def _validate_generated_grid(grid: np.ndarray) -> bool:
    try:
        bg, arrows, traces, order = _analyze_network(grid)
    except AssertionError:
        return False

    if len(arrows) < 4 or len(order) < 3:
        return False

    seen_tail = False
    seen_side = False
    owners: dict[tuple[int, int], int] = {}
    active = set(order)

    for idx in order:
        for target, kind in traces[idx]["contacts"]:
            if kind == "tail":
                seen_tail = True
            if kind == "side":
                seen_side = True
                if target not in active:
                    return False

        color = int(arrows[idx]["color"])
        for cell in traces[idx]["path"]:
            prev = owners.get(cell)
            if prev is None:
                owners[cell] = color
            elif prev != color:
                return False

    if not seen_tail or not seen_side:
        return False

    if len(owners) < 6:
        return False

    counts = Counter(int(v) for v in grid.ravel())
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return False

    return True


def _build_chain_specs(rng: np.random.Generator) -> list[dict]:
    gap1 = int(rng.integers(*GEN_GAP_RANGE))
    gap2 = int(rng.integers(*GEN_GAP_RANGE))
    gap3 = int(rng.integers(*GEN_GAP_RANGE))
    last_dir = (0, 1) if bool(rng.integers(0, 2)) else (0, -1)
    local = [
        {"center": (0, 0), "dir": (0, 1)},
        {"center": (0, gap1), "dir": (-1, 0)},
        {"center": (-gap2, gap1), "dir": (-1, 0)},
        {"center": (-gap2 - gap3, gap1), "dir": last_dir},
    ]

    turns = int(rng.integers(0, 4))
    rotated = [
        {
            "center": _rotate_point(spec["center"], turns),
            "dir": _rotate_dir(spec["dir"], turns),
        }
        for spec in local
    ]
    return rotated


def _translate_chain(rng: np.random.Generator, chain: list[dict], h: int, w: int) -> list[dict] | None:
    min_r = min(spec["center"][0] - 1 for spec in chain)
    max_r = max(spec["center"][0] + 1 for spec in chain)
    min_c = min(spec["center"][1] - 1 for spec in chain)
    max_c = max(spec["center"][1] + 1 for spec in chain)
    need_h = max_r - min_r + 1
    need_w = max_c - min_c + 1
    if need_h > h - 4 or need_w > w - 4:
        return None

    top = int(rng.integers(2, h - need_h - 1))
    left = int(rng.integers(2, w - need_w - 1))
    dr = top - min_r
    dc = left - min_c
    placed = []
    for spec in chain:
        placed.append({"center": (spec["center"][0] + dr, spec["center"][1] + dc), "dir": spec["dir"]})
    return placed


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GEN_SIZE_RANGE))
    w = int(rng.integers(*GEN_SIZE_RANGE))
    bg = int(rng.choice(GEN_BG_COLORS))

    chain = _build_chain_specs(rng)
    placed = _translate_chain(rng, chain, h, w)
    if placed is None:
        return None

    extras_target = int(rng.integers(*GEN_EXTRA_COUNT))
    extra_attempts = 0
    while len(placed) < 4 + extras_target and extra_attempts < 100:
        extra_attempts += 1
        cand = _random_extra_spec(rng, h, w)
        if _place_spec_ok(cand, placed, h, w):
            placed.append(cand)

    colors = list(set(range(10)) - {bg})
    if len(placed) > len(colors):
        return None
    rng.shuffle(colors)
    for spec, color in zip(placed, colors):
        spec["color"] = int(color)

    grid = _draw_specs((h, w), bg, placed)
    if not _validate_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-ray input")
