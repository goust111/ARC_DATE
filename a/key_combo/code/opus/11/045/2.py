from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5
FG_COLORS = (BLUE, RED)

_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

_COUNT_OPTIONS = [(2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3)]
_MODE_WEIGHTS = [("any", 2), ("v_obstacle", 1), ("h_obstacle", 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in _DIR4:
        yield r + dr, c + dc


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    start = tuple(map(int, cells[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0): int(r1) + 1, int(c0): int(c1) + 1].copy()


def _classify_mask(mask: np.ndarray) -> str:
    vertical = bool(np.array_equal(mask, np.fliplr(mask)))
    horizontal = bool(np.array_equal(mask, np.flipud(mask)))
    assert vertical ^ horizontal, "object must have exactly one axis symmetry"
    return "vertical" if vertical else "horizontal"


def _component_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    arr = np.array(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    for r, c in coords:
        mask[int(r - r0), int(c - c0)] = True
    return mask


def _find_objects(grid: np.ndarray, sep_row: int) -> list[dict]:
    upper = np.asarray(grid[:sep_row, :], dtype=int)
    seen = np.zeros_like(upper, dtype=bool)
    objects: list[dict] = []
    h, w = upper.shape
    for r in range(h):
        for c in range(w):
            color = int(upper[r, c])
            if color in (BG, GREY) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(upper[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            mask = _component_mask(coords)
            kind = _classify_mask(mask)
            objects.append(
                {
                    "color": color,
                    "coords": [tuple(map(int, pt)) for pt in coords],
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                    "kind": kind,
                }
            )
    return objects


def _can_shift(board: np.ndarray, coords: list[tuple[int, int]], dr: int, dc: int) -> bool:
    h, w = board.shape
    for r, c in coords:
        nr, nc = int(r + dr), int(c + dc)
        if not (0 <= nr < h and 0 <= nc < w):
            return False
        if int(board[nr, nc]) != BG:
            return False
    return True


def _shift_coords(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in coords]


def _paint(board: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        board[int(r), int(c)] = int(color)


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.array(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _solve(grid: np.ndarray, *, with_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_candidates = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    assert len(sep_candidates) == 1, "expected exactly one full gray separator row"
    sep = int(sep_candidates[0])
    assert np.all(g[sep + 1 :, :] == BG), "encoding area must start empty"

    objects = _find_objects(g, sep)
    kinds = [obj["kind"] for obj in objects]
    n_vertical = sum(kind == "vertical" for kind in kinds)
    n_horizontal = sum(kind == "horizontal" for kind in kinds)
    assert 2 <= n_vertical <= 4 and 2 <= n_horizontal <= 4 and n_vertical != n_horizontal

    board = g.copy()
    movement_meta: list[dict] = []

    vertical_objects = sorted(
        [obj for obj in objects if obj["kind"] == "vertical"],
        key=lambda obj: (-obj["bbox"][2], obj["bbox"][1], obj["bbox"][0]),
    )
    for obj in vertical_objects:
        start_coords = list(obj["coords"])
        start_bbox = obj["bbox"]
        for r, c in start_coords:
            board[r, c] = BG
        drop = 0
        while _can_shift(board, start_coords, 1, 0):
            start_coords = _shift_coords(start_coords, 1, 0)
            drop += 1
        _paint(board, start_coords, obj["color"])
        movement_meta.append(
            {
                "kind": "vertical",
                "start_bbox": start_bbox,
                "final_bbox": _bbox_from_coords(start_coords),
                "actual_shift": drop,
                "max_shift": sep - 1 - start_bbox[2],
            }
        )

    horizontal_objects = sorted(
        [obj for obj in objects if obj["kind"] == "horizontal"],
        key=lambda obj: (-obj["bbox"][1], obj["bbox"][0], obj["bbox"][2]),
    )
    for obj in horizontal_objects:
        start_coords = list(obj["coords"])
        start_bbox = obj["bbox"]
        for r, c in start_coords:
            board[r, c] = BG
        shift = 0
        while _can_shift(board, start_coords, 0, 1):
            start_coords = _shift_coords(start_coords, 0, 1)
            shift += 1
        _paint(board, start_coords, obj["color"])
        movement_meta.append(
            {
                "kind": "horizontal",
                "start_bbox": start_bbox,
                "final_bbox": _bbox_from_coords(start_coords),
                "actual_shift": shift,
                "max_shift": g.shape[1] - 1 - start_bbox[3],
            }
        )

    out = board
    out[sep + 1 :, :] = BG
    left_w = g.shape[1] // 2
    right_c0 = left_w
    for i in range(n_vertical):
        out[sep + 1 + i, i] = RED
    for i in range(n_horizontal):
        out[sep + 1 + i, right_c0 + i] = BLUE

    if with_meta:
        return out, {
            "sep": sep,
            "objects": objects,
            "n_vertical": n_vertical,
            "n_horizontal": n_horizontal,
            "movement": movement_meta,
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split symmetric objects by axis, move them in axis-dependent directions, and encode the two counts on diagonals below a gray separator.

    Input structure:
    - The grid has a black background with one full gray horizontal separator row.
    - Above the separator are several small single-color 4-connected objects in red or blue.
    - Each object's tight foreground mask has exactly one symmetry type: left-right mirror symmetry or up-down mirror symmetry.
    - The area below the separator starts empty and serves as a count-encoding zone.

    Rule:
    - Detect every upper object and classify its tight mask by symmetry.
    - Left-right symmetric objects fall downward until blocked by the separator or settled objects.
    - Up-down symmetric objects then slide rightward, processed from right to left, until blocked by the boundary or already moved objects.
    - Count the two symmetry classes and encode them below the separator: red on the left-half diagonal for the vertical-axis-symmetric count, blue on the right-half diagonal for the horizontal-axis-symmetric count.

    Key insight:
    - The same symmetry test controls both the motion branch and the final numerical encoding, so misclassifying an object breaks both its landing place and the diagonal counts.
    """
    return _solve(grid)


def _random_symmetric_mask(kind: str) -> np.ndarray | None:
    for _ in range(300):
        h = random.randint(2, 4)
        w = random.randint(2, 5)
        raw = np.zeros((h, w), dtype=bool)
        if kind == "vertical":
            half = (w + 1) // 2
            p = random.uniform(0.35, 0.7)
            for r in range(h):
                for c in range(half):
                    if random.random() < p:
                        raw[r, c] = True
                        raw[r, w - 1 - c] = True
        else:
            half = (h + 1) // 2
            p = random.uniform(0.35, 0.7)
            for r in range(half):
                for c in range(w):
                    if random.random() < p:
                        raw[r, c] = True
                        raw[h - 1 - r, c] = True
        if not raw.any():
            continue
        mask = _crop_mask(raw)
        area = int(mask.sum())
        if not (3 <= area <= 6):
            continue
        if not _is_connected(mask):
            continue
        vertical = bool(np.array_equal(mask, np.fliplr(mask)))
        horizontal = bool(np.array_equal(mask, np.flipud(mask)))
        if kind == "vertical" and not (vertical and not horizontal):
            continue
        if kind == "horizontal" and not (horizontal and not vertical):
            continue
        return mask
    return None


def _rel_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(mask)]


def _candidate_positions(mask: np.ndarray, blocked: np.ndarray, sep: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    rel = _rel_coords(mask)
    options: list[tuple[int, int]] = []
    for r0 in range(sep - h + 1):
        for c0 in range(blocked.shape[1] - w + 1):
            ok = True
            for rr, cc in rel:
                ar, ac = r0 + rr, c0 + cc
                if blocked[ar, ac]:
                    ok = False
                    break
            if ok:
                options.append((int(r0), int(c0)))
    return options


def _mark_blocked(blocked: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    for rr, cc in _rel_coords(mask):
        ar, ac = int(r0 + rr), int(c0 + cc)
        for dr, dc in _DIR8 + [(0, 0)]:
            nr, nc = ar + dr, ac + dc
            if 0 <= nr < blocked.shape[0] and 0 <= nc < blocked.shape[1]:
                blocked[nr, nc] = True


def _make_color_assignment(count: int) -> list[int]:
    colors = [random.choice(FG_COLORS) for _ in range(count)]
    if len(set(colors)) == 1:
        colors[random.randrange(count)] = RED if colors[0] == BLUE else BLUE
    return colors


def _choose_mode() -> str:
    modes = [mode for mode, weight in _MODE_WEIGHTS for _ in range(weight)]
    return random.choice(modes)


def _decode_count_diagonal(grid: np.ndarray, sep: int, start_col: int, color: int) -> int:
    n = 0
    while sep + 1 + n < grid.shape[0] and start_col + n < grid.shape[1] and int(grid[sep + 1 + n, start_col + n]) == color:
        n += 1
    return n


def _sample_valid(grid: np.ndarray, required_mode: str) -> bool:
    out, meta = _solve(grid, with_meta=True)
    sep = meta["sep"]
    assert np.all(grid[sep, :] == GREY)
    assert np.all(grid[sep + 1 :, :] == BG)

    left_w = grid.shape[1] // 2
    right_c0 = left_w
    lower = out[sep + 1 :, :].copy()
    n_vertical = _decode_count_diagonal(out, sep, 0, RED)
    n_horizontal = _decode_count_diagonal(out, sep, right_c0, BLUE)
    if n_vertical != meta["n_vertical"] or n_horizontal != meta["n_horizontal"]:
        return False
    for r in range(lower.shape[0]):
        for c in range(lower.shape[1]):
            global_r = sep + 1 + r
            global_c = c
            allowed = BG
            if global_c == r and r < meta["n_vertical"]:
                allowed = RED
            if global_c == right_c0 + r and r < meta["n_horizontal"]:
                allowed = BLUE
            if int(out[global_r, global_c]) != allowed:
                return False

    v_moves = [m for m in meta["movement"] if m["kind"] == "vertical"]
    h_moves = [m for m in meta["movement"] if m["kind"] == "horizontal"]
    if not any(m["actual_shift"] > 0 for m in v_moves):
        return False
    if not any(m["actual_shift"] > 0 for m in h_moves):
        return False

    v_obstacle = any(0 < m["actual_shift"] < m["max_shift"] for m in v_moves)
    h_obstacle = any(0 < m["actual_shift"] < m["max_shift"] for m in h_moves)
    if required_mode == "v_obstacle" and not v_obstacle:
        return False
    if required_mode == "h_obstacle" and not h_obstacle:
        return False
    return True


def _generate_one() -> np.ndarray | None:
    h = random.randint(15, 19)
    w = random.randint(15, 19)
    sep = h // 2
    if min(sep, h - sep - 1, w // 2, w - w // 2) < 4:
        return None

    n_vertical, n_horizontal = random.choice(_COUNT_OPTIONS)
    specs: list[dict] = []
    used_signatures: set[tuple[tuple[int, int], bytes]] = set()
    for kind, count in (("vertical", n_vertical), ("horizontal", n_horizontal)):
        for _ in range(count):
            mask = None
            for _try in range(200):
                candidate = _random_symmetric_mask(kind)
                if candidate is None:
                    continue
                sig = (candidate.shape, candidate.astype(np.uint8).tobytes())
                if sig in used_signatures:
                    continue
                used_signatures.add(sig)
                mask = candidate
                break
            if mask is None:
                return None
            specs.append({"kind": kind, "mask": mask})

    random.shuffle(specs)
    colors = _make_color_assignment(len(specs))
    for spec, color in zip(specs, colors):
        spec["color"] = color

    blocked = np.zeros((sep, w), dtype=bool)
    placements: list[dict] = []
    for spec in sorted(specs, key=lambda item: (-int(item["mask"].sum()), item["mask"].shape[1], item["mask"].shape[0])):
        options = _candidate_positions(spec["mask"], blocked, sep)
        if not options:
            return None
        r0, c0 = random.choice(options)
        placements.append({**spec, "r0": int(r0), "c0": int(c0)})
        _mark_blocked(blocked, spec["mask"], int(r0), int(c0))

    grid = np.zeros((h, w), dtype=int)
    grid[sep, :] = GREY
    for placed in placements:
        for rr, cc in _rel_coords(placed["mask"]):
            grid[int(placed["r0"] + rr), int(placed["c0"] + cc)] = int(placed["color"])

    required_mode = _choose_mode()
    if not _sample_valid(grid, required_mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(1200):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
