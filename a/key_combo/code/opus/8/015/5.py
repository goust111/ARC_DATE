from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
FRAME_COLORS = (1, 3, 4, 6)
CONTENT_COLORS = (2, 7)
BG = 0


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _flood_from_side(empty_mask: np.ndarray, side: str) -> set[tuple[int, int]]:
    h, w = empty_mask.shape
    if side == "U":
        seeds = [(0, c) for c in range(w) if empty_mask[0, c]]
    elif side == "D":
        seeds = [(h - 1, c) for c in range(w) if empty_mask[h - 1, c]]
    elif side == "L":
        seeds = [(r, 0) for r in range(h) if empty_mask[r, 0]]
    else:
        seeds = [(r, w - 1) for r in range(h) if empty_mask[r, w - 1]]

    reached = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS.values():
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and empty_mask[nr, nc] and (nr, nc) not in reached:
                reached.add((nr, nc))
                stack.append((nr, nc))
    return reached


def _expected_body_mask(height: int, width: int, tail: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if tail == "U":
        mask[0, :] = True
        mask[1:, 0] = True
        mask[1:, -1] = True
    elif tail == "D":
        mask[-1, :] = True
        mask[:-1, 0] = True
        mask[:-1, -1] = True
    elif tail == "L":
        mask[:, 0] = True
        mask[0, 1:] = True
        mask[-1, 1:] = True
    else:
        mask[:, -1] = True
        mask[0, :-1] = True
        mask[-1, :-1] = True
    return mask


def _detect_cup(grid: np.ndarray, color: int, coords: np.ndarray, bg: int) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros(grid.shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True

    edge_counts = {
        "U": int(mask[r0, c0 : c1 + 1].sum()),
        "D": int(mask[r1, c0 : c1 + 1].sum()),
        "L": int(mask[r0 : r1 + 1, c0].sum()),
        "R": int(mask[r0 : r1 + 1, c1].sum()),
    }
    tails = [side for side, count in edge_counts.items() if count == 1]
    if len(tails) != 1:
        return None
    tail = tails[0]

    spike: list[tuple[int, int]] = []
    if tail == "U":
        col = None
        rr = r0
        while rr <= r1:
            cols = np.flatnonzero(mask[rr, c0 : c1 + 1]) + c0
            if len(cols) != 1:
                break
            if col is None:
                col = int(cols[0])
            elif int(cols[0]) != col:
                return None
            spike.append((int(rr), col))
            rr += 1
        if not spike or rr > r1:
            return None
        body_bbox = (rr, c0, r1, c1)
        tip = spike[0]
    elif tail == "D":
        col = None
        rr = r1
        while rr >= r0:
            cols = np.flatnonzero(mask[rr, c0 : c1 + 1]) + c0
            if len(cols) != 1:
                break
            if col is None:
                col = int(cols[0])
            elif int(cols[0]) != col:
                return None
            spike.append((int(rr), col))
            rr -= 1
        if not spike or rr < r0:
            return None
        body_bbox = (r0, c0, rr, c1)
        tip = spike[0]
    elif tail == "L":
        row = None
        cc = c0
        while cc <= c1:
            rows = np.flatnonzero(mask[r0 : r1 + 1, cc]) + r0
            if len(rows) != 1:
                break
            if row is None:
                row = int(rows[0])
            elif int(rows[0]) != row:
                return None
            spike.append((row, int(cc)))
            cc += 1
        if not spike or cc > c1:
            return None
        body_bbox = (r0, cc, r1, c1)
        tip = spike[0]
    else:
        row = None
        cc = c1
        while cc >= c0:
            rows = np.flatnonzero(mask[r0 : r1 + 1, cc]) + r0
            if len(rows) != 1:
                break
            if row is None:
                row = int(rows[0])
            elif int(rows[0]) != row:
                return None
            spike.append((row, int(cc)))
            cc -= 1
        if not spike or cc < c0:
            return None
        body_bbox = (r0, c0, r1, cc)
        tip = spike[0]

    br0, bc0, br1, bc1 = body_bbox
    if br1 - br0 + 1 < 3 or bc1 - bc0 + 1 < 3:
        return None

    submask = mask[br0 : br1 + 1, bc0 : bc1 + 1]
    if not np.array_equal(submask, _expected_body_mask(submask.shape[0], submask.shape[1], tail)):
        return None

    empty = ~submask
    cavity_rel = _flood_from_side(empty, OPPOSITE[tail])
    if not cavity_rel:
        return None
    cavity_abs = {(br0 + r, bc0 + c) for r, c in cavity_rel}

    content_cells = []
    content_colors = set()
    for rr, cc in cavity_abs:
        value = int(grid[rr, cc])
        if value != bg and value != color:
            content_cells.append((int(rr), int(cc)))
            content_colors.add(value)
    if len(content_colors) != 1 or not content_cells:
        return None

    return {
        "tail": tail,
        "tip": tip,
        "border_color": int(color),
        "content_color": int(next(iter(content_colors))),
        "content_cells": sorted(content_cells),
        "border_cells": [tuple(map(int, rc)) for rc in coords.tolist()],
    }


def _extract_cups(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    cups = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        for coords in _components(grid == color):
            if coords.shape[0] < 7:
                continue
            cup = _detect_cup(grid, color, coords, bg)
            if cup is not None:
                cups.append(cup)
    return cups


def _settle_emitted(base: np.ndarray, emitted: list[tuple[int, int, int]], bg: int) -> np.ndarray:
    out = base.copy()
    by_col: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c, color in emitted:
        by_col[int(c)].append((int(r), int(color)))

    h, _w = out.shape
    for c, cells in by_col.items():
        cells.sort(key=lambda item: item[0], reverse=True)
        landing = h - 1
        for _src_r, color in cells:
            while landing >= 0 and int(out[landing, c]) != bg:
                landing -= 1
            if landing < 0:
                break
            out[landing, c] = int(color)
            landing -= 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject each cup's contents along its tail, erase the cups, then let the emitted pixels fall and stack at the bottom.

    Input structure:
    - The background is black.
    - The grid contains 2 to 4 monochrome cup frames in distinct frame colors; each cup is a three-sided rectangular frame with a 1-2 cell tail spike on the closed side.
    - Each cup contains 2 to 4 red or orange content pixels inside the open mouth cavity, and different cups may face different directions.

    Rule:
    - Detect each cup's tail direction from the unique spike tip side.
    - Count the content pixels inside that cup and erase both those pixels and the entire cup frame.
    - Starting from the spike tip, place one emitted pixel per erased content pixel on consecutive cells extending outward in the tail direction; off-grid emitted cells are discarded.
    - After all cups eject, let every emitted pixel fall straight down independently until blocked by the bottom edge or by a previously settled pixel in the same column.

    Key insight:
    - The tail direction does not determine the final row of a pixel; it only determines the release column pattern, and gravity converts that release pattern into bottom stacks.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    cups = _extract_cups(g)

    out = g.copy()
    emitted: list[tuple[int, int, int]] = []
    h, w = g.shape
    for cup in cups:
        for r, c in cup["border_cells"]:
            out[r, c] = bg
        for r, c in cup["content_cells"]:
            out[r, c] = bg

        dr, dc = DIRS[cup["tail"]]
        tip_r, tip_c = cup["tip"]
        for step in range(1, len(cup["content_cells"]) + 1):
            rr = tip_r + dr * step
            cc = tip_c + dc * step
            if 0 <= rr < h and 0 <= cc < w:
                emitted.append((int(rr), int(cc), int(cup["content_color"])))

    return _settle_emitted(out, emitted, bg)


def _rotate_coords(coords: list[tuple[int, int]], shape: tuple[int, int], k: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    out = [(int(r), int(c)) for r, c in coords]
    h, w = map(int, shape)
    for _ in range(k % 4):
        out = [(w - 1 - c, r) for r, c in out]
        h, w = w, h
    return out, (h, w)


def _build_tail_up(
    rng: np.random.Generator,
    border_color: int,
    content_color: int,
    inner_h: int,
    inner_w: int,
    spike_len: int,
    content_count: int,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    patch = np.zeros((spike_len + inner_h + 1, inner_w + 2), dtype=int)
    top = spike_len
    patch[top, :] = border_color
    patch[top + 1 :, 0] = border_color
    patch[top + 1 :, -1] = border_color
    tip_c = patch.shape[1] // 2
    patch[:spike_len, tip_c] = border_color

    cavity = [(top + 1 + rr, 1 + cc) for rr in range(inner_h) for cc in range(inner_w)]
    chosen_idx = rng.choice(len(cavity), size=content_count, replace=False)
    for idx in chosen_idx:
        rr, cc = cavity[int(idx)]
        patch[rr, cc] = content_color
    return patch, (0, tip_c), cavity


def _build_cup_spec(
    rng: np.random.Generator,
    border_color: int,
    content_color: int,
    direction: str,
    *,
    force_count: int | None = None,
) -> dict:
    content_count = int(force_count if force_count is not None else rng.integers(2, 5))
    inner_h = int(rng.integers(2, 4))
    inner_w = int(rng.integers(2, 4))
    while inner_h * inner_w < content_count:
        if inner_h <= inner_w:
            inner_h += 1
        else:
            inner_w += 1
    spike_len = int(rng.integers(1, 3))

    patch, tip_rel, cavity = _build_tail_up(
        rng, border_color, content_color, inner_h, inner_w, spike_len, content_count
    )
    rot_k = {"U": 0, "L": 1, "D": 2, "R": 3}[direction]
    rotated_patch = np.rot90(patch, k=rot_k).copy()
    rel_coords = [tuple(map(int, rc)) for rc in np.argwhere(patch != 0)]
    rel_coords, shape = _rotate_coords(rel_coords, patch.shape, rot_k)
    tip_list, _shape2 = _rotate_coords([tip_rel], patch.shape, rot_k)
    cavity_rot, _shape3 = _rotate_coords(cavity, patch.shape, rot_k)
    assert shape == rotated_patch.shape == _shape2 == _shape3

    content_rel = [tuple(map(int, rc)) for rc in np.argwhere(rotated_patch == content_color)]
    patch_bbox = (0, 0, rotated_patch.shape[0] - 1, rotated_patch.shape[1] - 1)
    return {
        "direction": direction,
        "border_color": int(border_color),
        "content_color": int(content_color),
        "content_count": int(content_count),
        "patch": rotated_patch,
        "tip_rel": tip_list[0],
        "content_rel": sorted(content_rel),
        "bbox_rel": patch_bbox,
        "shape": rotated_patch.shape,
        "cavity_rel": cavity_rot,
    }


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _shift_box(box: tuple[int, int, int, int], r0: int, c0: int) -> tuple[int, int, int, int]:
    br0, bc0, br1, bc1 = box
    return (r0 + br0, c0 + bc0, r0 + br1, c0 + bc1)


def _placement_emitted(spec: dict, r0: int, c0: int) -> list[tuple[int, int]]:
    dr, dc = DIRS[spec["direction"]]
    tip_r = r0 + spec["tip_rel"][0]
    tip_c = c0 + spec["tip_rel"][1]
    return [(tip_r + dr * step, tip_c + dc * step) for step in range(1, spec["content_count"] + 1)]


def _placement_content(spec: dict, r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + rr, c0 + cc) for rr, cc in spec["content_rel"]]


def _placement_ok(
    spec: dict,
    r0: int,
    c0: int,
    h: int,
    w: int,
    placed_boxes: list[tuple[int, int, int, int]],
    reserved_emits: set[tuple[int, int]],
) -> bool:
    ph, pw = spec["shape"]
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False
    if r0 + ph > h - 4:
        return False

    box = _shift_box(spec["bbox_rel"], r0, c0)
    if any(_boxes_overlap(box, other, gap=1) for other in placed_boxes):
        return False

    emitted = _placement_emitted(spec, r0, c0)
    for rr, cc in emitted:
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if (rr, cc) in reserved_emits:
            return False
        if any(br0 <= rr <= br1 and bc0 <= cc <= bc1 for br0, bc0, br1, bc1 in placed_boxes):
            return False
        if box[0] <= rr <= box[2] and box[1] <= cc <= box[3]:
            return False

    for rr, cc in reserved_emits:
        if box[0] <= rr <= box[2] and box[1] <= cc <= box[3]:
            return False
    return True


def _candidate_rows(spec: dict, h: int) -> list[int]:
    ph = spec["shape"][0]
    upper = max(0, h - ph - 4)
    rows = list(range(0, upper + 1))
    random.shuffle(rows)
    return rows


def _place_specs(rng: np.random.Generator, h: int, w: int, specs: list[dict], shared_col: int) -> list[dict] | None:
    placed: list[dict] = []
    placed_boxes: list[tuple[int, int, int, int]] = []
    reserved_emits: set[tuple[int, int]] = set()

    for idx, spec in enumerate(specs):
        candidates: list[tuple[int, int]] = []
        ph, pw = spec["shape"]
        if idx == 0:
            c0 = shared_col - spec["tip_rel"][1]
            for r0 in _candidate_rows(spec, h):
                candidates.append((r0, c0))
        elif idx == 1:
            if spec["direction"] == "R":
                share_step = int(rng.integers(1, spec["content_count"] + 1))
                tip_abs_c = shared_col - share_step
            else:
                share_step = int(rng.integers(1, spec["content_count"] + 1))
                tip_abs_c = shared_col + share_step
            c0 = tip_abs_c - spec["tip_rel"][1]
            for r0 in _candidate_rows(spec, h):
                candidates.append((r0, c0))
        else:
            cols = list(range(0, w - pw + 1))
            rows = _candidate_rows(spec, h)
            rng.shuffle(cols)
            for r0 in rows:
                for c0 in cols:
                    candidates.append((r0, c0))

        rng.shuffle(candidates)
        chosen = None
        for r0, c0 in candidates:
            if _placement_ok(spec, r0, c0, h, w, placed_boxes, reserved_emits):
                chosen = (r0, c0)
                break
        if chosen is None:
            return None

        r0, c0 = chosen
        box = _shift_box(spec["bbox_rel"], r0, c0)
        emitted = _placement_emitted(spec, r0, c0)
        placed_boxes.append(box)
        reserved_emits.update(emitted)
        placed.append(
            {
                **spec,
                "top_left": (r0, c0),
                "bbox_abs": box,
                "tip_abs": (r0 + spec["tip_rel"][0], c0 + spec["tip_rel"][1]),
                "content_abs": _placement_content(spec, r0, c0),
                "emitted_abs": emitted,
            }
        )
    return placed


def _compose_grid(h: int, w: int, placed: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for spec in placed:
        r0, c0 = spec["top_left"]
        patch = spec["patch"]
        ph, pw = patch.shape
        sub = grid[r0 : r0 + ph, c0 : c0 + pw]
        sub[patch != 0] = patch[patch != 0]
    return grid


def _expected_from_metadata(h: int, w: int, placed: list[dict]) -> np.ndarray:
    emitted = []
    for spec in placed:
        for r, c in spec["emitted_abs"]:
            emitted.append((int(r), int(c), int(spec["content_color"])))
    return _settle_emitted(np.zeros((h, w), dtype=int), emitted, BG)


def _naive_drop_original(h: int, w: int, placed: list[dict]) -> np.ndarray:
    original = []
    for spec in placed:
        for r, c in spec["content_abs"]:
            original.append((int(r), int(c), int(spec["content_color"])))
    return _settle_emitted(np.zeros((h, w), dtype=int), original, BG)


def _naive_single_column(h: int, w: int, placed: list[dict]) -> np.ndarray:
    emitted = []
    for spec in placed:
        tip_r, tip_c = spec["tip_abs"]
        for step in range(1, spec["content_count"] + 1):
            emitted.append((int(tip_r), int(tip_c), int(spec["content_color"])))
    return _settle_emitted(np.zeros((h, w), dtype=int), emitted, BG)


def _shared_columns(placed: list[dict]) -> Counter:
    cols = Counter()
    owners: dict[int, set[int]] = defaultdict(set)
    for idx, spec in enumerate(placed):
        for _r, c in spec["emitted_abs"]:
            cols[int(c)] += 1
            owners[int(c)].add(idx)
    return Counter({c: cols[c] for c in cols if len(owners[c]) >= 2})


def _valid_metadata(grid: np.ndarray, placed: list[dict], expected: np.ndarray) -> bool:
    directions = {spec["direction"] for spec in placed}
    if len(directions) < 2:
        return False
    if not any(d in {"U", "D"} for d in directions):
        return False
    if not any(d in {"L", "R"} for d in directions):
        return False

    shared = _shared_columns(placed)
    if not shared:
        return False

    if np.array_equal(expected, _naive_drop_original(*grid.shape, placed)):
        return False
    if np.array_equal(expected, _naive_single_column(*grid.shape, placed)):
        return False

    cups = _extract_cups(grid)
    if len(cups) != len(placed):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    return True


def _make_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(17, 23))
    w = int(rng.integers(17, 23))
    n = int(rng.integers(2, 5))

    frame_colors = list(rng.choice(FRAME_COLORS, size=n, replace=False))
    shared_col = int(rng.integers(4, w - 4))

    first_vertical = random.choice(["U", "D"])
    first_horizontal = random.choice(["L", "R"])
    remaining_dirs = [random.choice(["U", "D", "L", "R"]) for _ in range(max(0, n - 2))]
    directions = [first_vertical, first_horizontal, *remaining_dirs]
    if n > 2:
        tail_dirs = directions[2:]
        rng.shuffle(tail_dirs)
        directions[2:] = tail_dirs

    first_color, second_color = CONTENT_COLORS
    other_colors = [int(rng.choice(CONTENT_COLORS)) for _ in range(max(0, n - 2))]
    content_colors = [first_color, second_color, *other_colors]

    specs = []
    for idx in range(n):
        force_count = None
        if idx == 1 and directions[idx] in {"L", "R"}:
            force_count = int(rng.integers(2, 5))
        specs.append(
            _build_cup_spec(
                rng,
                frame_colors[idx],
                content_colors[idx],
                directions[idx],
                force_count=force_count,
            )
        )

    placed = _place_specs(rng, h, w, specs, shared_col)
    if placed is None:
        return None

    grid = _compose_grid(h, w, placed)
    expected = _expected_from_metadata(h, w, placed)
    if not _valid_metadata(grid, placed, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid eject-and-settle puzzle")
