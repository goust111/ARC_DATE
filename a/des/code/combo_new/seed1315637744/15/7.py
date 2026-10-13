import random

import numpy as np


BG_COLOR = 0
BAND_COLOR = 7
OBJECT_COLORS = [1, 2, 3, 4, 5, 6, 8, 9]
MAX_GENERATION_ATTEMPTS = 800


def _most_common(arr: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(arr, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _detect_band_color(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    samples = np.concatenate(
        [
            grid[1, 1:w - 1],
            grid[h - 2, 1:w - 1],
            grid[1:h - 1, 1],
            grid[1:h - 1, w - 2],
        ]
    )
    samples = samples[samples != bg]
    assert samples.size > 0, "expected a non-background border band"
    return _most_common(samples)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs = []
    i = 0
    n = int(mask.size)
    while i < n:
        if not bool(mask[i]):
            i += 1
            continue
        j = i
        while j < n and bool(mask[j]):
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def _extract_windows(grid: np.ndarray, band_color: int) -> list[dict]:
    h, w = grid.shape
    windows: list[dict] = []
    next_id = 0

    def add_horizontal(side: str, row: int):
        nonlocal next_id
        arr = grid[row, 2:w - 2]
        for start0, end0 in _runs(arr != band_color):
            start = 2 + start0
            end = 2 + end0
            windows.append(
                {
                    "id": next_id,
                    "side": side,
                    "orientation": "h",
                    "start": start,
                    "end": end,
                    "length": end - start + 1,
                    "center": (start + end) / 2.0,
                }
            )
            next_id += 1

    def add_vertical(side: str, col: int):
        nonlocal next_id
        arr = grid[2:h - 2, col]
        for start0, end0 in _runs(arr != band_color):
            start = 2 + start0
            end = 2 + end0
            windows.append(
                {
                    "id": next_id,
                    "side": side,
                    "orientation": "v",
                    "start": start,
                    "end": end,
                    "length": end - start + 1,
                    "center": (start + end) / 2.0,
                }
            )
            next_id += 1

    add_horizontal("top", 1)
    add_horizontal("bottom", h - 2)
    add_vertical("left", 1)
    add_vertical("right", w - 2)
    return windows


def _extract_objects(grid: np.ndarray, bg: int, band_color: int) -> list[dict]:
    h, w = grid.shape
    mask = (grid != bg) & (grid != band_color)
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if bool(seen[nr, nc]) or not bool(mask[nr, nc]):
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            height = int(r1 - r0 + 1)
            width = int(c1 - c0 + 1)
            assert height == 1 or width == 1, "foreground object must be a 1-pixel-thick bar"
            orientation = "h" if height == 1 else "v"
            length = width if orientation == "h" else height
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "orientation": orientation,
                    "length": int(length),
                }
            )

    return objects


def _window_sort_key(obj: dict, window: dict, h: int, w: int) -> tuple[float, float, int, int]:
    if obj["orientation"] == "h":
        band_coord = 1 if window["side"] == "top" else h - 2
        side_dist = abs(obj["bbox"][0] - band_coord)
        obj_center = (obj["bbox"][1] + obj["bbox"][3]) / 2.0
        side_rank = 0 if window["side"] == "top" else 1
    else:
        band_coord = 1 if window["side"] == "left" else w - 2
        side_dist = abs(obj["bbox"][1] - band_coord)
        obj_center = (obj["bbox"][0] + obj["bbox"][2]) / 2.0
        side_rank = 0 if window["side"] == "left" else 1

    parallel_dist = abs(obj_center - window["center"])
    return (float(side_dist), float(parallel_dist), side_rank, int(window["start"]))


def main(grid: np.ndarray) -> np.ndarray:
    """Move each interior bar into the border-band window whose orientation and length match it.

    Input structure:
    - The background is black or otherwise the dominant color.
    - A single shared band color forms one-pixel-wide border strips on the four inner edges of the grid, with one or more contiguous missing window segments on each strip.
    - Inside the grid are several monochrome solid bars of thickness 1; each bar is horizontal or vertical and does not touch the border strips.

    Rule:
    - Detect the four inner border strips and all of their missing window segments.
    - For each horizontal bar, consider only the top and bottom windows of the same length; for each vertical bar, consider only the left and right windows of the same length.
    - Move the bar to the best matching window, preferring the nearer candidate strip when the same length appears on both opposite sides, and recolor that window segment with the bar's color while clearing the original bar position.

    Key insight:
    - The target is encoded by three visible cues at once: the bar orientation restricts the candidate sides, the window length determines the geometric match, and the bar's distance to the opposite strips resolves which same-length window should be filled.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common(g)
    band_color = _detect_band_color(g, bg)
    windows = _extract_windows(g, band_color)
    objects = _extract_objects(g, bg, band_color)

    out = g.copy()
    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    used_windows: set[int] = set()
    for obj in objects:
        candidates = [
            window
            for window in windows
            if window["orientation"] == obj["orientation"] and window["length"] == obj["length"]
        ]
        assert candidates, "every object must match at least one window"
        ranked = sorted(candidates, key=lambda window: _window_sort_key(obj, window, h, w))
        chosen = None
        for window in ranked:
            if window["id"] not in used_windows:
                chosen = window
                break
        assert chosen is not None, "matching windows must be one-to-one"
        used_windows.add(int(chosen["id"]))

        if chosen["orientation"] == "h":
            row = 1 if chosen["side"] == "top" else h - 2
            out[row, chosen["start"] : chosen["end"] + 1] = obj["color"]
        else:
            col = 1 if chosen["side"] == "left" else w - 2
            out[chosen["start"] : chosen["end"] + 1, col] = obj["color"]

    return out


def _random_composition(total: int, parts: int) -> list[int]:
    pieces = [0] * parts
    for _ in range(total):
        pieces[random.randrange(parts)] += 1
    return pieces


def _place_side_windows(specs: list[dict], span_start: int, span_end: int) -> bool:
    if not specs:
        return True
    random.shuffle(specs)
    span = span_end - span_start + 1
    required = sum(int(spec["length"]) for spec in specs) + len(specs) - 1
    if required > span:
        return False
    extra = span - required
    gaps = _random_composition(extra, len(specs) + 1)
    pos = span_start + gaps[0]
    for i, spec in enumerate(specs):
        spec["start"] = int(pos)
        spec["end"] = int(pos + spec["length"] - 1)
        pos = spec["end"] + 1
        if i < len(specs) - 1:
            pos += 1 + gaps[i + 1]
    return True


def _can_place(occupied: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    h, w = occupied.shape
    r0, c0, r1, c1 = bbox
    pr0 = max(0, r0 - 1)
    pc0 = max(0, c0 - 1)
    pr1 = min(h - 1, r1 + 1)
    pc1 = min(w - 1, c1 + 1)
    return not bool(occupied[pr0 : pr1 + 1, pc0 : pc1 + 1].any())


def _mark_occupied(occupied: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    h, w = occupied.shape
    r0, c0, r1, c1 = bbox
    pr0 = max(0, r0 - 1)
    pc0 = max(0, c0 - 1)
    pr1 = min(h - 1, r1 + 1)
    pc1 = min(w - 1, c1 + 1)
    occupied[pr0 : pr1 + 1, pc0 : pc1 + 1] = True


def _place_object(object_spec: dict, occupied: np.ndarray, h: int, w: int) -> tuple[int, int, int, int] | None:
    length = int(object_spec["length"])
    orientation = object_spec["orientation"]
    side = object_spec["target_side"]
    candidates = []

    if orientation == "h":
        if side == "top":
            rows = range(3, (h - 2) // 2 + 1)
        else:
            rows = range(h // 2, h - 3)
        cols = range(3, w - 4 - length + 2)
        for row in rows:
            for col in cols:
                candidates.append((row, col, row, col + length - 1))
    else:
        rows = range(3, h - 4 - length + 2)
        if side == "left":
            cols = range(3, (w - 2) // 2 + 1)
        else:
            cols = range(w // 2, w - 3)
        for row in rows:
            for col in cols:
                candidates.append((row, col, row + length - 1, col))

    random.shuffle(candidates)
    for bbox in candidates:
        if _can_place(occupied, bbox):
            _mark_occupied(occupied, bbox)
            return bbox
    return None


def _window_cell_count(specs: list[dict]) -> int:
    if not specs:
        return 0
    return sum(int(spec["length"]) for spec in specs) + len(specs) - 1


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 28)
    w = random.randint(16, 28)
    h_lengths = list(range(2, min(7, w - 6) + 1))
    v_lengths = list(range(2, min(7, h - 6) + 1))
    if len(h_lengths) < 4 or len(v_lengths) < 4:
        return None

    n_h = random.randint(1, min(3, len(h_lengths) - 2))
    n_v = random.randint(1, min(3, len(v_lengths) - 2))
    chosen_h = random.sample(h_lengths, n_h)
    chosen_v = random.sample(v_lengths, n_v)

    object_specs = []
    for length in chosen_h:
        object_specs.append(
            {
                "orientation": "h",
                "length": int(length),
                "target_side": random.choice(["top", "bottom"]),
                "dual": False,
            }
        )
    for length in chosen_v:
        object_specs.append(
            {
                "orientation": "v",
                "length": int(length),
                "target_side": random.choice(["left", "right"]),
                "dual": False,
            }
        )

    horizontal_specs = [spec for spec in object_specs if spec["orientation"] == "h"]
    vertical_specs = [spec for spec in object_specs if spec["orientation"] == "v"]
    if horizontal_specs and random.random() < 0.5:
        random.choice(horizontal_specs)["dual"] = True
    if vertical_specs and random.random() < 0.5:
        random.choice(vertical_specs)["dual"] = True
    for spec in object_specs:
        if not spec["dual"] and random.random() < 0.3:
            spec["dual"] = True

    colors = random.sample(OBJECT_COLORS, len(object_specs))
    for idx, spec in enumerate(object_specs):
        spec["id"] = idx
        spec["color"] = int(colors[idx])

    side_specs = {side: [] for side in ("top", "bottom", "left", "right")}
    used_h_lengths = set()
    used_v_lengths = set()

    for spec in object_specs:
        orientation = spec["orientation"]
        side = spec["target_side"]
        length = int(spec["length"])
        record = {"length": length, "used_by": spec["id"]}
        if orientation == "h":
            used_h_lengths.add(length)
            if spec["dual"]:
                side_specs["top"].append(dict(record) if side == "top" else {"length": length, "used_by": None})
                side_specs["bottom"].append(dict(record) if side == "bottom" else {"length": length, "used_by": None})
            else:
                side_specs[side].append(record)
        else:
            used_v_lengths.add(length)
            if spec["dual"]:
                side_specs["left"].append(dict(record) if side == "left" else {"length": length, "used_by": None})
                side_specs["right"].append(dict(record) if side == "right" else {"length": length, "used_by": None})
            else:
                side_specs[side].append(record)

    remaining_h = [length for length in h_lengths if length not in used_h_lengths]
    remaining_v = [length for length in v_lengths if length not in used_v_lengths]
    random.shuffle(remaining_h)
    random.shuffle(remaining_v)

    for side in ("top", "bottom"):
        if not side_specs[side]:
            if not remaining_h:
                return None
            side_specs[side].append({"length": int(remaining_h.pop()), "used_by": None})
    for side in ("left", "right"):
        if not side_specs[side]:
            if not remaining_v:
                return None
            side_specs[side].append({"length": int(remaining_v.pop()), "used_by": None})

    for _ in range(random.randint(0, 1)):
        if remaining_h:
            side_specs[random.choice(["top", "bottom"])].append(
                {"length": int(remaining_h.pop()), "used_by": None}
            )
        if remaining_v:
            side_specs[random.choice(["left", "right"])].append(
                {"length": int(remaining_v.pop()), "used_by": None}
            )

    if _window_cell_count(side_specs["top"]) > w - 4:
        return None
    if _window_cell_count(side_specs["bottom"]) > w - 4:
        return None
    if _window_cell_count(side_specs["left"]) > h - 4:
        return None
    if _window_cell_count(side_specs["right"]) > h - 4:
        return None

    if not _place_side_windows(side_specs["top"], 2, w - 3):
        return None
    if not _place_side_windows(side_specs["bottom"], 2, w - 3):
        return None
    if not _place_side_windows(side_specs["left"], 2, h - 3):
        return None
    if not _place_side_windows(side_specs["right"], 2, h - 3):
        return None

    matched_windows: dict[int, dict] = {}
    for side, specs in side_specs.items():
        for spec in specs:
            spec["side"] = side
            if spec["used_by"] is not None:
                matched_windows[int(spec["used_by"])] = spec
    if len(matched_windows) != len(object_specs):
        return None

    occupied = np.zeros((h, w), dtype=bool)
    placed_objects = []
    for spec in sorted(object_specs, key=lambda item: (-int(item["length"]), item["orientation"])):
        bbox = _place_object(spec, occupied, h, w)
        if bbox is None:
            return None
        placed = dict(spec)
        placed["bbox"] = bbox
        placed_objects.append(placed)

    grid = np.full((h, w), BG_COLOR, dtype=int)
    grid[1, 1:w - 1] = BAND_COLOR
    grid[h - 2, 1:w - 1] = BAND_COLOR
    grid[1:h - 1, 1] = BAND_COLOR
    grid[1:h - 1, w - 2] = BAND_COLOR

    for spec in side_specs["top"]:
        grid[1, spec["start"] : spec["end"] + 1] = BG_COLOR
    for spec in side_specs["bottom"]:
        grid[h - 2, spec["start"] : spec["end"] + 1] = BG_COLOR
    for spec in side_specs["left"]:
        grid[spec["start"] : spec["end"] + 1, 1] = BG_COLOR
    for spec in side_specs["right"]:
        grid[spec["start"] : spec["end"] + 1, w - 2] = BG_COLOR

    for spec in placed_objects:
        r0, c0, r1, c1 = spec["bbox"]
        if spec["orientation"] == "h":
            grid[r0, c0 : c1 + 1] = spec["color"]
        else:
            grid[r0 : r1 + 1, c0] = spec["color"]

    out = grid.copy()
    for spec in placed_objects:
        r0, c0, r1, c1 = spec["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG_COLOR
        window = matched_windows[int(spec["id"])]
        if spec["orientation"] == "h":
            row = 1 if window["side"] == "top" else h - 2
            out[row, window["start"] : window["end"] + 1] = spec["color"]
        else:
            col = 1 if window["side"] == "left" else w - 2
            out[window["start"] : window["end"] + 1, col] = spec["color"]

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
